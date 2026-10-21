import json
import math

import httpx
import numpy as np
import pytest

from app.core.config import QwenSettings
from app.core.schema import Area, Source
from app.extraction.readers.paddle import reading_from_result
from app.extraction.readers.prompt import QwenOutput, json_schema
from app.extraction.readers.qwen import QwenVLReader, extract_json, reading_from_output
from app.extraction.token_confidence import value_confidences, value_spans

PADDLE_RESULT = {
    "width": 1654,
    "height": 2339,
    "doc_preprocessor_res": {"angle": 180},
    "parsing_res_list": [
        {
            "block_label": "doc_title",
            "block_content": "PLAN DE BORNAGE",
            "block_bbox": [100, 50, 900, 120],
            "block_order": 1,
        },
        {
            "block_label": "text",
            "block_content": "Titre Foncier n° 1738/DK   NICAD : 0123 4567 8901 2",
            "block_bbox": [100, 150, 1200, 200],
            "block_order": 2,
        },
        {
            "block_label": "table",
            "block_bbox": [100, 300, 1500, 700],
            "block_order": 3,
            "block_content": "<table><tr><td>Borne</td><td>X</td><td>Y</td></tr>"
            "<tr><td>B1</td><td>381210.57</td><td>1564042.96</td></tr>"
            "<tr><td>B2</td><td>381225.56</td><td>1564034.94</td></tr></table>",
        },
        {
            "block_label": "text",
            "block_content": "Superficie : 00 ha 06 a 49 ca",
            "block_bbox": [100, 750, 800, 800],
            "block_order": 4,
        },
        {
            "block_label": "seal",
            "block_content": "CADASTRE DAKAR",
            "block_bbox": [1200, 1900, 1500, 2200],
            "block_order": None,
        },
    ],
}


def test_lecture_paddle():
    r = reading_from_result(PADDLE_RESULT)
    assert r.source is Source.PADDLE_VL and r.orientation_deg == 180
    assert [(p.label, p.x, p.y) for p in r.points] == [
        ("B1", "381210.57", "1564042.96"),
        ("B2", "381225.56", "1564034.94"),
    ]
    assert r.points[0].bbox.y0 == 300
    assert r.field("title_number").raw.endswith("1738/DK")
    assert r.field("nicad").raw == "0123 4567 8901 2"
    assert "06 a 49 ca" in r.field("area").raw
    assert r.field("owner").raw is None  # PaddleOCR-VL ne sait pas qui est le propriétaire


def test_paddle_sans_tableau_repli_sur_le_texte():
    data = {
        "parsing_res_list": [
            {
                "block_label": "text",
                "block_content": "B1 381210.57 1564042.96\nB2 381225.56 1564034.94",
            }
        ]
    }
    assert len(reading_from_result(data).points) == 2


def test_positions_des_valeurs_json():
    text = '{"a": "x1", "b": [ {"c": null}, 12.5 ], "d": {"e": "é"}}'
    spans = value_spans(text)
    assert text[slice(*spans[("a",)])] == "x1"
    assert text[slice(*spans[("b", 0, "c")])] == "null"
    assert text[slice(*spans[("b", 1)])] == "12.5"
    assert text[slice(*spans[("d", "e")])] == "é"


def test_confiance_par_valeur_est_le_token_le_moins_sur():
    tokens = [
        ('{"x": "', 0.0),
        ("3812", math.log(0.99)),
        ("10.", math.log(0.6)),
        ("57", 0.0),
        ('"}', 0.0),
    ]
    text = "".join(t for t, _ in tokens)
    assert value_confidences(text, tokens)[("x",)] == pytest.approx(0.6)
    assert value_confidences(text + " ", tokens) == {}  # tokens incohérents : pas de confiance


def test_schema_impose_interdit_les_champs_inconnus():
    schema = json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) >= {"points", "title_number", "area"}


QWEN_JSON = {
    "document_type": "plan_bornage",
    "title_number": "TF 1738/DK",
    "nicad": None,
    "mother_title": None,
    "lot": None,
    "subdivision": None,
    "requisition": None,
    "land_registry": None,
    "region": None,
    "department": None,
    "arrondissement": None,
    "coordinate_system": None,
    "scale": None,
    "certified_on": None,
    "disclaimer": None,
    "owner": "Awa NDIAYE",
    "commune": "Dakar, Médina",
    "area": {"text": "00 ha 06 a 49 ca", "ha": 0, "a": 6, "ca": 49},
    "points": [{"label": "B1", "x": "381210.57", "y": "1564042.96"}],
    "sides": [{"from_label": "B1", "to_label": "B2", "length": "17.00"}],
}


def test_lecture_qwen():
    r = reading_from_output(QwenOutput.model_validate(QWEN_JSON), {("points", 0, "x"): 0.8}, "qwen")
    assert r.points[0].x_confidence == 0.8 and r.points[0].y_confidence is None
    assert r.field("area").raw == "00 ha 06 a 49 ca"
    assert r.sides[0].from_label == "B1"


def test_json_entoure_de_texte():
    data, start = extract_json('```json\n{"a": 1}\n```')
    assert data == {"a": 1} and start == 8


def test_client_llama_server_envoie_le_schema_et_lit_les_logprobs():
    content = json.dumps(QWEN_JSON, ensure_ascii=False)
    # Tokens qui recomposent exactement la réponse ; un token hésitant dans « 381210.57 ».
    i = content.index("381210.57")
    tokens = [
        {"token": content[:i], "logprob": 0.0},
        {"token": "381210", "logprob": math.log(0.7)},
        {"token": content[i + 6 :], "logprob": 0.0},
    ]
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}, "logprobs": {"content": tokens}}]},
        )

    reader = QwenVLReader(
        QwenSettings(server_url="http://gpu:8000/v1"),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    r = reader.read(np.full((40, 60, 3), 255, np.uint8))
    body = seen["body"]
    assert seen["url"] == "http://gpu:8000/v1/chat/completions"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["temperature"] == 0.0 and body["logprobs"] is True
    assert body["messages"][1]["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert r.points[0].x == "381210.57"
    assert r.points[0].x_confidence == pytest.approx(0.7)
    assert r.points[0].y_confidence == pytest.approx(1.0)
    assert r.field("area").area == Area(ha=0, a=6, ca=49)


# ------------------------------------------- le prompt ne doit rien souffler au modele


def test_le_prompt_ne_nomme_aucun_systeme_de_coordonnees():
    """Qwen a rendu « UTM zone 28N » sur deux documents qui n'impriment aucun systeme.

    C'etait la phrase du prompt — « Les coordonnees sont en UTM zone 28N » — qu'il nous
    renvoyait. Un nom de systeme ecrit dans la consigne est un nom que le modele peut
    recopier comme s'il l'avait lu. Aucun ne doit y figurer, pas meme en exemple.
    """
    from app.extraction.readers.prompt import SYSTEM_PROMPT, USER_PROMPT

    texte = (SYSTEM_PROMPT + USER_PROMPT).upper()
    interdits = ["UTM", "WGS", "EPSG", "RRS", "RSS", "ITRF", "GRS80", "ZONE 28", "LAMBERT"]
    trouves = [m for m in interdits if m in texte]
    assert not trouves, f"valeurs soufflees au modele : {trouves}"


def test_le_prompt_exige_null_sur_une_valeur_masquee():
    from app.extraction.readers.prompt import USER_PROMPT

    assert "cachet" in USER_PROMPT and "null" in USER_PROMPT


def test_le_schema_impose_reste_complet():
    """Retirer des exemples du prompt ne doit pas retirer de champs."""
    from app.core.schema import TEXT_FIELDS
    from app.extraction.readers.prompt import json_schema

    props = json_schema()["properties"]
    manquants = [f for f in TEXT_FIELDS if f not in props]
    assert not manquants, f"champs perdus dans le schema impose : {manquants}"
    assert "$defs" not in json_schema()          # grammaire llama.cpp : pas de $ref
