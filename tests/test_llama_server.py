"""Exécution sous Windows avec llama.cpp : réglages, passes séparées, schéma sans $ref."""

import json
import sys
import types

import numpy as np

from app.core.config import PaddleSettings, from_env
from app.core.schema import Source
from app.document.preprocess import apply_orientation
from app.extraction.pipeline import run
from app.extraction.readers.paddle import PaddleVLReader
from app.extraction.readers.prompt import json_schema
from eval import evaluate
from tests.fake_readers import FakeReader, reading
from tests.test_extract import scan  # noqa: F401 - fixture


def test_reglages_par_lenvironnement(monkeypatch):
    monkeypatch.setenv("CADASTRE_PADDLE_SERVER_KIND", "llama-cpp-server")
    monkeypatch.setenv("CADASTRE_PADDLE_DEVICE", "cpu")
    cfg = from_env()
    assert cfg.paddle.server_kind == "llama-cpp-server" and cfg.paddle.device == "cpu"


def test_paddle_branche_sur_llama_cpp(monkeypatch):
    seen = {}
    fake = types.ModuleType("paddleocr")
    fake.PaddleOCRVL = lambda **kw: seen.update(kw) or object()
    monkeypatch.setitem(sys.modules, "paddleocr", fake)
    assert PaddleVLReader(PaddleSettings(server_kind="llama-cpp-server", device="cpu")).pipeline
    assert seen["vl_rec_backend"] == "llama-cpp-server"
    assert seen["vl_rec_server_url"] == "http://127.0.0.1:8080/v1" and seen["device"] == "cpu"


def test_schema_sans_reference():
    text = json.dumps(json_schema())
    assert "$ref" not in text and "$defs" not in text
    assert json_schema()["properties"]["points"]["items"]["properties"]["x"]


def test_meme_rotation_que_paddleocr():
    img = np.arange(6, dtype=np.uint8).reshape(2, 3, 1)
    assert apply_orientation(img, 90).shape == (3, 2, 1)
    assert apply_orientation(img, 90)[0, 0, 0] == img[0, 2, 0]  # sens anti-horaire
    assert np.array_equal(apply_orientation(img, 180), img[::-1, ::-1])
    assert apply_orientation(img, None) is img


def test_seconde_passe_qwen_reprend_lorientation(scan):  # noqa: F811
    q = FakeReader(reading(Source.QWEN_VL))
    run(scan, paddle=None, qwen=q, use_paddle=False, orientation_deg=90)
    assert q.seen[0] == (600, 800, 3)  # page 800×600 tournée de 90°


def _paddle_qui_localise_le_tableau():
    """Un lecteur Paddle dont la mise en page designe un bloc « table » avec sa boite.

    On ne s'appuie pas sur le detecteur morphologique : il accroche ou non selon la
    version d'OpenCV, et un test ne doit pas dependre de l'environnement.
    """
    lecture = reading(Source.PADDLE_VL)
    lecture.raw_output = {
        "parsing_res_list": [
            {"block_label": "table", "block_bbox": [40.0, 120.0, 300.0, 260.0],
             "block_content": ""},
        ]
    }
    return FakeReader(lecture)


def test_le_tableau_est_relu_a_part(scan):  # noqa: F811
    """La page entiere est lue pour tous les champs, puis la seule region du tableau
    est relue, recadree et agrandie. `merge_pages` retiendra le meilleur des deux."""
    q = FakeReader(reading(Source.QWEN_VL))
    run(scan, paddle=_paddle_qui_localise_le_tableau(), qwen=q)
    assert len(q.seen) == 2, "la page, puis le fragment du tableau"
    page, fragment = q.seen
    assert fragment[:2] != page[:2], "le fragment n'est pas la page entiere"


def test_relecture_du_tableau_desactivable(scan):  # noqa: F811
    import dataclasses

    from app.core.config import SETTINGS

    q = FakeReader(reading(Source.QWEN_VL))
    cfg = dataclasses.replace(SETTINGS, table_rescan=False)
    run(scan, paddle=_paddle_qui_localise_le_tableau(), qwen=q, cfg=cfg)
    assert len(q.seen) == 1, "seule la page est lue"


def test_deux_passes_completent_les_lectures(tmp_path, monkeypatch):
    from app.review.annotation import AnnotatedPoint, Annotation, save_annotation

    raw, ann, out = tmp_path / "raw", tmp_path / "ann", tmp_path / "rep"
    raw.mkdir()
    (raw / "D01.png").write_bytes(b"")
    save_annotation(
        Annotation(
            document_id="D01",
            image="D01.png",
            points=[AnnotatedPoint(label="B1", x="381210.57", y="1564042.96")],
        ),
        ann,
    )
    p = reading(Source.PADDLE_VL)
    p.orientation_deg = 180
    calls = []

    def fake_run(img, doc_id, *, cfg, use_paddle, use_qwen, orientation_deg):
        calls.append((use_paddle, use_qwen, orientation_deg))
        readings = {}
        if use_paddle:
            readings["paddle_vl"] = p
        if use_qwen:
            readings["qwen_vl"] = reading(Source.QWEN_VL)
        return types.SimpleNamespace(readings=readings)

    monkeypatch.setattr(evaluate, "run", fake_run)
    evaluate.main(raw, ann, out, replay=False, only="paddle")
    evaluate.main(raw, ann, out, replay=False, only="qwen")
    assert calls == [(True, False, None), (False, True, 180)]
    cached = json.loads((out / "readings" / "D01.json").read_text())
    assert set(cached) == {"paddle_vl", "qwen_vl"}
    report = json.loads((out / "evaluation.json").read_text())
    assert report["summary"]["fusion"]["values"] == 2


def test_verification_des_serveurs(capsys):
    import httpx

    from app.core.config import Settings
    from app.extraction.readers import health as check_servers

    def handler(request):
        return httpx.Response(200 if request.url.port == 8080 else 503)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert check_servers.main(Settings(), client) == 1
    out = capsys.readouterr().out
    assert "✓ PaddleOCR-VL prêt sur http://127.0.0.1:8080" in out
    assert "Qwen3-VL démarre encore" in out
