"""Lecteur Qwen3-VL-8B-Instruct : remplit le JSON de l'`Extraction`, tableau compris.

Le modèle tourne derrière llama-server (API compatible OpenAI). Le JSON est imposé au décodage
(`response_format` json_schema, converti en grammaire par llama.cpp) et les logprobs donnent
la confiance par champ.

C'est une deuxième lecture indépendante : Qwen3-VL ne voit pas ce qu'a lu PaddleOCR-VL, sinon
leurs erreurs seraient corrélées et leur accord ne prouverait plus rien.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np

from app.core.config import QwenSettings
from model.preprocess import encode_png, limit_size
from model.qwen_prompt import SYSTEM_PROMPT, USER_PROMPT, QwenOutput, json_schema
from model.reading import DocumentReading, FieldReading, PointReading, SideReading
from app.core.schema import Area, Source, TEXT_FIELDS
from model.token_confidence import Path, value_confidences


def extract_json(text: str) -> tuple[dict, int]:
    """Premier objet JSON du texte (tolère « ```json … ``` ») et sa position de début."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("aucun objet JSON dans la réponse du modèle")
    return json.loads(text[start : end + 1]), start


def reading_from_output(
    out: QwenOutput, conf: dict[Path, float], model_id: str, raw: Any = None
) -> DocumentReading:
    """Sortie validée de Qwen3-VL (+ confiances par chemin JSON) → lecture brute. Fonction pure."""

    def c(*path: str | int) -> float | None:
        return conf.get(tuple(path))

    fields = {
        name: FieldReading(raw=getattr(out, name), confidence=c(name))
        for name in TEXT_FIELDS
    }
    if out.area is not None:
        a = out.area
        parts = [c("area", k) for k in ("ha", "a", "ca", "text")]
        known = [p for p in parts if p is not None]
        area_conf = min(known) if known else None
        # `raw` porte le texte tel qu'imprimé : c'est lui qui fait foi en aval. Les trois
        # nombres sont conservés en secours, pour les cas où le texte est illisible — le
        # lecteur les range parfois mal (« 80a 05ca » rendu ha=80, soit 100 fois trop).
        if any(v is not None for v in (a.ha, a.a, a.ca)):
            fields["area"] = FieldReading(
                raw=a.text or f"{a.ha or 0} ha {a.a or 0} a {a.ca or 0} ca",
                confidence=area_conf,
                area=Area(ha=a.ha or 0, a=a.a or 0, ca=a.ca or 0),
            )
        elif a.text:
            fields["area"] = FieldReading(raw=a.text, confidence=area_conf)
    points = [
        PointReading(
            label=p.label,
            x=p.x,
            y=p.y,
            row_index=i,
            x_confidence=c("points", i, "x"),
            y_confidence=c("points", i, "y"),
        )
        for i, p in enumerate(out.points)
    ]
    sides = [
        SideReading(
            length=s.length,
            from_label=s.from_label,
            to_label=s.to_label,
            confidence=c("sides", i, "length"),
        )
        for i, s in enumerate(out.sides)
    ]
    return DocumentReading(
        source=Source.QWEN_VL,
        model=model_id,
        fields=fields,
        points=points,
        sides=sides,
        raw_output=raw,
    )


def _messages(image: np.ndarray, max_side: int) -> list[dict]:
    b64 = base64.b64encode(encode_png(limit_size(image, max_side))).decode()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]


class QwenVLReader:
    source = Source.QWEN_VL

    def __init__(self, settings: QwenSettings | None = None, http_client=None) -> None:
        self.settings = settings or QwenSettings()
        self.model_id = self.settings.model
        self._http = http_client

    def _client(self):
        if self._http is None:
            import httpx

            self._http = httpx.Client(timeout=self.settings.timeout_s)
        return self._http

    def request_body(self, image: np.ndarray) -> dict:
        s = self.settings
        return {
            "model": s.model,
            "messages": _messages(image, s.max_image_side),
            "temperature": s.temperature,
            "max_tokens": s.max_new_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_cadastre",
                    "schema": json_schema(),
                    "strict": True,
                },
            },
            "logprobs": True,
        }

    def read(self, image: np.ndarray) -> DocumentReading:
        s = self.settings
        resp = self._client().post(
            f"{s.server_url.rstrip('/')}/chat/completions",
            json=self.request_body(image),
            headers={"Authorization": f"Bearer {s.api_key}"},
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        text = choice["message"]["content"]
        data, start = extract_json(text)
        tokens = [
            (t["token"], t["logprob"])
            for t in ((choice.get("logprobs") or {}).get("content") or [])
        ]
        conf = value_confidences(text, tokens, start)
        return reading_from_output(QwenOutput.model_validate(data), conf, self.model_id, data)
