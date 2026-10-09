"""Lecteur Qwen3-VL-8B-Instruct : remplit le JSON de l'`Extraction`, tableau compris.

Deux modes :
- « server » (recommandé sur GPU) : vLLM expose une API compatible OpenAI ; le JSON est imposé
  au décodage (`response_format` json_schema) et les logprobs donnent la confiance par champ ;
- « local » : transformers dans le processus ; le JSON est validé après coup.

C'est une deuxième lecture indépendante : Qwen3-VL ne voit pas ce qu'a lu PaddleOCR-VL, sinon
leurs erreurs seraient corrélées et leur accord ne prouverait plus rien.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np

from model.config import QwenSettings
from model.imaging import encode_png, limit_size
from model.readers.base import DocumentReading, FieldReading, PointReading, SideReading
from model.readers.json_spans import Path, value_confidences
from model.readers.prompts import SYSTEM_PROMPT, USER_PROMPT, QwenOutput, json_schema
from model.schema import Source


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
        for name in ("title_number", "nicad", "owner", "locality")
    }
    if out.area is not None:
        a = out.area
        raw_area = a.text
        if any(v is not None for v in (a.ha, a.a, a.ca)):
            raw_area = f"{a.ha or 0} ha {a.a or 0} a {a.ca or 0} ca"
        parts = [c("area", k) for k in ("ha", "a", "ca", "text")]
        known = [p for p in parts if p is not None]
        fields["area"] = FieldReading(raw=raw_area, confidence=min(known) if known else None)
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
        self._local = None

    # ------------------------------------------------------------------ serveur (vLLM)

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

    def _read_server(self, image: np.ndarray) -> DocumentReading:
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

    # ------------------------------------------------------------------ local (transformers)

    def _load_local(self):  # pragma: no cover - nécessite GPU et poids
        if self._local is None:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.settings.model, dtype=torch.bfloat16, device_map="auto"
            ).eval()
            self._local = (model, AutoProcessor.from_pretrained(self.settings.model))
        return self._local

    def _read_local(self, image: np.ndarray) -> DocumentReading:  # pragma: no cover - GPU
        import torch
        from PIL import Image

        model, processor = self._load_local()
        rgb = limit_size(image, self.settings.max_image_side)[:, :, ::-1]
        schema_hint = json.dumps(json_schema(), ensure_ascii=False)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.fromarray(np.ascontiguousarray(rgb))},
                    {
                        "type": "text",
                        "text": f"{USER_PROMPT}\n\nSchéma JSON à respecter :\n{schema_hint}",
                    },
                ],
            },
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        gen = out.sequences[0, inputs["input_ids"].shape[1] :]
        logprobs = model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True
        )[0]
        tok = processor.tokenizer
        tokens, prev = [], ""
        for k in range(len(gen)):
            cur = tok.decode(gen[: k + 1], skip_special_tokens=True)
            tokens.append((cur[len(prev) :], float(logprobs[k])))
            prev = cur
        text = prev
        data, _ = extract_json(text)
        conf = value_confidences(text, tokens, text.find("{"))
        return reading_from_output(QwenOutput.model_validate(data), conf, self.model_id, data)

    def read(self, image: np.ndarray) -> DocumentReading:
        if self.settings.backend == "local":
            return self._read_local(image)
        return self._read_server(image)
