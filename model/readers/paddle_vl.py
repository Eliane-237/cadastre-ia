"""Lecteur PaddleOCR-VL-1.6 : parsing de page, tableau de coordonnées en structure.

PaddleOCR-VL découpe la page en blocs (titres, paragraphes, tableaux, cachets…) et transcrit
chacun ; les tableaux sont rendus en HTML. Il ne sait pas quel texte est le numéro de titre :
ce lecteur en retient surtout le tableau de coordonnées, là où un chiffre faux coûte cher, et
quelques champs à forme fixe (n° de titre, NICAD, superficie) repérés par motif pour
recouper la lecture de Qwen3-VL.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from model.config import PaddleSettings
from model.numbers import fix_digits, parse_area
from model.readers.base import DocumentReading, FieldReading, PointReading
from model.schema import BBox, Source
from model.tables import best_coordinate_table, points_from_text

MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"

_TITLE = re.compile(
    r"(?:T\.?\s?[FM]\.?|titre\s*(?:foncier|m[èe]re))\s*(?:n\s?[°ºo]\.?|num[ée]ro|n\.?)?\s*:?\s*"
    r"(\d[\d .]*\d|\d)\s*/\s*([A-Z]{1,4})\b",
    re.IGNORECASE,
)
_NICAD = re.compile(r"NICAD\s*(?:n\s?[°ºo]\.?)?\s*:?\s*([\d\s\-]{10,24}\d)", re.IGNORECASE)
_AREA_KEY = re.compile(r"superficie|contenance|surface|contenant", re.IGNORECASE)


def _bbox(b: Any) -> BBox | None:
    try:
        x0, y0, x1, y1 = (float(v) for v in b)
        return BBox(x0=x0, y0=y0, x1=x1, y1=y1)
    except (TypeError, ValueError):
        return None


def _blocks(data: dict) -> list[dict]:
    blocks = data.get("parsing_res_list") or []
    # Ordre de lecture quand il est fourni, sinon ordre de la liste.
    return sorted(blocks, key=lambda b: (b.get("block_order") is None, b.get("block_order") or 0))


def page_text(data: dict) -> str:
    return "\n".join(
        str(b.get("block_content") or "") for b in _blocks(data) if b.get("block_label") != "table"
    )


def _fields_from_text(data: dict) -> dict[str, FieldReading]:
    fields: dict[str, FieldReading] = {}
    for block in _blocks(data):
        if block.get("block_label") == "table":
            continue
        text = fix_digits(str(block.get("block_content") or ""))
        bbox = _bbox(block.get("block_bbox"))
        if "title_number" not in fields and (m := _TITLE.search(text)):
            fields["title_number"] = FieldReading(raw=m.group(0).strip(), bbox=bbox)
        if "nicad" not in fields and (m := _NICAD.search(text)):
            fields["nicad"] = FieldReading(raw=m.group(1).strip(), bbox=bbox)
        if "area" not in fields and (k := _AREA_KEY.search(text)):
            tail = text[k.end() : k.end() + 60]
            if parse_area(tail) is not None:
                fields["area"] = FieldReading(raw=tail.strip(), bbox=bbox)
    return fields


def reading_from_result(data: dict) -> DocumentReading:
    """Sortie JSON de PaddleOCR-VL (`res.json["res"]`) → lecture brute. Fonction pure."""
    tables = [b for b in _blocks(data) if b.get("block_label") == "table"]
    table = best_coordinate_table([str(b.get("block_content") or "") for b in tables])
    points: list[PointReading]
    sides = []
    if table is not None:
        points, sides = table.points, table.sides
        # La cellule exacte n'est pas localisée par le pipeline : on rattache le bloc tableau.
        tbbox = next((_bbox(b.get("block_bbox")) for b in tables), None)
        for p in points:
            p.bbox = tbbox
    else:
        points = points_from_text(page_text(data))
    pre = data.get("doc_preprocessor_res") or {}
    angle = pre.get("angle") if isinstance(pre, dict) else None
    return DocumentReading(
        source=Source.PADDLE_VL,
        model=MODEL_ID,
        fields=_fields_from_text(data),
        points=points,
        sides=sides,
        raw_output={
            k: v
            for k, v in data.items()
            if k
            in ("parsing_res_list", "doc_preprocessor_res", "width", "height", "model_settings")
        },
        orientation_deg=angle if isinstance(angle, int) and angle >= 0 else None,
    )


class PaddleVLReader:
    source = Source.PADDLE_VL
    model_id = MODEL_ID

    def __init__(self, settings: PaddleSettings | None = None) -> None:
        self.settings = settings or PaddleSettings()
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            try:
                from paddleocr import PaddleOCRVL
            except ImportError as exc:  # pragma: no cover - dépend de l'environnement GPU
                raise RuntimeError("PaddleOCR-VL non installé : voir requirements-gpu.txt") from exc
            s = self.settings
            kwargs: dict[str, Any] = {
                "pipeline_version": s.pipeline_version,
                "use_doc_orientation_classify": s.use_doc_orientation_classify,
                "use_doc_unwarping": s.use_doc_unwarping,
            }
            if s.backend == "server":
                kwargs.update(vl_rec_backend="vllm-server", vl_rec_server_url=s.server_url)
            if s.device:
                kwargs["device"] = s.device
            self._pipeline = PaddleOCRVL(**kwargs)
        return self._pipeline

    def read(self, image: np.ndarray) -> DocumentReading:
        (res,) = list(self.pipeline.predict(image))[:1]
        data = res.json.get("res", res.json)
        reading = reading_from_result(data)
        # Image réorientée par PaddleOCR : c'est elle que doit lire Qwen3-VL.
        try:
            out = res["doc_preprocessor_res"]["output_img"]
            reading.image = np.ascontiguousarray(out) if out is not None else None
        except (KeyError, TypeError):
            reading.image = None
        return reading
