"""Parcelle de référence et lecteurs factices (aucun GPU nécessaire)."""

from __future__ import annotations

import math

from model.readers.base import DocumentReading, FieldReading, PointReading, SideReading
from model.schema import Source
from model.validate import shoelace_area

# Pentagone réaliste, coordonnées imprimées avec 2 décimales (Dakar, UTM 28N).
COORDS = [
    ("381210.57", "1564042.96"),
    ("381225.56", "1564034.94"),
    ("381236.10", "1564051.22"),
    ("381224.03", "1564066.85"),
    ("381208.44", "1564060.13"),
]
LABELS = [f"B{i + 1}" for i in range(len(COORDS))]
FLOATS = [(float(x), float(y)) for x, y in COORDS]
AREA_M2 = round(shoelace_area(FLOATS))
SIDES = [f"{math.dist(FLOATS[i], FLOATS[(i + 1) % 5]):.2f}" for i in range(5)]


def area_text(m2: int = AREA_M2) -> str:
    ha, rest = divmod(m2, 10_000)
    return f"{ha:02d} ha {rest // 100:02d} a {rest % 100:02d} ca"


def reading(
    source: Source, coords=COORDS, *, sides=True, fields=None, labels=LABELS
) -> DocumentReading:
    f = {
        "title_number": FieldReading("TF 1738/DK"),
        "nicad": FieldReading("0123456789012"),
        "area": FieldReading(area_text()),
    }
    if source is Source.QWEN_VL:
        f.update(
            owner=FieldReading("Awa NDIAYE", 0.95), locality=FieldReading("Dakar, Médina", 0.9)
        )
    if fields is not None:
        f = fields
    pts = [
        PointReading(label=lab, x=x, y=y, row_index=i)
        for i, (lab, (x, y)) in enumerate(zip(labels, coords, strict=True))
    ]
    s = [SideReading(length=d, row_index=i) for i, d in enumerate(SIDES)] if sides else []
    return DocumentReading(source=source, model=source.value, fields=f, points=pts, sides=s)


class FakeReader:
    def __init__(self, result: DocumentReading | Exception) -> None:
        self.result = result
        self.source = result.source if isinstance(result, DocumentReading) else Source.QWEN_VL
        self.model_id = "fake"
        self.seen = []

    def read(self, image):
        self.seen.append(image.shape)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result
