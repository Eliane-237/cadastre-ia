"""Contrat commun aux lecteurs : une lecture brute, avant toute fusion.

Un lecteur ne décide rien : il rapporte ce qu'il a lu, en chaînes (« 381210.57 » et non
381210.57), avec sa propre confiance quand il en a une. La fusion (model/fusion.py) compare
ensuite les lectures, et la validation géométrique départage.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel

from model.schema import BBox, Source

FIELDS = ("title_number", "nicad", "owner", "locality", "area")


@dataclass
class FieldReading:
    raw: str | None
    confidence: float | None = None
    bbox: BBox | None = None


@dataclass
class PointReading:
    label: str | None
    x: str | None  # Est, tel qu'imprimé
    y: str | None  # Nord, tel qu'imprimé
    row_index: int
    x_confidence: float | None = None
    y_confidence: float | None = None
    bbox: BBox | None = None  # cellule ou ligne du tableau, si connue


@dataclass
class SideReading:
    """Distance imprimée entre deux sommets (libellés quand le document les donne)."""

    length: str | None
    from_label: str | None = None
    to_label: str | None = None
    row_index: int | None = None  # côté « ligne i → ligne i+1 » quand c'est une colonne
    confidence: float | None = None


@dataclass
class DocumentReading:
    source: Source
    model: str
    fields: dict[str, FieldReading] = field(default_factory=dict)
    points: list[PointReading] = field(default_factory=list)
    sides: list[SideReading] = field(default_factory=list)
    raw_output: Any = None  # sortie brute du modèle, pour l'audit (JSON)
    # Image effectivement lue, si le lecteur l'a réorientée (les bbox s'y réfèrent).
    image: np.ndarray | None = None
    orientation_deg: int | None = None

    def field(self, name: str) -> FieldReading:
        return self.fields.get(name) or FieldReading(raw=None)


class Reader(Protocol):
    source: Source
    model_id: str

    def read(self, image: np.ndarray) -> DocumentReading: ...


# --------------------------------------------------------------------------- persistance
# Les lectures brutes sont sauvegardées pour rejouer la fusion et la calibrer sans GPU.


def _plain(o: Any) -> Any:
    if isinstance(o, BaseModel):
        return o.model_dump()
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def reading_to_dict(r: DocumentReading) -> dict[str, Any]:
    """Lecture → dict JSON (sans l'image)."""
    d = {f.name: getattr(r, f.name) for f in fields(r) if f.name != "image"}
    return _plain(
        {
            **d,
            "fields": {k: vars(v) for k, v in r.fields.items()},
            "points": [vars(p) for p in r.points],
            "sides": [vars(s) for s in r.sides],
        }
    )


def reading_from_dict(d: dict[str, Any]) -> DocumentReading:
    def bbox(b):
        return BBox(**b) if isinstance(b, dict) else None

    return DocumentReading(
        source=Source(d["source"]),
        model=d["model"],
        fields={
            k: FieldReading(raw=v["raw"], confidence=v.get("confidence"), bbox=bbox(v.get("bbox")))
            for k, v in d.get("fields", {}).items()
        },
        points=[PointReading(**{**p, "bbox": bbox(p.get("bbox"))}) for p in d.get("points", [])],
        sides=[SideReading(**s) for s in d.get("sides", [])],
        raw_output=d.get("raw_output"),
        orientation_deg=d.get("orientation_deg"),
    )


def merge_pages(pages: list[DocumentReading]) -> DocumentReading:
    """Plusieurs pages lues par un même lecteur → une lecture : le tableau vient de la page qui a
    le plus de points complets, chaque champ de la première page qui le contient."""
    if len(pages) == 1:
        return pages[0]
    best = max(pages, key=lambda r: sum(1 for p in r.points if p.x and p.y))
    merged: dict[str, FieldReading] = {}
    for r in pages:
        for k, v in r.fields.items():
            if v.raw is not None and k not in merged:
                merged[k] = v
    return DocumentReading(
        source=best.source,
        model=best.model,
        fields=merged,
        points=best.points,
        sides=best.sides,
        raw_output=[r.raw_output for r in pages],
        image=best.image,
        orientation_deg=best.orientation_deg,
    )
