"""Format d'annotation : la vérité terrain d'un document, un fichier JSON par titre.

Les coordonnées sont des chaînes, exactement comme imprimées (« 381210.57 »), pour ne perdre
ni zéro final ni nombre de décimales. Le nombre de décimales fait partie de la vérité.
Une annotation n'est pas « fiable » tant que `verified` est faux : elle doit être relue par un humain.
"""
from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from model.schema import Area

_COORD = re.compile(r"^\d{5,8}\.\d{1,3}$")
ANNOTATION_VERSION = "1.0"


class AnnotatedPoint(BaseModel):
    label: str
    x: str
    y: str

    @field_validator("x", "y")
    @classmethod
    def _format_coordonnee(cls, v: str) -> str:
        if not _COORD.match(v):
            raise ValueError(f"coordonnée mal formée : {v!r} (attendu 5-8 chiffres, point, 1-3 décimales)")
        return v


class Annotation(BaseModel):
    annotation_version: str = ANNOTATION_VERSION
    document_id: str
    image: str                                   # nom du fichier image dans data/raw/
    title_number: str | None = None
    nicad: str | None = None
    locality: str | None = None
    printed_area: Area | None = None
    points: list[AnnotatedPoint] = Field(default_factory=list)
    decimals: int | None = Field(default=None, ge=0, le=3)
    quality: str | None = None                   # lisibilité de l'image, texte libre
    annotator: str = "inconnu"
    verified: bool = False                       # vrai seulement après relecture humaine
    notes: str | None = None

    @field_validator("points")
    @classmethod
    def _labels_uniques(cls, pts: list[AnnotatedPoint]) -> list[AnnotatedPoint]:
        labels = [p.label for p in pts]
        if len(labels) != len(set(labels)):
            raise ValueError("labels de points en double")
        return pts


def save_annotation(a: Annotation, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{a.document_id}.json"
    path.write_text(a.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_annotation(path: Path) -> Annotation:
    return Annotation.model_validate_json(path.read_text(encoding="utf-8"))


def load_all(directory: Path) -> list[Annotation]:
    return [load_annotation(p) for p in sorted(directory.glob("*.json"))]
