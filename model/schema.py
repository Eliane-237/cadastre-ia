"""Contrat de sortie du modèle d'extraction.

Principe : le modèle ne « décide » rien. Il produit des valeurs, chacune accompagnée de sa
confiance, de sa provenance et de sa position dans l'image. Toute la suite (validation
géométrique, revue par un agent) s'appuie sur ce contrat.

Deux lecteurs indépendants (PaddleOCR-VL-1.6 et Qwen3-VL-8B) lisent chaque document. Chaque
valeur garde la trace de toutes ses lectures (`readings`) : la valeur retenue, sa confiance et
la raison du choix restent vérifiables par l'agent et par le journal d'audit.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from model.config import SETTINGS

T = TypeVar("T")

UTM_28N = "EPSG:32628"
SCHEMA_VERSION = "2.0"
REVIEW_THRESHOLD = SETTINGS.review_threshold  # valeur de travail, voir model/config.py


class Source(str, Enum):
    OCR = "ocr"              # lu directement par un moteur OCR
    PARSER = "parser"        # reconstruit par le parseur (regex, regroupement de tableau)
    DERIVED = "derived"      # recalculé/corrigé par une règle (ex. distance, superficie)
    MANUAL = "manual"        # saisi ou corrigé par un agent
    PADDLE_VL = "paddle_vl"  # lu par PaddleOCR-VL (tableau structuré, texte de page)
    QWEN_VL = "qwen_vl"      # lu par Qwen3-VL (JSON au schéma imposé)
    CONSENSUS = "consensus"  # les deux lecteurs donnent la même chaîne
    GEOMETRY = "geometry"    # désaccord tranché par la cohérence géométrique


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class BBox(BaseModel):
    """Boîte englobante en pixels de l'image d'origine (origine en haut à gauche)."""
    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def _ordonnee(self) -> BBox:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("BBox invalide : x1/y1 doivent être >= x0/y0")
        return self


class Reading(BaseModel):
    """Une lecture brute d'une valeur par un lecteur, conservée pour l'audit."""
    source: Source
    raw_text: str | None = None          # chaîne telle que lue
    normalized: str | None = None        # forme canonique comparée entre lecteurs
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)  # propre au lecteur, si connue
    bbox: BBox | None = None


class Extracted(BaseModel, Generic[T]):
    """Une valeur extraite, avec de quoi la juger et la retrouver dans l'image."""
    value: T | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: Source = Source.OCR
    raw_text: str | None = None   # texte brut lu, avant nettoyage
    bbox: BBox | None = None
    readings: list[Reading] = Field(default_factory=list)

    @property
    def present(self) -> bool:
        return self.value is not None

    def is_reliable(self, threshold: float = REVIEW_THRESHOLD) -> bool:
        return self.present and self.confidence >= threshold


class BoundaryPoint(BaseModel):
    """Un sommet du polygone, tel qu'imprimé dans le tableau de coordonnées."""
    label: str                                   # ex. « B1 », « 12 »
    x: Extracted[float]                          # Est, UTM 28N (m)
    y: Extracted[float]                          # Nord, UTM 28N (m)
    row_index: int = Field(ge=0)                 # position dans le tableau, 0 = première ligne

    @property
    def complete(self) -> bool:
        return self.x.present and self.y.present


class Area(BaseModel):
    """Superficie imprimée (ha, a, ca) et sa valeur normalisée en m²."""
    ha: int = 0
    a: int = 0
    ca: int = 0

    @property
    def m2(self) -> float:
        return self.ha * 10_000 + self.a * 100 + self.ca


class Issue(BaseModel):
    """Anomalie constatée. Alerte d'aide à la décision, jamais un verdict de fraude."""
    code: str                                    # ex. « AREA_MISMATCH », « POINT_MISSING »
    severity: Severity
    message: str
    point_labels: list[str] = Field(default_factory=list)


class Provenance(BaseModel):
    """De quoi rejouer et auditer une extraction : document, modèles, réglages, durées."""
    input_sha256: str | None = None
    pipeline_version: str = SCHEMA_VERSION
    models: dict[str, str] = Field(default_factory=dict)       # rôle -> identifiant du modèle
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    timings_s: dict[str, float] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)       # lecteur -> erreur rencontrée


class Extraction(BaseModel):
    """Résultat complet pour un document."""
    schema_version: str = SCHEMA_VERSION
    document_id: str
    crs: str = UTM_28N

    title_number: Extracted[str] = Field(default_factory=Extracted)   # n° du titre foncier
    nicad: Extracted[str] = Field(default_factory=Extracted)
    owner: Extracted[str] = Field(default_factory=Extracted)
    locality: Extracted[str] = Field(default_factory=Extracted)
    printed_area: Extracted[Area] = Field(default_factory=Extracted)

    points: list[BoundaryPoint] = Field(default_factory=list)
    # Distances imprimées sur le plan : l'indice i est le côté du point i vers le point i+1
    # (le dernier revient au premier).
    printed_sides: list[Extracted[float]] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)

    @property
    def has_error(self) -> bool:
        return any(i.severity is Severity.ERROR for i in self.issues)

    def low_confidence_fields(self, threshold: float = REVIEW_THRESHOLD) -> list[str]:
        """Noms des champs à faire relire : valeur absente ou confiance sous le seuil."""
        out: list[str] = []
        for name in ("title_number", "nicad", "owner", "locality", "printed_area"):
            if not getattr(self, name).is_reliable(threshold):
                out.append(name)
        for p in self.points:
            if not p.x.is_reliable(threshold):
                out.append(f"points[{p.label}].x")
            if not p.y.is_reliable(threshold):
                out.append(f"points[{p.label}].y")
        return out

    def needs_review(self, threshold: float = REVIEW_THRESHOLD) -> bool:
        return self.has_error or bool(self.low_confidence_fields(threshold))