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

from app.core.config import SETTINGS

T = TypeVar("T")

UTM_28N = "EPSG:32628"
UTM_29N = "EPSG:32629"

# Le Sénégal s'étend sur deux fuseaux UTM : le 28N jusqu'à 12°O, le 29N au-delà
# (est de Tambacounda, Kédougou, Saraya). La quasi-totalité des titres sont en 28N, mais
# lire un titre de l'est comme du 28N pose la parcelle 652 km à côté — sans que ni sa
# forme, ni sa superficie, ni ses côtés ne changent d'un centimètre. Aucun contrôle
# géométrique ne peut donc rattraper un fuseau faux ; seul l'easting le trahit.
FUSEAUX: dict[str, str] = {"28": UTM_28N, "29": UTM_29N}

# Fuseau supposé quand le document ne l'imprime pas, ce qui est le cas le plus courant :
# un titre porte « RRS 04 », qui nomme le référentiel, pas la projection. L'hypothèse est
# écrite ici plutôt que dissimulée dans un défaut, et le contrôle de plage des eastings
# est là pour buter dessus quand elle est fausse.
FUSEAU_DEFAUT = UTM_28N

# Le RRS04 — Réseau de Référence du Sénégal, établi en 2004 par la DTGC avec l'IGN — est
# une réalisation locale de l'ITRF2000 à l'époque 2004,55, sur l'ellipsoïde GRS80.
#
# GRS80 et WGS84 partagent le même ellipsoïde au millième de millimètre près, et les
# réalisations ITRF2000 et WGS84 s'accordent au centimètre. Ce qui sépare réellement nos
# coordonnées d'un relevé GPS d'aujourd'hui, c'est le TEMPS : elles sont figées à
# l'époque 2004,55, et la plaque africaine dérive vers le nord-est d'environ 3 cm par an.
# Vingt ans plus tard, l'écart est de l'ordre de 60 cm, et il grandit.
#
# Deux documents du cadastre se confrontent donc entre eux sans réserve — ils sont figés
# à la même époque. Confronter un titre à un relevé GPS ou à une image satellite demande,
# lui, une transformation rigoureuse que la DTGC seule peut fournir : le RRS04 n'a ni code
# EPSG ni grille NTv2 publiés.
DATUM_RRS04 = "RRS04 (ITRF2000, époque 2004.55, ellipsoïde GRS80)"
# Tant que le système imprimé n'a pas été lu et reconnu, le datum reste inconnu : un
# traitement en aval doit buter dessus plutôt que de supposer du WGS84.
DATUM_INCONNU = "inconnu"
SCHEMA_VERSION = "2.0"

# Anomalies qui n'empêchent pas de construire la géométrie mais qui doivent passer sous
# un œil humain. Une alerte seule ne déclenche pas la revue : celles-ci, si.
REVIEW_CODES: frozenset[str] = frozenset({"TITRE_EGAL_MERE"})
REVIEW_THRESHOLD = SETTINGS.review_threshold  # valeur de travail, voir app/core/config.py

# Champs texte du document, dans l'ordre du formulaire. Une case imprimée = un champ.
TEXT_FIELDS: tuple[str, ...] = (
    "title_number",       # « TITRE N° » — null si la case est vide
    "mother_title",       # « Titre Mère n° »
    "lot",                # « Lot n° »
    "subdivision",        # « Lotissement »
    "requisition",        # « Réquisition n° »
    "land_registry",      # « LIVRE FONCIER DE »
    "nicad",
    "region",
    "department",
    "arrondissement",
    "commune",
    "owner",              # « Requérant » ou propriétaire
    "coordinate_system",  # « Système de Coordonnées » — ex. « RSS 04 »
    "scale",              # « ECHELLE » du plan principal
    "certified_on",       # date de l'extrait certifié conforme, telle qu'imprimée
    "disclaimer",         # ex. « Ce plan ne constitue pas un titre de propriété »
)

# L'étiquette imprimée au-dessus ou devant chaque case. Un lecteur qui rend l'étiquette
# au lieu de son contenu n'a pas lu une valeur — c'est arrivé sur « Requérant », rendu
# tel quel comme nom du propriétaire. Aucun arbitrage n'est nécessaire dans ce cas : une
# case ne se nomme pas elle-même.
#
# Plusieurs libellés par champ, les formulaires variant d'un livre foncier à l'autre.
# Comparés après `canonical_text` : sans accents, sans ponctuation, en bas de casse.
ETIQUETTES: dict[str, tuple[str, ...]] = {
    "title_number":      ("titre n", "titre no", "titre numero"),
    "mother_title":      ("titre mere", "titre mere n"),
    "lot":               ("lot", "lot n"),
    "subdivision":       ("lotissement",),
    "requisition":       ("requisition", "requisition n"),
    "land_registry":     ("livre foncier", "livre foncier de"),
    "nicad":             ("nicad",),
    "region":            ("region",),
    "department":        ("departement", "dept"),
    "arrondissement":    ("arrondissement", "arrond"),
    "commune":           ("commune", "com ville", "ca cr"),
    "owner":             ("requerant", "proprietaire"),
    "coordinate_system": ("systeme de coordonnees", "systeme"),
    "scale":             ("echelle",),
    "certified_on":      ("certifie conforme", "date"),
}

# Champs dont la valeur est un nom de localité : un répertoire peut alors départager
# deux lectures, une localité connue l'emportant sur une inconnue.
CHAMPS_LOCALITE: tuple[str, ...] = ("region", "department", "arrondissement", "commune")

# Champs dont l'absence ou le doute justifie une relecture humaine. Les autres
# (échelle, date, mention de portée) sont utiles sans être bloquants.
CORE_FIELDS: tuple[str, ...] = (
    "title_number",
    "nicad",
    "commune",
    "owner",
    "printed_area",
)



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
    # Deux questions distinctes, et un document peut répondre à l'une sans l'autre :
    # `crs` dit OÙ projeter (quel fuseau), `datum` dit SUR QUEL RÉFÉRENTIEL. « UTM zone
    # 28N » donne le premier sans le second ; « RRS 04 » donne le second sans le premier.
    crs: str = FUSEAU_DEFAUT
    datum: str = DATUM_INCONNU                   # renseigné par `validated()` d'après le document

        # Identification du titre — une case du formulaire par champ, jamais de concaténation.
    title_number: Extracted[str] = Field(default_factory=Extracted)
    mother_title: Extracted[str] = Field(default_factory=Extracted)
    lot: Extracted[str] = Field(default_factory=Extracted)
    subdivision: Extracted[str] = Field(default_factory=Extracted)
    requisition: Extracted[str] = Field(default_factory=Extracted)
    land_registry: Extracted[str] = Field(default_factory=Extracted)
    nicad: Extracted[str] = Field(default_factory=Extracted)

    # Localisation administrative : un niveau de la hiérarchie officielle par champ.
    region: Extracted[str] = Field(default_factory=Extracted)
    department: Extracted[str] = Field(default_factory=Extracted)
    arrondissement: Extracted[str] = Field(default_factory=Extracted)
    commune: Extracted[str] = Field(default_factory=Extracted)

    owner: Extracted[str] = Field(default_factory=Extracted)
    printed_area: Extracted[Area] = Field(default_factory=Extracted)
    coordinate_system: Extracted[str] = Field(default_factory=Extracted)
    scale: Extracted[str] = Field(default_factory=Extracted)
    certified_on: Extracted[str] = Field(default_factory=Extracted)
    disclaimer: Extracted[str] = Field(default_factory=Extracted)

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
        for name in TEXT_FIELDS + ("printed_area",):
            field = getattr(self, name)
            if field.present and field.confidence < threshold:
                out.append(name)          # lu, mais douteux → relire
            elif not field.present and name in CORE_FIELDS:
                out.append(name) 
        for p in self.points:
            if not p.x.is_reliable(threshold):
                out.append(f"points[{p.label}].x")
            if not p.y.is_reliable(threshold):
                out.append(f"points[{p.label}].y")
        return out

    def needs_review(self, threshold: float = REVIEW_THRESHOLD) -> bool:
        return (
            self.has_error
            or any(i.code in REVIEW_CODES for i in self.issues)
            or bool(self.low_confidence_fields(threshold))
        )