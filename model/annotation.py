"""Format d'annotation : la vérité terrain d'un document, un fichier JSON par titre.

Les coordonnées sont des chaînes, exactement comme imprimées (« 381210.57 »), pour ne perdre
ni zéro final ni nombre de décimales. Le nombre de décimales fait partie de la vérité.
Une annotation n'est pas « fiable » tant que `verified` est faux : elle doit être relue par un humain.

Un champ vide ne dit rien par lui-même. `no_value` tranche : un champ qui y figure est un
fait constaté sur le document, avec sa raison ; un champ ni rempli ni listé est un oubli de
notre part. Les confondre, c'est compter nos trous comme des inventions du modèle le jour où
on mesurera l'extraction.

Les quatre raisons viennent du corpus, pas d'une idée a priori : une case vide, un cachet ou
une rature au marqueur par-dessus la valeur, un cliché qui coupe l'en-tête, et un formulaire
qui fusionne plusieurs niveaux administratifs dans une seule case.
"""
from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from model.schema import TEXT_FIELDS, Area

_COORD = re.compile(r"^\d{5,8}\.\d{1,3}$")
ANNOTATION_VERSION = "2.0"

# Champs dont on peut constater qu'ils n'ont pas de valeur. On en exclut `document_id` et
# `image`, qui sont de notre fait, et `points`, sans quoi il n'y a pas de parcelle du tout.
# `printed_area` en fait partie : un document peut ne pas imprimer sa superficie — on perd
# alors l'arbitrage géométrique, ce qui est un constat, pas une faute d'annotation.
DECLARABLE_FIELDS: tuple[str, ...] = TEXT_FIELDS + ("printed_area",)


class NoValue(str, Enum):
    """Pourquoi un champ n'a pas de valeur. Aucune de ces raisons n'est un oubli."""

    ABSENT = "absent"              # pas de case sur le formulaire, ou case laissée vide
    ILLISIBLE = "illisible"        # imprimé mais indéchiffrable : cachet, rature, flou
    HORS_CADRAGE = "hors_cadrage"  # la case existe, le cliché ne la contient pas
    FUSIONNE = "fusionne"          # le formulaire fond plusieurs niveaux dans une case

    @property
    def mesurable(self) -> bool:
        """Vrai si l'extraction doit rendre ce champ vide. Faux : champ à écarter du calcul.

        Sur un champ ABSENT, un modèle qui produit une valeur invente, et cela doit compter
        contre lui. Sur les trois autres, la valeur existe sur le papier : l'absence vient de
        notre cliché ou de notre schéma, pas du modèle. Les compter serait le punir pour nos
        propres limites.
        """
        return self is NoValue.ABSENT


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
    """Vérité terrain d'un document. Un champ = une case imprimée ; null si la case
    est vide ou illisible. On ne déduit rien, on ne concatène rien."""

    annotation_version: str = ANNOTATION_VERSION
    document_id: str
    image: str                                   # nom du fichier image dans data/raw/

    title_number: str | None = None
    mother_title: str | None = None
    lot: str | None = None
    subdivision: str | None = None
    requisition: str | None = None
    land_registry: str | None = None
    nicad: str | None = None

    region: str | None = None
    department: str | None = None
    arrondissement: str | None = None
    commune: str | None = None

    owner: str | None = None
    printed_area: Area | None = None
    coordinate_system: str | None = None
    scale: str | None = None
    certified_on: str | None = None
    disclaimer: str | None = None

    # Champs constatés sans valeur, et pourquoi. Voir DECLARABLE_FIELDS et NoValue.
    no_value: dict[str, NoValue] = Field(default_factory=dict)

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

    @field_validator("no_value")
    @classmethod
    def _champs_declarables(cls, d: dict[str, NoValue]) -> dict[str, NoValue]:
        inconnus = [n for n in d if n not in DECLARABLE_FIELDS]
        if inconnus:
            raise ValueError(
                f"no_value : champs inconnus {inconnus} ; "
                f"attendu parmi {list(DECLARABLE_FIELDS)}"
            )
        return d

    @model_validator(mode="after")
    def _pas_de_contradiction(self) -> Annotation:
        remplis = [n for n in self.no_value if getattr(self, n) is not None]
        if remplis:
            raise ValueError(
                "champs à la fois renseignés et déclarés absents du document : "
                f"{remplis}"
            )
        return self

    @property
    def not_annotated(self) -> list[str]:
        """Champs ni renseignés ni déclarés absents : des oublis, pas des constats."""
        return [
            n for n in DECLARABLE_FIELDS
            if getattr(self, n) is None and n not in self.no_value
        ]

    @property
    def measurable_fields(self) -> list[str]:
        """Champs sur lesquels on a le droit de noter l'extraction.

        Un champ renseigné (on sait ce qu'il faut lire) ou déclaré ABSENT (on sait qu'il ne
        faut rien lire). Tout le reste — illisible, hors cadrage, fusionné, pas encore
        annoté — n'est pas une vérité terrain et doit rester hors du calcul.
        """
        return [
            n for n in DECLARABLE_FIELDS
            if getattr(self, n) is not None
            or self.no_value.get(n) is NoValue.ABSENT
        ]


def save_annotation(a: Annotation, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{a.document_id}.json"
    path.write_text(a.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_annotation(path: Path) -> Annotation:
    return Annotation.model_validate_json(path.read_text(encoding="utf-8"))


def load_all(directory: Path) -> list[Annotation]:
    return [load_annotation(p) for p in sorted(directory.glob("*.json"))]
