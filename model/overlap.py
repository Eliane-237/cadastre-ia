"""Confrontation de deux parcelles : y a-t-il recouvrement, et de quelle nature.

Le système ne déclare jamais une fraude. Il constate un recouvrement géométrique et
le qualifie ; dire si c'est une fraude, une erreur de levé, une limite périmée ou un
morcellement non encore enregistré relève de l'humain et du droit. Un système qui
crie à la fraude et se trompe une fois perd toute crédibilité auprès du cadastre.

Trois précautions portent ce module :

- Deux titres voisins partagent une limite. Un simple `intersects()` les signalerait
  tous, et deux levés de la même limite diffèrent déjà de quelques centimètres. On
  mesure donc l'aire recouverte, rapportée à la plus petite des deux parcelles.
- Une parcelle entièrement contenue dans une autre n'est pas forcément une fraude :
  c'est souvent un morcellement régulier, et le titre mère le dit. Le document de
  Thiès porte trois fois « Surplus TF 6652/TH ».
- Deux parcelles exprimées dans des datums différents ou inconnus ne se comparent
  pas. Un mètre d'écart, sur une parcelle aux côtés de quinze mètres, en fait 7 %.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from model.config import SETTINGS, Settings
from model.geometry import Parcel


class Verdict(str, Enum):
    VOISINAGE = "voisinage"            # limite partagée : rien à signaler
    CHEVAUCHEMENT = "chevauchement"    # recouvrement partiel : conflit
    INCLUSION = "inclusion"            # une parcelle dans l'autre : conflit
    MORCELLEMENT = "morcellement"      # inclusion expliquée par le titre mère


class NonComparable(Exception):
    """Les deux parcelles ne sont pas dans le même référentiel."""


@dataclass(frozen=True)
class Recouvrement:
    """Ce que deux parcelles ont en commun, et comment il faut le lire."""

    verdict: Verdict
    a: str
    b: str
    aire_m2: float
    part_pct: float                     # du recouvrement dans la plus petite parcelle

    @property
    def conflit(self) -> bool:
        return self.verdict in (Verdict.CHEVAUCHEMENT, Verdict.INCLUSION)

    @property
    def message(self) -> str:
        if self.verdict is Verdict.VOISINAGE:
            return (f"{self.a} et {self.b} se touchent sur {self.aire_m2:.2f} m² "
                    f"({self.part_pct:.2f} %) : limite partagée.")
        if self.verdict is Verdict.MORCELLEMENT:
            return (f"{self.a} est contenu dans {self.b} à {self.part_pct:.1f} %, "
                    "et le déclare pour titre mère : morcellement.")
        if self.verdict is Verdict.INCLUSION:
            return (f"{self.a} est contenu dans {self.b} à {self.part_pct:.1f} % "
                    f"({self.aire_m2:.0f} m²) sans titre mère qui l'explique.")
        return (f"{self.a} et {self.b} se recouvrent sur {self.aire_m2:.0f} m², "
                f"soit {self.part_pct:.1f} % de la plus petite.")


def _lie_par_titre_mere(a: Parcel, b: Parcel) -> bool:
    """L'une des deux se déclare-t-elle détachée de l'autre ?"""
    paires = ((a.mother_title, b.title_number), (b.mother_title, a.title_number))
    return any(mere and titre and mere.strip() == titre.strip() for mere, titre in paires)


def compare(a: Parcel, b: Parcel, cfg: Settings = SETTINGS) -> Recouvrement | None:
    """Ce que les deux parcelles ont en commun, ou None si elles sont disjointes.

    Lève `NonComparable` si leur référentiel n'est pas le même ou n'est pas connu :
    un recouvrement calculé entre deux datums différents ne veut rien dire.
    """
    if not (a.comparable and b.comparable):
        inconnu = a.document_id if not a.comparable else b.document_id
        raise NonComparable(f"{inconnu} : datum inconnu, la confrontation n'a pas de sens.")
    if a.crs != b.crs or a.datum != b.datum:
        raise NonComparable(
            f"{a.document_id} et {b.document_id} ne sont pas dans le même référentiel "
            f"({a.datum} / {b.datum})."
        )

    commun = a.polygon.intersection(b.polygon)
    if commun.is_empty or commun.area <= 0:
        return None

    plus_petite = min(a.area_m2, b.area_m2)
    part = commun.area / plus_petite * 100 if plus_petite else 0.0

    if part < cfg.overlap_min_share_pct:
        verdict = Verdict.VOISINAGE
    elif part >= cfg.overlap_inclusion_pct:
        verdict = Verdict.MORCELLEMENT if _lie_par_titre_mere(a, b) else Verdict.INCLUSION
    else:
        verdict = Verdict.CHEVAUCHEMENT

    # La contenue en premier, pour que le message se lise dans le bon sens.
    premier, second = (a, b) if a.area_m2 <= b.area_m2 else (b, a)
    return Recouvrement(
        verdict=verdict,
        a=premier.title_number or premier.document_id,
        b=second.title_number or second.document_id,
        aire_m2=commun.area,
        part_pct=part,
    )


def scan(nouvelle: Parcel, base: list[Parcel], cfg: Settings = SETTINGS) -> list[Recouvrement]:
    """Confronte une parcelle à une base de référence, les conflits d'abord.

    L'index spatial ne retient que les parcelles dont l'enveloppe touche celle de la
    nouvelle : confronter chaque titre du Sénégal à tous les autres serait inutile.
    """
    from shapely.strtree import STRtree

    voisines = [p for p in base if p.document_id != nouvelle.document_id]
    if not voisines:
        return []
    index = STRtree([p.polygon for p in voisines])

    trouves: list[Recouvrement] = []
    for position in index.query(nouvelle.polygon):
        resultat = compare(nouvelle, voisines[int(position)], cfg)
        if resultat is not None:
            trouves.append(resultat)
    trouves.sort(key=lambda r: (not r.conflit, -r.part_pct))
    return trouves
