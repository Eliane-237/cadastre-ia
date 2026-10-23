"""La parcelle comme objet géométrique, une fois ses coordonnées validées.

Jusqu'ici le polygone n'existait que le temps d'un calcul de surface, à l'intérieur
de la validation. Pour confronter deux titres entre eux il faut qu'il devienne un
objet à part entière, qui transporte avec lui son système de coordonnées ET son
datum : sans cela, un traitement en aval comparera un titre sénégalais à une donnée
WGS84 sans savoir qu'il se trompe d'un mètre.

Le module refuse de construire une géométrie à partir de coordonnées douteuses. Un
polygone faux se propagerait dans la détection de chevauchement, où il accuserait
un propriétaire ou en blanchirait un autre.
"""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from app.core.schema import DATUM_INCONNU, UTM_28N, Extraction, Severity
from app.geospatial.measures import classe_qualite, precision_position

MIN_POINTS = 3


class GeometryRefusee(Exception):
    """La géométrie n'a pas pu être construite, et la raison est dans le message."""


@dataclass(frozen=True)
class Parcel:
    """Une parcelle géoréférencée, prête à être confrontée à d'autres."""

    document_id: str
    polygon: Polygon
    crs: str
    datum: str
    title_number: str | None = None
    mother_title: str | None = None
    nicad: str | None = None           # l'identifiant cadastral, quand il est imprimé
    lot: str | None = None             # en copropriété, ce qui distingue deux lots d'un NICAD
    owner: str | None = None           # tel qu'écrit sur le document, jamais normalisé ici
    printed_m2: float | None = None

    @property
    def area_m2(self) -> float:
        """L'aire calculée depuis les sommets, en mètres carrés (la projection est métrique)."""
        return self.polygon.area

    @property
    def precision_m(self) -> float | None:
        """Incertitude de position des sommets, déduite du document. None si inconnue.

        C'est elle qui dit si un recouvrement mesuré est du signal ou du bruit de levé.
        """
        return precision_position(self.area_m2, self.printed_m2, self.polygon.length)

    @property
    def qualite(self) -> str:
        """La classe de qualité (B à F, « ? » si inconnue), au sens du Kadaster."""
        return classe_qualite(self.precision_m)

    @property
    def comparable(self) -> bool:
        """Confronter deux parcelles n'a de sens que si leur datum est connu et commun."""
        return self.datum != DATUM_INCONNU

    def __str__(self) -> str:
        titre = self.title_number or self.document_id
        return f"{titre} — {self.area_m2:.1f} m², {len(self.polygon.exterior.coords) - 1} sommets"


def _valeur(champ) -> str | None:
    return champ.value if champ.present else None


def _canonique(polygone: Polygon) -> Polygon:
    """Même terrain, même suite de sommets, quel que soit l'ordre de lecture du tableau.

    `orient` fixe le sens de parcours mais pas le sommet de départ : deux lectures du
    même tableau, commencées à des bornes différentes, donneraient deux représentations
    distinctes du même polygone. On repart toujours du sommet le plus au sud-ouest.
    """
    anneau = list(orient(polygone, sign=1.0).exterior.coords)[:-1]
    depart = min(range(len(anneau)), key=lambda i: anneau[i])
    return Polygon(anneau[depart:] + anneau[:depart])


def parcel(extraction: Extraction) -> Parcel:
    """L'extraction validée → la parcelle géoréférencée. Lève `GeometryRefusee` sinon.

    Les sommets sont orientés dans le sens direct, pour que deux parcelles construites
    séparément soient comparables sans dépendre de l'ordre de lecture du tableau.
    """
    bloquantes = [i for i in extraction.issues if i.severity is Severity.ERROR]
    if bloquantes:
        codes = ", ".join(sorted({i.code for i in bloquantes}))
        raise GeometryRefusee(
            f"{extraction.document_id} : coordonnées à corriger d'abord ({codes})."
        )

    sommets = [(p.x.value, p.y.value) for p in extraction.points if p.complete]
    if len(sommets) < MIN_POINTS:
        raise GeometryRefusee(
            f"{extraction.document_id} : {len(sommets)} sommet(s) complet(s), "
            f"il en faut au moins {MIN_POINTS}."
        )
    if len(set(sommets)) < len(sommets):
        raise GeometryRefusee(f"{extraction.document_id} : deux sommets confondus.")

    polygone = Polygon(sommets)
    if not polygone.is_valid:
        # `buffer(0)` réparerait l'auto-intersection en inventant une forme : refus.
        raise GeometryRefusee(
            f"{extraction.document_id} : le polygone se recoupe lui-même, "
            "l'ordre des sommets ou une coordonnée est faux."
        )

    aire = extraction.printed_area
    return Parcel(
        document_id=extraction.document_id,
        polygon=_canonique(polygone),            # sens direct, départ au sud-ouest
        crs=extraction.crs or UTM_28N,
        datum=extraction.datum,
        title_number=_valeur(extraction.title_number),
        mother_title=_valeur(extraction.mother_title),
        nicad=_valeur(extraction.nicad),
        lot=_valeur(extraction.lot),
        owner=_valeur(extraction.owner),
        printed_m2=float(aire.value.m2) if aire.present and aire.value else None,
    )
