"""Ce que l'API rend, et qui n'est pas ce que le métier manipule.

Un `Parcel` porte un polygone Shapely, un `Jugement` est une dataclass : ni l'un ni
l'autre ne se sérialise seul. Ces modèles sont la frontière — les changer ne touche pas
le métier, et changer le métier ne casse pas les clients sans qu'on s'en aperçoive.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.schema import Extraction
from app.geospatial.overlap import Recouvrement
from app.geospatial.parcel import Parcel
from app.validation.verdict import Jugement


class JugementOut(BaseModel):
    decision: str
    raisons: list[str] = Field(default_factory=list)
    entre_en_base: bool

    @classmethod
    def de(cls, j: Jugement) -> JugementOut:
        return cls(decision=j.decision.value, raisons=j.raisons, entre_en_base=j.entre_en_base)


class ParcelleOut(BaseModel):
    document_id: str
    titre: str | None = None
    titre_mere: str | None = None
    crs: str
    datum: str
    comparable: bool
    aire_m2: float
    aire_imprimee_m2: float | None = None
    ecart_pct: float | None = None
    sommets: list[tuple[float, float]]

    @classmethod
    def de(cls, p: Parcel) -> ParcelleOut:
        ecart = (abs(p.area_m2 - p.printed_m2) / p.printed_m2 * 100
                 if p.printed_m2 else None)
        return cls(
            document_id=p.document_id, titre=p.title_number, titre_mere=p.mother_title,
            crs=p.crs, datum=p.datum, comparable=p.comparable,
            aire_m2=round(p.area_m2, 2), aire_imprimee_m2=p.printed_m2,
            ecart_pct=round(ecart, 3) if ecart is not None else None,
            sommets=[(round(x, 3), round(y, 3))
                     for x, y in list(p.polygon.exterior.coords)[:-1]],
        )


class RecouvrementOut(BaseModel):
    verdict: str
    a: str
    b: str
    aire_m2: float
    part_pct: float
    conflit: bool
    message: str

    @classmethod
    def de(cls, r: Recouvrement) -> RecouvrementOut:
        return cls(verdict=r.verdict.value, a=r.a, b=r.b, aire_m2=round(r.aire_m2, 2),
                   part_pct=round(r.part_pct, 2), conflit=r.conflit, message=r.message)


class LectureOut(BaseModel):
    """Le résultat d'une lecture de document : ce qu'on a lu, et ce qu'on en juge.

    `geojson` et `centre` sont là pour la carte, `apercu_png` pour l'œil. Tout arrive
    dans la même réponse parce que rien n'est encore enregistré : il n'y a pas
    d'identifiant pour aller rechercher le dessin dans un second appel.
    """

    extraction: Extraction
    jugement: JugementOut
    parcelle: ParcelleOut | None = None
    geometrie_refusee: str | None = None
    geojson: dict | None = Field(
        default=None, description="FeatureCollection en WGS84, prête pour une carte.")
    centre: tuple[float, float] | None = Field(
        default=None, description="Longitude et latitude du centroïde, pour poser un repère.")
    apercu_png: str | None = Field(
        default=None,
        description="Le tracé en PNG, encodé en base64. Demandé par `apercu=true` : "
                    "il alourdit la réponse de quelques dizaines de kilo-octets.")


class EnregistrementIn(BaseModel):
    extraction: Extraction


class DecisionIn(BaseModel):
    action: str
    auteur: str
    motif: str = ""
    detail: dict | None = None
