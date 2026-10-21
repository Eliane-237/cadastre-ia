"""Le verdict d'une extraction : une décision unique, et ses raisons.

Les 22 contrôles rendent une liste d'`Issue`. C'est juste pour auditer, inutilisable pour
décider : un agent du cadastre ne lit pas vingt-deux codes, il lit une décision. Et sans
décision explicite, la seule barrière du système était une exception Python levée au fond
de `geospatial.parcel.parcel()` — efficace, mais invisible.

Ce module ne contrôle rien de nouveau. Il traduit ce que les contrôles ont déjà dit, et
il porte la responsabilité qui manquait : dire ce qui a le droit d'entrer dans la base.
Car une extraction fausse qu'on enregistre ne reste pas fausse toute seule — elle fera
demain un faux chevauchement, et bloquera un titre légitime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.core.schema import DATUM_INCONNU, REVIEW_THRESHOLD, Extraction, Severity


class Decision(str, Enum):
    EXPLOITABLE = "exploitable"   # peut être enregistrée et confrontée
    A_RELIRE = "a_relire"         # un humain doit trancher avant
    REFUSE = "refuse"             # inexploitable en l'état


@dataclass(frozen=True)
class Jugement:
    """La décision, et pourquoi. Les raisons sont écrites pour être lues par un agent."""

    decision: Decision
    raisons: list[str] = field(default_factory=list)

    @property
    def entre_en_base(self) -> bool:
        return self.decision is Decision.EXPLOITABLE

    def __str__(self) -> str:
        if not self.raisons:
            return self.decision.value
        return f"{self.decision.value} : " + " ; ".join(self.raisons)


def juger(extraction: Extraction, seuil: float = REVIEW_THRESHOLD) -> Jugement:
    """Traduit les anomalies constatées en une décision unique.

    Trois niveaux, et l'ordre compte : on refuse d'abord, on doute ensuite.
    """
    raisons: list[str] = []

    # --- refus : la géométrie ne peut pas être construite sur ces coordonnées.
    erreurs = sorted({i.code for i in extraction.issues if i.severity is Severity.ERROR})
    if erreurs:
        raisons.append("anomalies bloquantes : " + ", ".join(erreurs))
        return Jugement(Decision.REFUSE, raisons)

    if not extraction.points:
        return Jugement(Decision.REFUSE, ["aucun sommet lu : il n'y a pas de parcelle"])

    # --- revue : la géométrie tient, mais quelque chose demande un œil humain.
    if extraction.datum == DATUM_INCONNU:
        systeme = extraction.coordinate_system
        declare = systeme.value if systeme.present else None
        # On dit « datum inconnu » et non « système non reconnu » : la cause peut être
        # l'un ou l'autre, et affirmer la mauvaise envoie l'agent vérifier le mauvais
        # endroit du document.
        raisons.append(
            f"datum inconnu, pour le système imprimé « {declare} »" if declare
            else "aucun système de coordonnées imprimé : datum inconnu"
        )
        # Sans datum, la parcelle existe mais ne se confronte à rien : la confrontation
        # dirait n'importe quoi, et un humain doit décider s'il assume le référentiel.

    alertes = sorted({i.code for i in extraction.issues if i.severity is Severity.WARNING})
    if alertes:
        raisons.append("à vérifier : " + ", ".join(alertes))

    champs = extraction.low_confidence_fields(seuil)
    if champs:
        apercu = ", ".join(champs[:6]) + ("…" if len(champs) > 6 else "")
        raisons.append(f"{len(champs)} champ(s) à relire : {apercu}")

    if raisons:
        return Jugement(Decision.A_RELIRE, raisons)
    return Jugement(Decision.EXPLOITABLE, [])
