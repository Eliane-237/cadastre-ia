"""Persistance par fichiers JSON : un fichier par document.

Ce qu'on écrit, c'est l'extraction — pas le polygone. La géométrie est reconstruite à la
lecture par `geospatial.parcel.parcel()`, la même fonction que partout ailleurs. Deux
raisons : une seule source de vérité, et un polygone enregistré qui cesserait de
correspondre à son extraction serait un mensonge durable.

L'enveloppe, elle, est écrite : c'est le seul champ dérivé, et il permet à `near()` de
filtrer sans reconstruire tous les polygones de la base. C'est exactement ce qu'un index
GiST fera en PostGIS, à la main et en moins bien — mais sur quelques centaines de
parcelles, la différence ne se voit pas.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.core.schema import Extraction
from app.geospatial.parcel import GeometryRefusee, Parcel, parcel
from app.storage.repository import (
    Emprise,
    NonEnregistrable,
    ParcelleEnregistree,
)
from app.validation.verdict import Decision, Jugement

VERSION = "1.0"


def _se_touchent(a: Emprise, b: Emprise) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


class JsonParcelStore:
    """Un dossier, un fichier `<document_id>.json` par parcelle enregistrée."""

    def __init__(self, dossier: Path | str = Path("data/parcelles")) -> None:
        self.dossier = Path(dossier)
        self.dossier.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ écriture

    def save(self, parcelle: Parcel, extraction: Extraction,
             jugement: Jugement) -> ParcelleEnregistree:
        """Enregistre une parcelle dont la géométrie tient, et refuse les autres."""
        if not jugement.entre_en_base:
            raise NonEnregistrable(
                f"{parcelle.document_id} : {jugement}. "
                "Une parcelle sans géométrie n'a rien à faire en base."
            )
        if parcelle.document_id != extraction.document_id:
            raise NonEnregistrable(
                f"la parcelle dit « {parcelle.document_id} » et l'extraction "
                f"« {extraction.document_id} » : on n'enregistre pas les deux ensemble."
            )
        quand = datetime.now(timezone.utc)
        self._chemin(parcelle.document_id).write_text(
            json.dumps(
                {
                    "store_version": VERSION,
                    "enregistree_le": quand.isoformat(),
                    "emprise": list(parcelle.polygon.bounds),
                    "jugement": {
                        "decision": jugement.decision.value,
                        "raisons": jugement.raisons,
                    },
                    "extraction": json.loads(extraction.model_dump_json()),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return ParcelleEnregistree(parcelle, extraction, jugement, quand)

    def delete(self, document_id: str) -> bool:
        chemin = self._chemin(document_id)
        if not chemin.exists():
            return False
        chemin.unlink()
        return True

    # ------------------------------------------------------------------ lecture

    def get(self, document_id: str) -> ParcelleEnregistree | None:
        chemin = self._chemin(document_id)
        return self._charge(chemin) if chemin.exists() else None

    def near(self, emprise: Emprise) -> list[Parcel]:
        """Les parcelles dont l'enveloppe touche `emprise`.

        On lit l'enveloppe écrite dans le fichier avant de reconstruire quoi que ce soit :
        un document qui ne peut pas toucher l'emprise ne coûte qu'une lecture de JSON.
        """
        trouvees = []
        for chemin in sorted(self.dossier.glob("*.json")):
            try:
                brut = json.loads(chemin.read_text(encoding="utf-8"))
                if not _se_touchent(tuple(brut["emprise"]), emprise):
                    continue
                enregistree = self._depuis(brut)
            except (OSError, ValueError, KeyError, GeometryRefusee):
                continue          # un fichier abîmé ne doit pas vider la base
            if enregistree is not None:
                trouvees.append(enregistree.parcelle)
        return trouvees

    def all(self) -> list[Parcel]:
        out = []
        for chemin in sorted(self.dossier.glob("*.json")):
            enregistree = self._charge(chemin)
            if enregistree is not None:
                out.append(enregistree.parcelle)
        return out

    # ------------------------------------------------------------------ interne

    def _chemin(self, document_id: str) -> Path:
        # Un identifiant vient d'un document : il peut contenir « / » (« 6652/TH »).
        return self.dossier / f"{document_id.replace('/', '_')}.json"

    def _charge(self, chemin: Path) -> ParcelleEnregistree | None:
        try:
            return self._depuis(json.loads(chemin.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, GeometryRefusee):
            return None

    @staticmethod
    def _depuis(brut: dict) -> ParcelleEnregistree | None:
        extraction = Extraction.model_validate(brut["extraction"])
        jugement = Jugement(
            Decision(brut["jugement"]["decision"]), list(brut["jugement"]["raisons"])
        )
        return ParcelleEnregistree(
            parcelle=parcel(extraction),          # reconstruite, jamais relue telle quelle
            extraction=extraction,
            jugement=jugement,
            enregistree_le=datetime.fromisoformat(brut["enregistree_le"]),
        )
