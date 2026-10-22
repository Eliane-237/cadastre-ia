"""L'interface de persistance, et ce qu'on enregistre.

Trois méthodes portent tout le reste du projet :

- `save` n'accepte qu'une parcelle jugée exploitable. Une extraction fausse qu'on
  enregistre ne reste pas fausse toute seule : elle fera demain un faux chevauchement
  et bloquera un titre légitime. La garde est ici, au seuil, et nulle part ailleurs.
- `near` rend les parcelles dont l'enveloppe touche une emprise donnée. C'est elle que
  `geospatial.overlap.scan` consomme, et c'est elle que portera l'interpolation par
  distance inverse le jour où on l'écrira — « near things are more related ».
- `all` sert aux inventaires et aux reprises ; elle n'a pas vocation à passer à l'échelle.

L'implémentation JSON parcourt des fichiers ; PostGIS fera le même travail avec un index
GiST. Aucun module métier ne doit pouvoir faire la différence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from app.core.schema import Extraction
from app.geospatial.parcel import Parcel
from app.validation.verdict import Jugement

Emprise = tuple[float, float, float, float]      # x0, y0, x1, y1


@dataclass(frozen=True)
class ParcelleEnregistree:
    """Une parcelle en base, avec ce qui a justifié son enregistrement."""

    parcelle: Parcel
    extraction: Extraction
    jugement: Jugement
    enregistree_le: datetime

    @property
    def document_id(self) -> str:
        return self.parcelle.document_id


class NonEnregistrable(Exception):
    """La parcelle ne remplit pas les conditions d'entrée en base."""


@runtime_checkable
class ParcelRepository(Protocol):
    """Le contrat. JSON aujourd'hui, PostGIS demain, rien d'autre ne change."""

    def save(self, parcelle: Parcel, extraction: Extraction,
             jugement: Jugement) -> ParcelleEnregistree: ...

    def get(self, document_id: str) -> ParcelleEnregistree | None: ...

    def near(self, emprise: Emprise) -> list[Parcel]: ...

    def all(self) -> list[Parcel]: ...

    def delete(self, document_id: str) -> bool: ...
