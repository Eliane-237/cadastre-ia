"""Ce que les routeurs reçoivent, et d'où ça vient.

Le dépôt est injecté, jamais construit dans un routeur. C'est la seule raison pour
laquelle passer de JSON à PostGIS ne touchera que ce fichier — et pour laquelle les
tests de l'API tournent sur un dossier temporaire sans rien savoir du reste.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from app.core.config import Settings, from_env
from app.review.history import Historique
from app.storage.json_store import JsonParcelStore
from app.storage.repository import ParcelRepository


@lru_cache(maxsize=1)
def settings() -> Settings:
    return from_env()


@lru_cache(maxsize=1)
def depot() -> ParcelRepository:
    """Le dépôt de parcelles. CADASTRE_DATA_DIR déplace la base ; PostGIS la remplacera."""
    return JsonParcelStore(Path(os.getenv("CADASTRE_DATA_DIR", "data")) / "parcelles")


@lru_cache(maxsize=1)
def historique() -> Historique:
    return Historique(Path(os.getenv("CADASTRE_DATA_DIR", "data")) / "historique.jsonl")


def reinitialise() -> None:
    """Vide les caches. Les tests s'en servent pour repointer le dépôt ailleurs."""
    settings.cache_clear()
    depot.cache_clear()
    historique.cache_clear()
