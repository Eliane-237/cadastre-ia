"""Paramètres de travail du modèle.

Aucune de ces valeurs n'est mesurée à ce stade : elles seront recalibrées sur les titres annotés
(voir eval/). On les garde ici, et nulle part ailleurs, pour pouvoir les changer en un point.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Confiance minimale pour accepter une valeur sans relecture (à calibrer).
    review_threshold: float = 0.85
    # Tolérances de cohérence géométrique (à calibrer).
    side_tolerance_m: float = 0.05
    area_tolerance_pct: float = 0.5


SETTINGS = Settings()