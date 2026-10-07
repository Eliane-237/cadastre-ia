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
        # Plages plausibles en UTM 28N pour le Sénégal (m). Larges à dessein : elles écartent les
        # lectures absurdes, pas les parcelles inhabituelles. À resserrer avec les données du cadastre.
        easting_range: tuple[float, float] = (100_000.0, 900_000.0)
        northing_range: tuple[float, float] = (1_300_000.0, 1_900_000.0)
        # Étendue maximale d'une parcelle (m) : au-delà, un sommet est probablement mal lu.
        max_parcel_extent_m: float = 10_000.0


SETTINGS = Settings()