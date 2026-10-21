"""Mesures géométriques élémentaires sur des coordonnées projetées.

Ces deux fonctions vivaient dans la validation, où elles n'avaient rien à faire : ce ne
sont pas des règles métier mais des primitives. Elles étaient importées par la fusion des
lecteurs, qui dépendait donc de la validation pour arbitrer un désaccord. La fusion arbitre
par l'aire, pas par des règles — c'est de la géométrie qu'elle dépend, et c'est ce que
rend maintenant le graphe des imports.
"""

from __future__ import annotations

import math

Coord = tuple[float, float]


def shoelace_area(coords: list[Coord]) -> float:
    """Superficie par la formule de Gauss. Coordonnées recentrées sur le premier point pour
    éviter la perte de précision (les valeurs UTM font ~1e6 et leurs produits ~1e12)."""
    x0, y0 = coords[0]
    pts = [(x - x0, y - y0) for x, y in coords]
    s = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
            for i in range(len(pts)))
    return abs(s) / 2


def side_length(a: Coord, b: Coord) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])
