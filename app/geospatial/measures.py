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


# Le centimètre est le pas de ce que nos documents impriment : deux relevés de la même
# borne ne se distinguent pas en dessous. C'est un plancher, jamais une précision
# atteinte.
PRECISION_PLANCHER_M = 0.01

# Classes reprises du cadastre néerlandais (Kadaster, FIG 2020), qui publie une qualité
# par élément plutôt qu'une précision globale. Les bornes sont les leurs ; ce qu'on y
# fait entrer est calculé sur nos documents, pas déclaré.
CLASSES_QUALITE: tuple[tuple[float, str], ...] = (
    (0.05, "B"),   # 5 cm
    (0.10, "C"),   # 10 cm
    (0.20, "D"),   # 20 cm — leur défaut urbain
    (0.40, "E"),   # 40 cm — leur défaut rural
)


def precision_position(aire_calculee: float, aire_imprimee: float | None,
                       perimetre: float) -> float | None:
    """Incertitude de position des sommets, en mètres. None si le document ne permet pas
    de la déduire.

    Nos coordonnées ne sont pas mesurées sur le dessin : elles sont imprimées dans un
    tableau. Leur incertitude n'est donc pas celle du tracé, c'est celle du levé
    d'origine — que le document ne déclare nulle part.

    Mais il la trahit. Un défaut de position uniforme ε sur toutes les limites d'un
    polygone de périmètre P modifie son aire d'environ P·ε. L'écart entre l'aire
    calculée depuis les sommets et l'aire imprimée impose donc une borne basse :

        ε ≈ |aire calculée − aire imprimée| / périmètre

    C'est peu, et c'est honnête : la seule mesure de qualité que le papier nous donne.
    Sans superficie imprimée, il n'y a rien à confronter et on rend None — une qualité
    inconnue se dit, elle ne se remplace pas par une valeur par défaut.
    """
    if aire_imprimee is None or perimetre <= 0:
        return None
    return max(abs(aire_calculee - aire_imprimee) / perimetre, PRECISION_PLANCHER_M)


def classe_qualite(precision_m: float | None) -> str:
    """La classe de qualité correspondante, « ? » si la précision est inconnue."""
    if precision_m is None:
        return "?"
    for borne, lettre in CLASSES_QUALITE:
        if precision_m <= borne:
            return lettre
    return "F"      # au-delà de 40 cm : hors des classes publiées par le Kadaster
