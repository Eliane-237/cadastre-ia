"""Réparation d'un chiffre mal lu, sous contrainte de la superficie imprimée.

Quand les deux lecteurs se trompent de la même façon, l'arbitrage entre eux ne peut rien :
aucune des valeurs proposées n'est la bonne. Mais la superficie imprimée sur le document est
une contrainte dure, indépendante de la lecture des coordonnées. On énumère alors les
variantes plausibles d'un chiffre et on garde celle — et seulement celle — qui replace
l'aire du polygone dans la tolérance.

Ce module ne décide jamais entre deux candidates : si zéro ou plusieurs variantes passent le
contrôle, il ne répare rien et la valeur part en revue humaine. Mieux vaut un champ signalé
qu'une coordonnée corrigée au hasard.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Confusions courantes des moteurs OCR sur les chiffres imprimés en petits caractères.
# Table symétrique : si 6 peut être lu 8, alors 8 peut être lu 6.
_CONFUSIONS: dict[str, str] = {
    "0": "689",
    "1": "47",
    "2": "7",
    "3": "589",
    "4": "19",
    "5": "368",
    "6": "058",
    "7": "12",
    "8": "0356",
    "9": "034",
}

_COORD = re.compile(r"^(\d+)\.(\d+)$")


@dataclass(frozen=True)
class Repair:
    """Une réparation retenue, avec de quoi la justifier dans le journal d'audit."""

    index: int           # rang du point dans le tableau
    axis: str            # « x » ou « y »
    before: str
    after: str
    kind: str            # « substitution », « insertion » ou « suppression »
    area_before: float
    area_after: float
    printed_m2: float

    @property
    def message(self) -> str:
        return (
            f"Point #{self.index + 1}, {self.axis.upper()} : « {self.before} » → "
            f"« {self.after} » ({self.kind}). L'aire passe de {self.area_before:.1f} m² à "
            f"{self.area_after:.1f} m², pour {self.printed_m2:.0f} m² imprimés."
        )


def shoelace(points: list[tuple[float, float]]) -> float:
    """Aire du polygone par la formule de Gauss."""
    total = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


def _substitutions(value: str) -> list[str]:
    """Variantes obtenues en remplaçant un seul chiffre par une confusion plausible."""
    out = []
    for i, c in enumerate(value):
        for other in _CONFUSIONS.get(c, ""):
            out.append(value[:i] + other + value[i + 1 :])
    return out


def _length_fixes(value: str, expected_digits: int) -> list[str]:
    """Variantes par insertion ou suppression d'un chiffre, quand la longueur de la partie
    entière s'écarte de celle des autres lignes de la colonne. « 38210.57 » dans une colonne
    à six chiffres signale un chiffre perdu, et les positions possibles sont peu nombreuses."""
    m = _COORD.match(value)
    if not m:
        return []
    entier, dec = m.groups()
    out = []

    if len(entier) == expected_digits - 1:  # un chiffre manque
        for i in range(len(entier) + 1):
            for d in "0123456789":
                out.append(f"{entier[:i]}{d}{entier[i:]}.{dec}")
    elif len(entier) == expected_digits + 1:  # un chiffre en trop
        for i in range(len(entier)):
            out.append(f"{entier[:i]}{entier[i + 1:]}.{dec}")
    return out


def _expected_digits(values: list[str]) -> int | None:
    """Longueur dominante de la partie entière d'une colonne."""
    lengths = [len(m.group(1)) for v in values if (m := _COORD.match(v))]
    if not lengths:
        return None
    return max(set(lengths), key=lengths.count)


def _sides(points: list[tuple[float, float]]) -> list[float]:
    """Longueurs des côtés, du point i au point i+1, le dernier revenant au premier."""
    return [math.dist(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]


def _sides_ok(
    points: list[tuple[float, float]],
    printed: list[float | None] | None,
    tolerance_m: float,
) -> bool:
    """La superficie seule ne discrimine pas toujours : plusieurs corrections d'un chiffre
    peuvent donner la même aire. Les cotes portées sur le plan lèvent l'ambiguïté."""
    if not printed:
        return True
    calcule = _sides(points)
    for i, attendu in enumerate(printed):
        if attendu is None or i >= len(calcule):
            continue
        if abs(calcule[i] - attendu) > tolerance_m:
            return False
    return True


def _gap_pct(area: float, printed: float) -> float:
    return abs(area - printed) / printed * 100 if printed else float("inf")


def find_repair(
    xs: list[str],
    ys: list[str],
    printed_m2: float,
    tolerance_pct: float = 0.5,
    printed_sides: list[float | None] | None = None,
    side_tolerance_m: float = 0.05,
) -> Repair | None:
    """Cherche la réparation d'UN chiffre qui replace l'aire dans la tolérance.

    `printed_sides` porte les longueurs de côtés lues sur le plan, dans l'ordre des points.
    Elles sont facultatives mais précieuses : sur une petite parcelle, plusieurs corrections
    donnent la même aire, et seules les cotes les départagent.

    Retourne None si l'aire est déjà correcte, si aucune variante ne convient, ou si
    plusieurs conviennent — dans ce dernier cas la contrainte ne discrimine pas, et
    choisir serait arbitraire.
    """
    if len(xs) != len(ys) or len(xs) < 3 or not printed_m2:
        return None
    try:
        base = [(float(x), float(y)) for x, y in zip(xs, ys)]
    except ValueError:
        return None

    area_before = shoelace(base)
    if _gap_pct(area_before, printed_m2) <= tolerance_pct:
        return None  # rien à réparer

    exp_x = _expected_digits(xs)
    exp_y = _expected_digits(ys)

    retenues: list[Repair] = []
    for axis, values, expected in (("x", xs, exp_x), ("y", ys, exp_y)):
        for i, value in enumerate(values):
            candidates = [(v, "substitution") for v in _substitutions(value)]
            if expected is not None:
                candidates += [(v, "longueur") for v in _length_fixes(value, expected)]

            for variant, kind in candidates:
                if variant == value:
                    continue
                try:
                    nouveau = float(variant)
                except ValueError:
                    continue
                points = list(base)
                x, y = points[i]
                points[i] = (nouveau, y) if axis == "x" else (x, nouveau)
                area = shoelace(points)
                if _gap_pct(area, printed_m2) <= tolerance_pct and _sides_ok(
                    points, printed_sides, side_tolerance_m
                ):
                    precis = (
                        "insertion" if kind == "longueur" and len(variant) > len(value)
                        else "suppression" if kind == "longueur"
                        else "substitution"
                    )
                    retenues.append(
                        Repair(i, axis, value, variant, precis,
                               area_before, area, printed_m2)
                    )

    # Plusieurs variantes différentes qui passent : la contrainte ne tranche pas.
    distinctes = {(r.index, r.axis, r.after) for r in retenues}
    if len(distinctes) != 1:
        return None
    return retenues[0]
