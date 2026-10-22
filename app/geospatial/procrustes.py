"""Deux relevés de la même parcelle décrivent-ils la même forme ?

Le plan dessiné sur le titre foncier porte la même parcelle que le tableau de
coordonnées, mais sans coordonnées : il est à une échelle, orienté comme l'arpenteur
l'a voulu, et posé n'importe où sur la feuille. Le comparer au polygone calculé exige
donc une mesure insensible à la translation, à la rotation et au facteur d'échelle.

C'est exactement ce que fait l'analyse procustéenne : on superpose les deux nuages de
points au mieux, et ce qui reste est l'écart de forme. Comme les sommets portent les
mêmes étiquettes des deux côtés — B1 avec B1, B2 avec B2 — la correspondance est
connue et la solution est exacte, donnée par une décomposition en valeurs singulières.

Deux précautions propres au foncier :

- Une forme obtenue par réflexion n'est PAS la même parcelle. Un terrain et son miroir
  ont même aire et mêmes côtés ; les confondre reviendrait à valider une géométrie
  retournée. La réflexion est détectée et rapportée, jamais absorbée en silence.
- Le facteur d'échelle retrouvé est une information à part entière : confronté à
  l'échelle imprimée du plan, il dit si les deux documents parlent bien de la même
  parcelle, ou si le plan appartient à un autre titre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MIN_POINTS = 3


class RecalageImpossible(Exception):
    """Les deux relevés ne peuvent pas être superposés, et le message dit pourquoi."""


@dataclass(frozen=True)
class Recalage:
    """Le résultat de la superposition : la transformation trouvée et ce qui reste."""

    echelle: float              # facteur appliqué au candidat pour épouser la référence
    rotation_deg: float         # rotation appliquée, dans le sens direct
    residu: float               # écart quadratique moyen, dans l'unité de la référence
    residu_pct: float           # le même, rapporté à la taille de la référence
    reflexion: bool             # la superposition exige-t-elle un retournement
    n: int

    @property
    def message(self) -> str:
        if self.reflexion:
            return (f"Les deux relevés ne coïncident qu'au prix d'un retournement : "
                    f"ce n'est pas la même parcelle (résidu {self.residu_pct:.2f} %).")
        return (f"{self.n} sommets superposés — échelle {self.echelle:.4f}, rotation "
                f"{self.rotation_deg:+.2f}°, résidu {self.residu:.3f} "
                f"({self.residu_pct:.2f} % de la taille de la parcelle).")


def _nuage(points) -> np.ndarray:
    tableau = np.asarray([(float(x), float(y)) for x, y in points], dtype=float)
    if tableau.ndim != 2 or tableau.shape[1] != 2:
        raise RecalageImpossible("Les points doivent être des couples (x, y).")
    return tableau


def superpose(reference, candidat, *, autoriser_reflexion: bool = False) -> Recalage:
    """Superpose `candidat` sur `reference` au mieux, et mesure ce qui reste.

    Les deux suites doivent décrire les mêmes sommets dans le même ordre — c'est ce
    que garantissent les étiquettes B1, B2, … portées par les deux documents.
    """
    a, b = _nuage(reference), _nuage(candidat)
    if len(a) != len(b):
        raise RecalageImpossible(
            f"{len(a)} sommets d'un côté, {len(b)} de l'autre : rien à superposer."
        )
    if len(a) < MIN_POINTS:
        raise RecalageImpossible(f"{len(a)} sommet(s) : il en faut au moins {MIN_POINTS}.")

    a0, b0 = a - a.mean(axis=0), b - b.mean(axis=0)
    norme_a, norme_b = np.linalg.norm(a0), np.linalg.norm(b0)
    if norme_a == 0 or norme_b == 0:
        raise RecalageImpossible("Un des relevés a tous ses sommets confondus.")

    # Rotation optimale par décomposition en valeurs singulières (Kabsch).
    u, valeurs, vt = np.linalg.svd(b0.T @ a0)
    rotation = u @ vt
    reflexion = bool(np.linalg.det(rotation) < 0)
    if reflexion and not autoriser_reflexion:
        # On force une rotation pure : le résidu dira l'ampleur du désaccord.
        correction = np.diag([1.0, -1.0])
        rotation = u @ correction @ vt
        valeurs = valeurs.copy()
        valeurs[-1] *= -1

    echelle = float(valeurs.sum()) / (norme_b**2)
    recale = (b0 @ rotation) * echelle
    residu = float(np.sqrt(((a0 - recale) ** 2).sum() / len(a)))
    taille = float(norme_a / math.sqrt(len(a)))     # écart-type des sommets au centre

    return Recalage(
        echelle=echelle,
        rotation_deg=float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))),
        residu=residu,
        residu_pct=residu / taille * 100 if taille else math.inf,
        reflexion=reflexion,
        n=len(a),
    )


def concordent(recalage: Recalage, tolerance_pct: float = 1.0) -> bool:
    """Les deux relevés décrivent-ils la même forme, au seuil donné."""
    return not recalage.reflexion and recalage.residu_pct <= tolerance_pct
