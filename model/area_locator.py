"""Où est la superficie imprimée sur la page.

La superficie est le juge de toute la validation géométrique : sans elle, l'aire calculée
par la formule de Gauss n'est confrontée à rien et le document ne peut plus se contredire
lui-même. Sur un titre où aucun des deux lecteurs ne la voit, on perd l'arbitrage entier —
d'où `AREA_UNAVAILABLE`. Comme pour le tableau de coordonnées, le remède est de localiser
la région et de la faire relire de près.

Trois pistes, de la plus sûre à la plus faible. Une région fausse coûte plus cher qu'une
région absente : relire le mauvais fragment donnerait une superficie inventée, et une
superficie inventée est pire qu'une superficie manquante — elle fait mentir l'arbitrage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from model.normalize import parse_area
from model.table_locator import _bbox, _texte, _trop_grande

_MOT = re.compile(r"superfici\w*", re.IGNORECASE)

# Repli géométrique. Sur les cinq titres du corpus, la superficie est imprimée en haut à
# droite du cadre du plan, sur la ligne du « NORD ». Mesuré sur les images : le mot
# « Superficie » commence entre 0,56 et 0,66 de la largeur, et la ligne tombe entre 0,05
# et 0,33 de la hauteur — ce dernier extrême étant 54987580, dont l'en-tête administratif
# est le plus haut du corpus. La bande ci-dessous garde une marge franche des deux côtés.
# Cinq documents ne font pas une règle : c'est la piste la plus faible, et elle ne sert
# que lorsque les deux autres n'ont rien donné.
BANDEAU = (0.45, 0.00, 1.00, 0.42)


@dataclass(frozen=True)
class AreaRegion:
    """Une région candidate pour la superficie, avec d'où elle vient."""

    bbox: tuple[float, float, float, float]
    source: str                # "contenu" | "etiquette" | "bandeau"
    note: str

    @property
    def confiance(self) -> str:
        return {"contenu": "certaine", "etiquette": "plausible"}.get(self.source, "faible")


def locate(raw: dict, taille: tuple[int, int] | None = None) -> list[AreaRegion]:
    """Régions candidates pour la superficie, la plus sûre d'abord."""
    regions: list[AreaRegion] = []
    for bloc in raw.get("parsing_res_list") or []:
        if not isinstance(bloc, dict):
            continue
        bbox = _bbox(bloc)
        if bbox is None or _trop_grande(bbox, taille):
            continue
        texte = _texte(bloc.get("block_content"))
        if not _MOT.search(texte):
            continue
        aire = parse_area(texte)
        if aire is not None and aire.m2 > 0:
            regions.append(AreaRegion(bbox, "contenu",
                                      f"superficie lue dans ce bloc : {aire.m2:.0f} m²"))
        else:
            # Le mot est là, la valeur non : la boîte reste exactement ce qu'il faut
            # pour aller la relire de près.
            regions.append(AreaRegion(bbox, "etiquette",
                                      "« superficie » reconnu, valeur non transcrite"))
    regions.sort(key=lambda r: r.source != "contenu")
    return regions


def locate_by_band(taille: tuple[int, int]) -> AreaRegion:
    """Le haut droit du cadre, quand la mise en page n'a rien donné du tout."""
    largeur, hauteur = taille
    x0, y0, x1, y1 = BANDEAU
    return AreaRegion(
        (x0 * largeur, y0 * hauteur, x1 * largeur, y1 * hauteur),
        "bandeau",
        "position habituelle de la superficie, haut droit du cadre",
    )


def best_region(raw: dict | None, image=None) -> AreaRegion | None:
    """La meilleure région candidate : la mise en page d'abord, la bande habituelle ensuite."""
    taille = (image.shape[1], image.shape[0]) if image is not None else None
    regions = locate(raw or {}, taille)
    if regions:
        return regions[0]
    return locate_by_band(taille) if taille is not None else None
