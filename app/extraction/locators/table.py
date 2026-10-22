"""Où est le tableau de coordonnées sur la page.

Rien ne cherchait cette région jusqu'ici : les lecteurs recevaient la page entière,
où les chiffres du tableau occupent quelques pour cent de la surface, et rendaient
des valeurs plausibles faute de voir les vraies. Localiser la région est le préalable
à toute relecture ciblée.

Trois pistes, de la plus sûre à la plus faible, et un refus franc quand aucune ne
donne rien. Une région fausse coûte plus cher qu'une région absente : relire le
mauvais fragment produirait des coordonnées inventées avec l'aplomb d'une relecture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Un couple de coordonnées sénégalais : l'Est tient sur 6 chiffres, le Nord sur 7.
_EASTING = re.compile(r"\b[12]\d{5}(?:[.,]\d{1,3})?\b")
_NORTHING = re.compile(r"\b1[0-9]\d{5}(?:[.,]\d{1,3})?\b")
_TAGS = re.compile(r"<[^>]+>")

MIN_PAIRS = 3                  # moins de trois couples : ce n'est pas un tableau de bornage
MAX_AREA_RATIO = 0.60          # au-delà, c'est le cadre de la page, pas une région

# Un tableau de bornage loge deux colonnes de six et sept chiffres sur au moins trois
# lignes : il ne peut pas être minuscule. Mesuré sur les tableaux réellement localisés du
# corpus — 30,1 %, 23,2 % et 21,1 % de la largeur de la page — contre 3,1 % pour une
# fausse grille trouvée par le repli sur 54987580, qui a fait relire 31 pixels de large au
# lieu du tableau. Sous ce seuil les chiffres seraient sous-pixelliques.
MIN_WIDTH_RATIO = 0.12
MIN_HEIGHT_RATIO = 0.03


@dataclass(frozen=True)
class Region:
    """Une région candidate, avec d'où elle vient et ce qui la rend crédible."""

    bbox: tuple[float, float, float, float]
    source: str                # "contenu" | "etiquette" | "grille"
    pairs: int                 # couples (X, Y) lisibles dedans ; 0 si non transcrite
    note: str

    @property
    def confiance(self) -> str:
        return {"contenu": "certaine", "etiquette": "plausible", "grille": "faible"}[self.source]


def _texte(contenu: object) -> str:
    """Le contenu d'un bloc, débarrassé de son balisage, pour y chercher des nombres."""
    if contenu is None:
        return ""
    if isinstance(contenu, (list, tuple)):
        return " ".join(_texte(c) for c in contenu)
    return _TAGS.sub(" ", str(contenu))


def _pairs(texte: str) -> int:
    """Combien de couples (Est, Nord) plausibles le texte contient."""
    return min(len(_EASTING.findall(texte)), len(_NORTHING.findall(texte)))


def _bbox(bloc: dict) -> tuple[float, float, float, float] | None:
    boite = bloc.get("block_bbox")
    if not boite or len(boite) != 4:
        return None
    x0, y0, x1, y1 = (float(v) for v in boite)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _trop_grande(bbox: tuple[float, float, float, float], taille: tuple[int, int] | None) -> bool:
    if taille is None:
        return False
    largeur, hauteur = taille
    aire = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return aire > MAX_AREA_RATIO * largeur * hauteur


def _trop_petite(bbox: tuple[float, float, float, float], taille: tuple[int, int] | None) -> bool:
    """Trop petite pour contenir un tableau de coordonnées lisible."""
    if taille is None:
        return False
    largeur, hauteur = taille
    return ((bbox[2] - bbox[0]) < MIN_WIDTH_RATIO * largeur
            or (bbox[3] - bbox[1]) < MIN_HEIGHT_RATIO * hauteur)


def locate(raw: dict, taille: tuple[int, int] | None = None) -> list[Region]:
    """Régions candidates pour le tableau de coordonnées, la plus sûre d'abord.

    `raw` est la sortie de mise en page de la passe 1 ; `taille` la largeur et la
    hauteur de la page, qui sert à écarter une région aussi grande que la feuille.
    """
    regions: list[Region] = []
    for bloc in raw.get("parsing_res_list") or []:
        if not isinstance(bloc, dict):
            continue
        bbox = _bbox(bloc)
        if bbox is None or _trop_grande(bbox, taille) or _trop_petite(bbox, taille):
            continue
        etiquette = (bloc.get("block_label") or "").lower()
        couples = _pairs(_texte(bloc.get("block_content")))
        if couples >= MIN_PAIRS:
            regions.append(Region(bbox, "contenu", couples,
                                  f"{couples} couples (Est, Nord) lus dans ce bloc"))
        elif etiquette == "table":
            # Le bloc est reconnu comme un tableau mais son contenu n'a pas été
            # transcrit : sa boîte reste exactement ce qu'il faut pour aller le relire.
            regions.append(Region(bbox, "etiquette", 0,
                                  "bloc reconnu comme tableau, contenu non transcrit"))
    regions.sort(key=lambda r: (r.source != "contenu", -r.pairs))
    return regions


def locate_by_grid(image, taille: tuple[int, int] | None = None) -> list[Region]:
    """Dernier recours : la grille de traits, quand la mise en page n'a rien donné.

    Importé paresseusement : ce repli a besoin d'OpenCV, dont le reste du module
    se passe.
    """
    from app.extraction.locators.grid import find_grids

    regions = []
    for g in find_grids(image):
        bbox = (float(g.x0), float(g.y0), float(g.x1), float(g.y1))
        if _trop_grande(bbox, taille) or _trop_petite(bbox, taille):
            continue
        regions.append(Region(bbox, "grille", 0,
                              f"grille de {g.rows} lignes sur {g.cols} colonnes"))
    return regions


# --------------------------------------------------------------- recadrage

MARGIN_RATIO = 0.06            # les bordures et la ligne d'en-tête aident le lecteur


def crop_region(image, bbox: tuple[float, float, float, float], zoom: int = 1):
    """Le fragment de page correspondant à la région, avec une marge et un grossissement.

    Agrandir ne crée aucun détail, mais un modèle de vision redimensionne son entrée sur
    une grille fixe : en lui donnant le fragment seul, il y consacre toute sa résolution
    au lieu de la répartir sur la feuille entière.
    """
    import cv2

    hauteur, largeur = image.shape[:2]
    x0, y0, x1, y1 = bbox
    mx, my = (x1 - x0) * MARGIN_RATIO, (y1 - y0) * MARGIN_RATIO
    x0, y0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
    x1, y1 = min(largeur, int(x1 + mx)), min(hauteur, int(y1 + my))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    fragment = image[y0:y1, x0:x1]
    if zoom > 1:
        fragment = cv2.resize(fragment, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    return fragment


def best_region(raw: dict | None, image=None) -> Region | None:
    """La meilleure région candidate : la mise en page d'abord, la grille de traits ensuite."""
    regions = locate(raw or {}, (image.shape[1], image.shape[0]) if image is not None else None)
    if not regions and image is not None:
        try:
            regions = locate_by_grid(image, (image.shape[1], image.shape[0]))
        except Exception:                      # noqa: BLE001 - un repli absent n'est pas une panne
            regions = []
    return regions[0] if regions else None
