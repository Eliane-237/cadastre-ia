"""Validation géométrique d'une extraction. Fonctions pures : aucune image, aucun OCR.

Le résultat est une liste d'`Issue` : des alertes d'aide à la décision. Un écart de superficie
peut venir d'une mauvaise lecture, d'une erreur dans le document d'origine, ou d'autre chose :
ce module constate, il ne conclut pas à la fraude.
"""
from __future__ import annotations

import math
import statistics

from shapely.geometry import Polygon

from model.config import SETTINGS, Settings
from model.schema import BoundaryPoint, Extraction, Issue, Severity

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


def _coords(points: list[BoundaryPoint]) -> list[Coord]:
    return [(p.x.value, p.y.value) for p in points]  # type: ignore[misc]  # complétude vérifiée avant


def _drop_closing_point(points: list[BoundaryPoint]) -> list[BoundaryPoint]:
    """Certains tableaux répètent le premier point à la fin pour fermer le polygone."""
    if len(points) >= 4 and points[0].complete and points[-1].complete \
            and points[0].x.value == points[-1].x.value and points[0].y.value == points[-1].y.value:
        return points[:-1]
    return points


# Systemes de reference rencontres sur les titres senegalais. RRS / RSS sont le reseau
# national, exprime en UTM 28N ; HATT est une autre projection, incompatible avec nos
# plages et avec un calcul de surface en metres.
_CRS_UTM28N = ("RRS", "RSS")
_CRS_AUTRES = ("HATT",)


def crs_is_utm28n(declared: str | None) -> bool | None:
    """True si le systeme declare est compatible UTM 28N, False si non, None si inconnu."""
    if not declared:
        return None
    text = declared.upper()
    if any(k in text for k in _CRS_AUTRES):
        return False
    if any(k in text for k in _CRS_UTM28N):
        return True
    return None


def _out_of_range(p: BoundaryPoint, cfg: Settings) -> list[str]:
    bad = []
    if not cfg.easting_range[0] <= p.x.value <= cfg.easting_range[1]:  # type: ignore[operator]
        bad.append(f"X={p.x.value}")
    if not cfg.northing_range[0] <= p.y.value <= cfg.northing_range[1]:  # type: ignore[operator]
        bad.append(f"Y={p.y.value}")
    return bad


def _check_values(points: list[BoundaryPoint], cfg: Settings) -> list[Issue]:
    issues: list[Issue] = []

    # Si la plupart des points sortent de la plage, ce n'est pas une lecture fautive :
    # le document est dans un autre referentiel. Un seul point hors plage, en revanche,
    # est bien un chiffre mal lu (« 38210.57 » pour « 381210.57 »).
    complete = [p for p in points if p.complete]
    hors = [p for p in complete if _out_of_range(p, cfg)]
    if complete and len(hors) > len(complete) / 2:
        return [
            Issue(
                code="CRS_HORS_UTM28N",
                severity=Severity.WARNING,
                point_labels=[p.label for p in hors],
                message=(
                    f"{len(hors)} des {len(complete)} point(s) complets hors de la plage "
                    f"UTM 28N (sur {len(points)} lus) : le document est probablement "
                    "exprimé dans un autre système de coordonnées. Contrôles géométriques "
                    "non applicables."
                ),
            )
        ]

    for p in points:
        if not p.complete:
            missing = [a for a in ("x", "y") if not getattr(p, a).present]
            issues.append(Issue(code="POINT_INCOMPLETE", severity=Severity.ERROR, point_labels=[p.label],
                                message=f"Point {p.label} : coordonnée {', '.join(missing)} non lue."))
            continue
        bad = _out_of_range(p, cfg)
        if bad:
            issues.append(Issue(code="COORD_OUT_OF_RANGE", severity=Severity.ERROR, point_labels=[p.label],
                                message=f"Point {p.label} hors de la plage UTM 28N attendue ({', '.join(bad)})."))
    ok = [p for p in points if p.complete]
    if len(ok) >= 3:
        mx = statistics.median(p.x.value for p in ok)  # type: ignore[misc]
        my = statistics.median(p.y.value for p in ok)  # type: ignore[misc]
        flagged = {lab for i in issues for lab in i.point_labels}
        for p in ok:
            if p.label in flagged:
                continue
            if max(abs(p.x.value - mx), abs(p.y.value - my)) > cfg.max_parcel_extent_m:  # type: ignore[operator]
                issues.append(Issue(code="COORD_OUTLIER", severity=Severity.ERROR, point_labels=[p.label],
                                    message=f"Point {p.label} à plus de {cfg.max_parcel_extent_m:.0f} m "
                                            "des autres : probablement mal lu."))
    return issues


def _regular_polygon_area(side: float, n: int) -> float:
    """Aire du polygone régulier : le maximum atteignable avec n côtés tous égaux."""
    return n * side * side / (4 * math.tan(math.pi / n))


def unreliable_sides(
    printed: list[float | None], printed_m2: float | None, cfg: Settings = SETTINGS
) -> str | None:
    """Pourquoi la colonne des côtés imprimés n'est pas croyable, ou None si elle l'est.

    Cette colonne sert de preuve indépendante pour départager deux lectures d'une
    coordonnée. Mal lue, elle fait condamner des lectures correctes et absoudre des
    fausses : une colonne absente ne coûte qu'une contrainte en moins, une colonne
    fausse coûte la vérité. En cas de doute on la rejette entière, car une colonne
    où une case a mal été découpée n'est fiable sur aucune de ses lignes.
    """
    reels = [c for c in printed if c is not None]
    if not reels:
        return None
    perimetre = sum(reels)

    # Inégalité polygonale : dans tout polygone fermé, un côté est strictement
    # inférieur à la somme de tous les autres.
    for c in reels:
        if c >= perimetre - c:
            return f"côté imprimé {c:.2f} m supérieur à la somme des autres ({perimetre - c:.2f} m)"

    if printed_m2:
        caracteristique = math.sqrt(printed_m2)
        debris = [c for c in reels if c < cfg.sides_fragment_ratio * caracteristique]
        if debris:
            return (
                f"côté(s) imprimé(s) {', '.join(f'{c:.2f}' for c in debris)} m sous "
                f"{cfg.sides_fragment_ratio:.0%} de la longueur caractéristique "
                f"({caracteristique:.2f} m) : découpage de cellule manqué"
            )
        if len(set(reels)) == 1 and len(reels) >= 3:
            maxi = _regular_polygon_area(reels[0], len(reels))
            if abs(maxi - printed_m2) / printed_m2 > cfg.sides_regular_tolerance_pct / 100:
                return (
                    f"colonne constante ({reels[0]:.2f} m sur {len(reels)} lignes) "
                    f"incompatible avec la superficie : un polygone régulier de ce côté "
                    f"ferait {maxi:.0f} m² contre {printed_m2:.0f} m² imprimés ; "
                    "la case a été recopiée"
                )
        # Inégalité isopérimétrique : aucune forme fermée de périmètre P n'enferme
        # plus de P²/4π. Ne vaut que si la colonne est complète.
        if len(reels) == len(printed) and printed_m2 > perimetre**2 / (4 * math.pi):
            return (
                f"superficie imprimée ({printed_m2:.0f} m²) supérieure au maximum "
                f"qu'enferme ce périmètre ({perimetre:.2f} m donne au plus "
                f"{perimetre ** 2 / (4 * math.pi):.0f} m²)"
            )
    return None


def _check_sides(
    points: list[BoundaryPoint], printed: list, printed_m2: float | None, cfg: Settings
) -> list[Issue]:
    # Une colonne mal lue produirait un côté fautif par ligne et ferait accuser des
    # coordonnées correctes : on la rejette d'un bloc, avec sa raison.
    if raison := unreliable_sides(
        [d.value if d.present else None for d in printed], printed_m2, cfg
    ):
        return [
            Issue(
                code="PRINTED_SIDES_UNRELIABLE",
                severity=Severity.WARNING,
                message=f"Longueurs imprimées écartées — {raison}.",
            )
        ]
    n = len(points)
    coords = _coords(points)
    failing: list[int] = []
    for i in range(min(n, len(printed))):
        d = printed[i]
        if not d.present:
            continue
        calc = side_length(coords[i], coords[(i + 1) % n])
        if abs(calc - d.value) > cfg.side_tolerance_m:
            failing.append(i)
    issues = [Issue(code="SIDE_MISMATCH", severity=Severity.WARNING,
                    point_labels=[points[i].label, points[(i + 1) % n].label],
                    message=f"Côté {points[i].label}-{points[(i + 1) % n].label} : calculé "
                            f"{side_length(coords[i], coords[(i + 1) % n]):.2f} m, imprimé "
                            f"{printed[i].value:.2f} m.") for i in failing]
    # Un sommet commun à deux côtés fautifs consécutifs est le suspect le plus probable.
    fset = set(failing)
    for i in sorted(fset):
        if (i - 1) % n in fset and n > 2:
            issues.append(Issue(code="POINT_SUSPECT", severity=Severity.WARNING, point_labels=[points[i].label],
                                message=f"Point {points[i].label} : les deux côtés qui y aboutissent "
                                        "sont incohérents avec le plan."))
    return issues


def validate(extraction: Extraction, cfg: Settings = SETTINGS) -> list[Issue]:
    """Contrôles de cohérence. Les contrôles géométriques ne sont pas faits si une coordonnée est
    manquante ou aberrante : calculer sur des valeurs connues pour fausses induirait en erreur."""
    points = _drop_closing_point(extraction.points)
    issues: list[Issue] = []

    declared = extraction.coordinate_system.value if extraction.coordinate_system.present else None
    if crs_is_utm28n(declared) is False:
        issues.append(
            Issue(
                code="CRS_NON_SUPPORTE",
                severity=Severity.WARNING,
                message=(
                    f"Système de coordonnées « {declared} » : ni la plage attendue ni le "
                    "calcul de surface en UTM 28N ne s'appliquent. Géométrie non contrôlée."
                ),
            )
        )
        return issues

    issues += _check_values(points, cfg)

    if any(i.severity is Severity.ERROR for i in issues) or any(
        i.code == "CRS_HORS_UTM28N" for i in issues
    ):
        issues.append(Issue(code="GEOMETRY_SKIPPED", severity=Severity.INFO,
                            message="Superficie et côtés non contrôlés : des coordonnées sont à corriger d'abord."))
        return issues
    if len(points) < 3:
        issues.append(Issue(code="TOO_FEW_POINTS", severity=Severity.ERROR,
                            message=f"{len(points)} point(s) lu(s) : au moins 3 sont nécessaires."))
        return issues

    coords = _coords(points)
    if not Polygon(coords).is_valid:
        issues.append(Issue(code="POLYGON_INVALID", severity=Severity.ERROR,
                            point_labels=[p.label for p in points],
                            message="Le polygone se recoupe lui-même : ordre des points ou lecture à vérifier."))

    area = shoelace_area(coords)
    printed_m2: float | None = None
    if extraction.printed_area.present:
        printed = extraction.printed_area.value.m2  # type: ignore[union-attr]
        printed_m2 = float(printed) if printed else None
        gap = abs(area - printed) / printed * 100 if printed else math.inf
        if gap > cfg.area_tolerance_pct:
            issues.append(Issue(code="AREA_MISMATCH", severity=Severity.ERROR,
                                message=f"Superficie calculée {area:.1f} m² contre {printed:.0f} m² imprimés "
                                        f"(écart {gap:.1f} %, tolérance {cfg.area_tolerance_pct} %)."))
    else:
        issues.append(Issue(code="AREA_UNAVAILABLE", severity=Severity.INFO,
                            message="Superficie imprimée non lue : cohérence non vérifiée."))

    issues.extend(_check_sides(points, extraction.printed_sides, printed_m2, cfg))
    return issues


def validated(extraction: Extraction, cfg: Settings = SETTINGS) -> Extraction:
    """Copie de l'extraction avec les anomalies détectées ajoutées."""
    return extraction.model_copy(update={"issues": [*extraction.issues, *validate(extraction, cfg)]})