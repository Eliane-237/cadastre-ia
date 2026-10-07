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


def _check_values(points: list[BoundaryPoint], cfg: Settings) -> list[Issue]:
    issues: list[Issue] = []
    for p in points:
        if not p.complete:
            missing = [a for a in ("x", "y") if not getattr(p, a).present]
            issues.append(Issue(code="POINT_INCOMPLETE", severity=Severity.ERROR, point_labels=[p.label],
                                message=f"Point {p.label} : coordonnée {', '.join(missing)} non lue."))
            continue
        bad = []
        if not cfg.easting_range[0] <= p.x.value <= cfg.easting_range[1]:  # type: ignore[operator]
            bad.append(f"X={p.x.value}")
        if not cfg.northing_range[0] <= p.y.value <= cfg.northing_range[1]:  # type: ignore[operator]
            bad.append(f"Y={p.y.value}")
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


def _check_sides(points: list[BoundaryPoint], printed: list, cfg: Settings) -> list[Issue]:
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
    issues = _check_values(points, cfg)

    if any(i.severity is Severity.ERROR for i in issues):
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
    if extraction.printed_area.present:
        printed = extraction.printed_area.value.m2  # type: ignore[union-attr]
        gap = abs(area - printed) / printed * 100 if printed else math.inf
        if gap > cfg.area_tolerance_pct:
            issues.append(Issue(code="AREA_MISMATCH", severity=Severity.ERROR,
                                message=f"Superficie calculée {area:.1f} m² contre {printed:.0f} m² imprimés "
                                        f"(écart {gap:.1f} %, tolérance {cfg.area_tolerance_pct} %)."))
    else:
        issues.append(Issue(code="AREA_UNAVAILABLE", severity=Severity.INFO,
                            message="Superficie imprimée non lue : cohérence non vérifiée."))

    issues.extend(_check_sides(points, extraction.printed_sides, cfg))
    return issues


def validated(extraction: Extraction, cfg: Settings = SETTINGS) -> Extraction:
    """Copie de l'extraction avec les anomalies détectées ajoutées."""
    return extraction.model_copy(update={"issues": [*extraction.issues, *validate(extraction, cfg)]})