"""Passe `find_repair` sur toutes les extractions deja produites, sans rien modifier.

    python -m tools.repair_scan

Lit `reports/readings/*.json` (les extractions fusionnees : points, superficie
imprimee, longueurs de cotes) et affiche, document par document, la reparation
que `app.extraction.digit_repair` proposerait. Purement diagnostique : n'ecrit aucun
fichier et n'est appele par aucun module de `app/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.normalize import parse_area  # noqa: E402
from app.extraction.digit_repair import find_repair, shoelace  # noqa: E402
from app.validation.rules import unreliable_sides  # noqa: E402

READINGS = Path("reports/readings")


def _printed_m2(extraction: dict) -> float | None:
    """Le texte imprime fait foi ; la structure du lecteur ne sert que de secours.

    Meme regle que `app.extraction.merge` : un lecteur range parfois les ares
    dans les hectares (« 44a 89ca » lu ha=44, a=89), ce qui multiplierait la
    superficie par cent et fausserait tout le jugement geometrique.
    """
    field = extraction.get("printed_area") or {}
    raw = field.get("raw_text")
    if isinstance(raw, str):
        area = parse_area(raw)
        if area is not None and area.m2:
            return float(area.m2)
    value = field.get("value")
    if not isinstance(value, dict):
        return None
    m2 = (value.get("ha") or 0) * 10_000 + (value.get("a") or 0) * 100 + (value.get("ca") or 0)
    return float(m2) if m2 else None


def _raw(cell: dict | None, fallback: float | None) -> str | None:
    if not isinstance(cell, dict):
        return None
    raw = cell.get("raw_text")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if isinstance(fallback, (int, float)):
        return f"{fallback:.2f}"
    return None


def _coords(extraction: dict) -> tuple[list[str], list[str]] | None:
    xs: list[str] = []
    ys: list[str] = []
    for point in extraction.get("points") or []:
        x = point.get("x") or {}
        y = point.get("y") or {}
        rx = _raw(x, x.get("value"))
        ry = _raw(y, y.get("value"))
        if rx is None or ry is None:
            return None
        xs.append(rx)
        ys.append(ry)
    if len(xs) < 3:
        return None
    return xs, ys


def _sides(extraction: dict,
           printed_m2: float | None) -> tuple[list[float | None] | None, str | None]:
    sides = [s.get("value") for s in (extraction.get("printed_sides") or [])]
    if not any(s is not None for s in sides):
        return None, None
    if raison := unreliable_sides(sides, printed_m2):
        return None, raison
    return sides, None


def scan(path: Path, tolerance_pct: float, side_tolerance_m: float) -> str:
    extraction = json.loads(path.read_text(encoding="utf-8"))
    name = path.stem

    # Les fichiers de lectures appariees ({paddle_vl, qwen_vl}) ne sont pas des
    # extractions : les compter comme documents faussait le decompte du corpus.
    if "points" not in extraction and {"paddle_vl", "qwen_vl"} & extraction.keys():
        return ""

    coords = _coords(extraction)
    if coords is None:
        return f"{name:42s} -- moins de 3 points complets, rien a verifier"
    xs, ys = coords

    printed = _printed_m2(extraction)
    if printed is None:
        return f"{name:42s} -- superficie imprimee absente, pas de juge geometrique"

    points = [(float(x), float(y)) for x, y in zip(xs, ys, strict=False)]
    area = shoelace(points)
    gap = abs(area - printed) / printed * 100

    sides, raison = _sides(extraction, printed)
    if raison:
        marque = f"  (cotes ecartes : {raison})"
    elif not sides:
        marque = "  (sans longueurs de cotes)"
    else:
        marque = ""

    repair = find_repair(
        xs,
        ys,
        printed,
        tolerance_pct=tolerance_pct,
        printed_sides=sides,
        side_tolerance_m=side_tolerance_m,
    )

    if repair is None:
        if gap <= tolerance_pct:
            return (
                f"{name:42s} OK  {area:12,.0f} m2 vs {printed:10,.0f} imprimes "
                f"({gap:5.2f} %) -- deja coherent"
            )
        return (
            f"{name:42s} --  {area:12,.0f} m2 vs {printed:10,.0f} imprimes "
            f"({gap:5.2f} %) -- aucune reparation a un seul chiffre{marque}"
        )

    return (
        f"{name:42s} ++  {repair.message}\n"
        f"{'':42s}     {repair.area_before:,.0f} m2 -> {repair.area_after:,.0f} m2 "
        f"pour {printed:,.0f} imprimes{marque}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance-pct", type=float, default=0.5)
    parser.add_argument("--side-tolerance-m", type=float, default=0.05)
    parser.add_argument("--dir", type=Path, default=READINGS)
    args = parser.parse_args()

    fichiers = sorted(
        p for p in args.dir.glob("*.json") if not p.stem.endswith("_readings")
    )
    if not fichiers:
        print(f"aucune extraction dans {args.dir}")
        return

    lignes = [scan(p, args.tolerance_pct, args.side_tolerance_m) for p in fichiers]
    lignes = [ligne for ligne in lignes if ligne]
    print(f"{len(lignes)} extraction(s) retenue(s)\n")
    for ligne in lignes:
        print(ligne)


if __name__ == "__main__":
    main()
