"""Donne à voir la détection de grilles sur un document ou un dossier.

    python app/extraction/locators/grid.py data/raw/tf.jpg
    python app/extraction/locators/grid.py data/raw --out reports/table_detect

Le calcul est dans `app/extraction/locators/grid.py` ; cet outil ne fait que l'exécuter
et dessiner le résultat.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.extraction.locators.grid import Grid, find_grids  # noqa: E402,F401


def annotate(image: np.ndarray, grids: list[Grid]) -> np.ndarray:
    out = image.copy()
    for i, g in enumerate(grids[:5]):
        color = (0, 200, 0) if i == 0 else (0, 150, 255)
        cv2.rectangle(out, (g.x0, g.y0), (g.x1, g.y1), color, 2)
        cv2.putText(out, f"#{i+1} {g.rows}x{g.cols}", (g.x0, max(16, g.y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Detecte les tableaux traces d'un document.")
    ap.add_argument("source", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/table_detect"))
    ap.add_argument("--no-preprocess", action="store_true")
    args = ap.parse_args()

    try:
        from app.document.loader import load_pages
        from app.document.preprocess import preprocess
    except ImportError:
        load_pages = preprocess = None

    files = (sorted(p for p in args.source.iterdir() if p.is_file())
             if args.source.is_dir() else [args.source])
    args.out.mkdir(parents=True, exist_ok=True)
    report, found = {}, 0

    for path in files:
        try:
            if load_pages is not None:
                image = load_pages(path)[0]
                if not args.no_preprocess:
                    image, _ = preprocess(image)
            else:
                image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("illisible")
        except Exception as exc:
            print(f"{path.name:<46} ERREUR {type(exc).__name__}: {exc}")
            continue

        grids = find_grids(image)
        cv2.imwrite(str(args.out / f"{path.stem}.png"), annotate(image, grids))
        report[path.name] = [asdict(g) for g in grids[:5]]
        if grids:
            found += 1
            g = grids[0]
            print(f"{path.name:<46} {len(grids)} grille(s) | {g.rows}x{g.cols}, "
                  f"{g.nodes} noeuds, {g.area_ratio*100:4.1f}% page")
        else:
            print(f"{path.name:<46} AUCUNE GRILLE")

    (args.out / "detections.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{found}/{len(files)} documents avec au moins une grille. Sorties : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
