"""Detection de tableaux traces par morphologie, independante de tout modele.

Les tableaux de bornage sont encadres. On isole les traits horizontaux et verticaux par
ouverture morphologique, on croise les deux masques pour obtenir les NOEUDS de la grille,
puis on regroupe ces noeuds par proximite : chaque groupe regulier est une grille. Travailler
sur les noeuds plutot que sur les traits evite de fusionner le tableau avec le cadre de la page.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

MIN_LINE_RATIO = 0.006     # longueur minimale d'un trait, en fraction du cote de la page
MIN_ROWS, MIN_COLS = 2, 2  # au moins 2 lignes et 2 colonnes de cellules
MIN_NODES = 9              # une grille 2x2 a 9 noeuds
MAX_AREA_RATIO = 0.55      # au-dela, c'est le cadre de la page


@dataclass
class Grid:
    x0: int
    y0: int
    x1: int
    y1: int
    rows: int
    cols: int
    nodes: int
    area_ratio: float

    @property
    def score(self) -> float:
        """Un tableau de bornage : 2 a 5 colonnes, plusieurs lignes, densite de noeuds elevee."""
        col_fit = 1.0 if 2 <= self.cols <= 5 else 0.35
        filled = self.nodes / float((self.rows + 1) * (self.cols + 1))
        return col_fit * min(self.rows, 25) * min(filled, 1.0)


def _cluster_1d(values: np.ndarray, tol: float) -> int:
    """Nombre de positions distinctes, les valeurs proches etant regroupees."""
    if values.size == 0:
        return 0
    v = np.sort(values)
    return int(1 + np.count_nonzero(np.diff(v) > tol))


def _node_points(binary: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    hk = max(8, int(w * MIN_LINE_RATIO))
    vk = max(8, int(h * MIN_LINE_RATIO))
    horizontal = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    )
    vertical = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    )
    # Les traits sont rarement parfaitement jointifs : on les epaissit avant de les croiser.
    pad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    nodes = cv2.bitwise_and(cv2.dilate(horizontal, pad), cv2.dilate(vertical, pad))

    n, _, stats, centroids = cv2.connectedComponentsWithStats(nodes, connectivity=8)
    pts = np.array([centroids[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 4])
    return pts, nodes


def _group(points: np.ndarray, tol: float) -> list[np.ndarray]:
    """Regroupe les noeuds en grilles : deux noeuds voisins a moins de `tol` sont du meme
    tableau. Union-find simple, suffisant pour quelques centaines de points."""
    n = len(points)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        d = np.abs(points[i + 1 :] - points[i])
        close = np.nonzero((d[:, 0] <= tol) & (d[:, 1] <= tol))[0]
        for j in close:
            a, b = find(i), find(i + 1 + j)
            if a != b:
                parent[a] = b

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [points[idx] for idx in groups.values() if len(idx) >= MIN_NODES]


def find_grids(image: np.ndarray) -> list[Grid]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 15
    )
    points, _ = _node_points(binary, w, h)
    if len(points) < MIN_NODES:
        return []

    # Tolerance : un peu plus qu'une hauteur de cellule, estimee par les ecarts verticaux.
    ys = np.sort(points[:, 1])
    gaps = np.diff(ys)
    gaps = gaps[gaps > 2]
    cell = float(np.median(gaps)) if gaps.size else h * 0.03
    tol = max(12.0, cell * 3.0)

    grids: list[Grid] = []
    for g in _group(points, tol):
        x0, y0 = g.min(axis=0)
        x1, y1 = g.max(axis=0)
        gw, gh = x1 - x0, y1 - y0
        if gw < 20 or gh < 20:
            continue
        ratio = (gw * gh) / float(w * h)
        if ratio > MAX_AREA_RATIO:
            continue
        rows = _cluster_1d(g[:, 1], max(4.0, gh * 0.02)) - 1
        cols = _cluster_1d(g[:, 0], max(4.0, gw * 0.03)) - 1
        if rows < MIN_ROWS or cols < MIN_COLS:
            continue
        grids.append(Grid(int(x0), int(y0), int(x1), int(y1), rows, cols, len(g), ratio))

    grids.sort(key=lambda z: z.score, reverse=True)
    return grids


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
        from model.preprocess import load_pages, preprocess
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
