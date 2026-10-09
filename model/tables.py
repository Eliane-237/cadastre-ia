"""Tableaux de coordonnées : HTML (sortie de PaddleOCR-VL) → points et côtés.

PaddleOCR-VL rend chaque tableau en HTML (lignes et colonnes). Ce module retrouve, parmi les
tableaux de la page, celui des coordonnées, et identifie ses colonnes :
- d'abord par les en-têtes (« Borne », « X », « Y », « Est », « Nord », « Distance »…) ;
- sinon par la forme des valeurs : en UTM 28N au Sénégal, X a 6 chiffres entiers et Y 7.
Les valeurs restent des chaînes : la conversion se fait après comparaison des lecteurs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

from model.numbers import canonical_coordinate, canonical_length, fix_digits, integer_digits
from model.readers.base import PointReading, SideReading

_LABEL = re.compile(r"^[A-Za-z]{0,3}\s?[-.]?\s?\d{1,3}[A-Za-z]?$")
_SIDE_PAIR = re.compile(
    r"^\s*([A-Za-z]{0,3}\s?\d{1,3})\s*(?:-|–|—|à|a|/)\s*([A-Za-z]{0,3}\s?\d{1,3})\s*$"
)
_HEADER_X = re.compile(r"^\s*(x|e|est|easting|abscisse)s?\b", re.IGNORECASE)
_HEADER_Y = re.compile(r"^\s*(y|n|nord|northing|ordonn[ée]e)s?\b", re.IGNORECASE)
_HEADER_LABEL = re.compile(r"born|point|sommet|^\s*n\s?[°ºo]|^\s*pts?\b|^\s*rep", re.IGNORECASE)
_HEADER_DIST = re.compile(r"dist|long", re.IGNORECASE)
_HEADER_PAIR = re.compile(r"c[ôo]t[ée]|segment", re.IGNORECASE)
# Nord (7 chiffres, commence par 1) cherché d'abord, puis Est (6 chiffres) dans le reste.
# Séparateurs de milliers cohérents (« 1 564 042 », « 1.564.042 », « 1564042 ») : sinon le « 1 »
# d'un libellé « B1 » suivi d'un X serait pris pour le début d'un Y.
_NORTHING = re.compile(r"(?<![\d.,])1(?:\d{6}|\s\d{3}\s\d{3}|\.\d{3}\.\d{3})(?:[.,]\d{1,3})?(?!\d)")
_EASTING = re.compile(r"(?<![\d.,])\d{3}[\s.]?\d{3}(?:[.,]\d{1,3})?(?!\d)")


class _TableParser(HTMLParser):
    """HTML de tableau → grille, colspan/rowspan dépliés (la cellule est répétée)."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[tuple[str, int, int]] | None = None
        self._cell: list[str] | None = None
        self._span = (1, 1)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
            self._span = (int(a.get("rowspan") or 1), int(a.get("colspan") or 1))
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(("".join(self._cell).strip(), *self._span))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)  # type: ignore[arg-type]
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_to_grid(html: str) -> list[list[str]]:
    p = _TableParser()
    p.feed(html)
    grid: list[list[str]] = []
    pending: dict[int, tuple[str, int]] = {}  # colonne -> (texte, lignes restantes)
    for raw_row in p.rows:
        row: list[str] = []
        cells = iter(raw_row)
        col = 0
        while True:
            if col in pending:
                text, left = pending[col]
                row.append(text)
                if left > 1:
                    pending[col] = (text, left - 1)
                else:
                    del pending[col]
                col += 1
                continue
            try:
                text, rowspan, colspan = next(cells)
            except StopIteration:
                break
            for _ in range(colspan):
                row.append(text)
                if rowspan > 1:
                    pending[col] = (text, rowspan - 1)
                col += 1
        while col in pending:  # rowspans en fin de ligne
            text, left = pending.pop(col)
            row.append(text)
            if left > 1:
                pending[col] = (text, left - 1)
            col += 1
        grid.append(row)
    width = max((len(r) for r in grid), default=0)
    return [r + [""] * (width - len(r)) for r in grid]


# --------------------------------------------------------------------------- colonnes


def _split_coords(text: str) -> tuple[list[str], list[str]]:
    """X et Y fusionnés dans un même texte : (valeurs Est, valeurs Nord), reconnues par forme."""
    text = fix_digits(text)
    ys = [m.group() for m in _NORTHING.finditer(text)]
    rest = _NORTHING.sub(lambda m: " " * len(m.group()), text)
    xs = [m.group() for m in _EASTING.finditer(rest)]
    return xs, ys


def _coord_kind(cell: str) -> str | None:
    c = canonical_coordinate(cell)
    if c is None:
        return None
    return {6: "x", 7: "y"}.get(integer_digits(c))


@dataclass
class CoordinateTable:
    points: list[PointReading]
    sides: list[SideReading]
    header_row: int | None
    columns: dict[str, int]


def _column_profile(rows: list[list[str]], col: int) -> dict[str, float]:
    cells = [r[col] for r in rows if r[col].strip()]
    n = len(cells) or 1
    return {
        "x": sum(_coord_kind(c) == "x" for c in cells) / n,
        "y": sum(_coord_kind(c) == "y" for c in cells) / n,
        "label": sum(bool(_LABEL.match(c.strip())) for c in cells) / n,
        "length": sum(
            canonical_length(c) is not None and "." in (canonical_length(c) or "") for c in cells
        )
        / n,
        "pair": sum(bool(_SIDE_PAIR.match(c)) for c in cells) / n,
        "filled": len(cells) / max(len(rows), 1),
    }


def _find_header(grid: list[list[str]]) -> int | None:
    for i, row in enumerate(grid[:3]):
        hits = sum(
            bool(
                _HEADER_X.match(c)
                or _HEADER_Y.match(c)
                or _HEADER_LABEL.search(c)
                or _HEADER_DIST.search(c)
                or _HEADER_PAIR.search(c)
            )
            for c in row
        )
        has_coord = any(_coord_kind(c) for c in row)
        if hits >= 2 and not has_coord:
            return i
    return None


def parse_coordinate_table(grid: list[list[str]]) -> CoordinateTable | None:
    """Identifie les colonnes et lit les points ; None si ce n'est pas un tableau de coordonnées."""
    if not grid or not grid[0]:
        return None
    header = _find_header(grid)
    body = grid[header + 1 :] if header is not None else grid
    if not body:
        return None
    ncols = len(grid[0])
    profiles = [_column_profile(body, c) for c in range(ncols)]
    cols: dict[str, int] = {}

    if header is not None:
        for c, h in enumerate(grid[header]):
            if _HEADER_X.match(h) and "x" not in cols:
                cols["x"] = c
            elif _HEADER_Y.match(h) and "y" not in cols:
                cols["y"] = c
            elif _HEADER_DIST.search(h) and "length" not in cols:
                cols["length"] = c
            elif _HEADER_PAIR.search(h) and "pair" not in cols:
                cols["pair"] = c
            elif _HEADER_LABEL.search(h) and "label" not in cols:
                cols["label"] = c
    # Les valeurs l'emportent sur un en-tête contradictoire (X/Y inversés dans l'en-tête).
    for kind in ("x", "y"):
        best = max(range(ncols), key=lambda c: profiles[c][kind])
        if profiles[best][kind] >= 0.5:
            cols[kind] = best
    if "x" not in cols or "y" not in cols or cols["x"] == cols["y"]:
        merged = _merged_coordinates(body)
        return merged
    used = {cols["x"], cols["y"]}
    if "label" not in cols:
        cands = [c for c in range(ncols) if c not in used and profiles[c]["label"] >= 0.5]
        if cands:
            cols["label"] = min(cands)  # le libellé est en général la première colonne
    if "pair" not in cols:
        cands = [c for c in range(ncols) if c not in used and profiles[c]["pair"] >= 0.5]
        if cands:
            cols["pair"] = cands[0]
    if "length" not in cols:
        cands = [
            c
            for c in range(ncols)
            if c not in used | {cols.get("label")} and profiles[c]["length"] >= 0.5
        ]
        if cands:
            cols["length"] = max(cands, key=lambda c: profiles[c]["length"])

    points: list[PointReading] = []
    row_of_point: list[int] = []
    sides: list[SideReading] = []
    for r, row in enumerate(body):
        x, y = row[cols["x"]].strip(), row[cols["y"]].strip()
        label = row[cols["label"]].strip() if "label" in cols else None
        length = row[cols["length"]].strip() if "length" in cols else ""
        if x or y:
            points.append(
                PointReading(label=label or None, x=x or None, y=y or None, row_index=len(points))
            )
            row_of_point.append(r)
            if length:
                pair = _SIDE_PAIR.match(row[cols["pair"]]) if "pair" in cols else None
                sides.append(
                    SideReading(
                        length=length,
                        from_label=pair.group(1) if pair else None,
                        to_label=pair.group(2) if pair else None,
                        row_index=None if pair else len(points) - 1,
                    )
                )
        elif length:
            # Ligne intercalaire « côté » entre deux sommets.
            pair = _SIDE_PAIR.match(row[cols["pair"]]) if "pair" in cols else None
            sides.append(
                SideReading(
                    length=length,
                    from_label=pair.group(1) if pair else None,
                    to_label=pair.group(2) if pair else None,
                    row_index=None if pair else max(len(points) - 1, 0),
                )
            )
    if len(points) < 2:
        return None
    return CoordinateTable(points=points, sides=sides, header_row=header, columns=cols)


def _merged_coordinates(body: list[list[str]]) -> CoordinateTable | None:
    """Cas où l'OCR a fusionné X et Y dans une même cellule : on lit ligne par ligne."""
    points = []
    for row in body:
        xs, ys = _split_coords("  ".join(row))
        if xs and ys:
            label = next((c.strip() for c in row if _LABEL.match(c.strip())), None)
            points.append(PointReading(label=label, x=xs[0], y=ys[0], row_index=len(points)))
    if len(points) < 2:
        return None
    return CoordinateTable(points=points, sides=[], header_row=None, columns={})


def best_coordinate_table(html_tables: list[str]) -> CoordinateTable | None:
    """Le tableau de la page qui contient le plus de points complets."""
    best: CoordinateTable | None = None
    for html in html_tables:
        t = parse_coordinate_table(html_to_grid(html))
        if t is None:
            continue
        score = sum(1 for p in t.points if p.x and p.y)
        if best is None or score > sum(1 for p in best.points if p.x and p.y):
            best = t
    return best


_TEXT_LABEL = re.compile(r"^\s*([A-Za-z]{0,3}\s?\d{1,3})(?=[\s:.)\-])")


def points_from_text(text: str) -> list[PointReading]:
    """Repli quand le tableau n'a pas été détecté comme tel : lignes « B1 381210.57 1564042.96 »."""
    points = []
    for line in fix_digits(text).splitlines():
        xs, ys = _split_coords(line)
        if xs and ys:
            m = _TEXT_LABEL.match(line)
            label = m.group(1).strip() if m and not line.strip().startswith(xs[0]) else None
            points.append(PointReading(label=label, x=xs[0], y=ys[0], row_index=len(points)))
    return points
