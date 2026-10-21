"""Relecture ciblee du tableau de bornage : recadre la region, l'agrandit, la relit seule.

Lire une page entiere dilue l'attention du modele sur une zone ou les chiffres occupent
quelques pour cent de la surface. Mesure sur tf.jpg : le meme tableau, lu dans une region
de 148x73 px au lieu de la page, passe de trois chiffres faux a zero.

Cet outil ne modifie pas le pipeline : il compare, pour un document, les coordonnees de la
passe 1 (page entiere) a celles d'une relecture du seul fragment, a plusieurs grossissements.

    python tools/table_rescan.py "data/raw/images (5).jfif" \
        "reports/readings/images (5)_readings.json"
    python tools/table_rescan.py data/raw/tf.jpg reports/readings/tf_readings.json --scales 1 2 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import from_env  # noqa: E402
from app.document.loader import load_pages
from app.document.preprocess import preprocess  # noqa: E402
from tools.table_by_content import candidates  # noqa: E402

# Marge autour de la region : les bordures du tableau et la ligne d'en-tete aident le modele.
MARGIN_RATIO = 0.06


def locate_table(readings_path: Path) -> tuple[list[float], str] | None:
    """Boite du bloc contenant le plus de coordonnees, d'apres la passe 1."""
    import json

    data = json.loads(readings_path.read_text(encoding="utf-8"))
    best, best_reader = None, None
    for reader, reading in data.items():
        raw = (reading or {}).get("raw_output") or {}
        for cand in candidates(raw):
            if cand.bbox and (best is None or cand.pairs > best.pairs):
                best, best_reader = cand, reader
    return (list(best.bbox), best_reader) if best else None


def crop(image: np.ndarray, bbox: list[float], scale: int) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    mx = (x1 - x0) * MARGIN_RATIO
    my = (y1 - y0) * MARGIN_RATIO
    x0 = max(0, int(x0 - mx))
    y0 = max(0, int(y0 - my))
    x1 = min(w, int(x1 + mx))
    y1 = min(h, int(y1 + my))
    fragment = image[y0:y1, x0:x1]
    if scale > 1:
        fragment = cv2.resize(
            fragment, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    return fragment


def read_points(reader, image: np.ndarray) -> list[tuple[str | None, str | None, str | None]]:
    try:
        reading = reader.read(image)
    except Exception as exc:
        print(f"      ECHEC {type(exc).__name__}: {exc}")
        return []
    return [(p.label, p.x, p.y) for p in reading.points]


def show(title: str, points: list) -> None:
    print(f"    {title} : {len(points)} point(s)")
    for label, x, y in points[:20]:
        print(f"      {str(label):<6} {str(x):<14} {str(y)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Relit le seul tableau de bornage, agrandi.")
    ap.add_argument("document", type=Path)
    ap.add_argument("readings", type=Path, help="Le *_readings.json de la passe 1.")
    ap.add_argument("--scales", type=int, nargs="+", default=[1, 3],
                    help="Grossissements a comparer (1 = fragment a sa taille).")
    ap.add_argument("--reader", choices=["qwen", "paddle", "both"], default="both")
    ap.add_argument("--save", type=Path, default=Path("reports/rescan"),
                    help="Ou ecrire les fragments, pour verifier le cadrage.")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="Region du tableau, en pixels de l'image d'origine. Court-circuite "
                         "toute localisation : utile quand la mise en page ne voit rien.")
    ap.add_argument("--truth", type=Path,
                    help="Annotation JSON du document : la relecture est alors comparee "
                         "coordonnee par coordonnee a la verite terrain.")
    args = ap.parse_args()

    source = None
    bbox = tuple(args.bbox) if args.bbox else None
    if bbox is None:
        found = locate_table(args.readings)
        if found is not None:
            bbox, source = found
    if bbox is None:
        # La passe 1 n'a rien vu : on redemande au localisateur, qui sait aussi
        # exploiter un bloc reconnu comme tableau dont le contenu n'a pas ete transcrit.
        import json as _json

        from app.extraction.locators.table import locate

        donnees = _json.loads(args.readings.read_text(encoding="utf-8"))
        regions = []
        for lecture in donnees.values():
            regions += locate((lecture or {}).get("raw_output") or {})
        if regions:
            bbox, source = regions[0].bbox, f"table_locator ({regions[0].confiance})"
    if bbox is None:
        print("Aucune region de tableau localisee. Donner --bbox X0 Y0 X1 Y1 pour "
              "indiquer la region a la main.")
        return 1
    print(f"Region du tableau ({source or 'donnee a la main'}) : "
          f"[{', '.join(f'{v:.0f}' for v in bbox)}]")

    cfg = from_env()
    page = load_pages(args.document)[0]
    if args.bbox:
        # La region a ete mesuree sur l'image d'origine : la redresser la deplacerait.
        image = page
        print(f"Page d'origine (sans pretraitement) : {image.shape[1]}x{image.shape[0]} px")
    else:
        image, _ = preprocess(page)
        print(f"Page prétraitée : {image.shape[1]}x{image.shape[0]} px")

    verite = None
    if args.truth:
        import json as _json

        ann = _json.loads(args.truth.read_text(encoding="utf-8"))
        verite = {p["label"]: (p["x"], p["y"]) for p in ann.get("points", [])}
        print(f"Verite terrain : {len(verite)} bornes ({args.truth})")

    readers = []
    if args.reader in ("paddle", "both"):
        from app.extraction.readers.paddle import PaddleVLReader
        readers.append(("paddle", PaddleVLReader(cfg.paddle)))
    if args.reader in ("qwen", "both"):
        from app.extraction.readers.qwen import QwenVLReader
        readers.append(("qwen", QwenVLReader(cfg.qwen)))

    args.save.mkdir(parents=True, exist_ok=True)
    stem = args.document.stem

    for scale in args.scales:
        fragment = crop(image, bbox, scale)
        path = args.save / f"{stem}_x{scale}.png"
        cv2.imwrite(str(path), fragment)
        print(f"\n=== grossissement x{scale} — fragment {fragment.shape[1]}x{fragment.shape[0]} px "
              f"({path}) ===")
        for name, reader in readers:
            lus = read_points(reader, fragment)
            show(name, lus)
            if verite:
                bons = sum(
                    1
                    for lab, x, y in lus
                    for attendu, valeur in zip(verite.get(lab or "", (None, None)),
                                               (x, y), strict=False)
                    if attendu is not None and str(valeur) == str(attendu)
                )
                total = 2 * len(verite)
                print(f"      -> {bons}/{total} coordonnees exactes ({bons / total:.0%})")
                for lab, x, y in lus:
                    attendu = verite.get(lab or "")
                    if attendu and (str(x) != attendu[0] or str(y) != attendu[1]):
                        print(f"         {lab}: lu {x} / {y}   attendu {attendu[0]} / {attendu[1]}")

    print("\nComparer ces points à ceux de la passe 1 : un chiffre qui change est un gain "
          "ou une perte, selon la vérité terrain du document.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
