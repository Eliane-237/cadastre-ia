"""Banc d'essai du prétraitement et de la détection de mise en page.

Chaque variante ne change qu'une chose à la fois. On regarde les régions détectées et,
surtout, le contenu du tableau de bornes : c'est là que se juge la qualité réelle.

    python tools\\layout_sweep.py data\\raw\\tf.jpg
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.config import from_env                      # noqa: E402
from model.preprocess import load_pages, preprocess    # noqa: E402

_TAGS = re.compile(r"<[^>]+>")

# (nom, options de prétraitement, surcharges de mise en page).
# La configuration de mise en page est celle retenue ; seul le redressement varie.
VARIANTS: list[tuple[str, dict, dict]] = [
    ("référence (sans redressement)", {}, {}),
    ("deskew=True", {"deskew": True}, {}),
]


def flatten(content: str, width: int = 110) -> str:
    text = _TAGS.sub(" ", str(content or ""))
    return re.sub(r"\s+", " ", text).strip()[:width]


def run(image, cfg, overrides: dict):
    from paddleocr import PaddleOCRVL

    s = cfg.paddle
    kwargs: dict = {
        "pipeline_version": s.pipeline_version,
        "use_doc_orientation_classify": s.use_doc_orientation_classify,
        "use_doc_unwarping": s.use_doc_unwarping,
        "vl_rec_backend": s.server_kind,
        "vl_rec_server_url": s.server_url,
        "merge_layout_blocks": s.merge_layout_blocks,
        "layout_merge_bboxes_mode": s.layout_merge_bboxes_mode,
    }
    if s.device:
        kwargs["device"] = s.device
    kwargs.update(overrides)
    (res,) = list(PaddleOCRVL(**kwargs).predict(image))[:1]
    data = res.json.get("res", res.json)
    return data.get("parsing_res_list") or []


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare prétraitement et mise en page.")
    ap.add_argument("document", type=Path)
    ap.add_argument("--all", action="store_true", help="Affiche tous les blocs, pas le résumé.")
    args = ap.parse_args()

    cfg = from_env()
    page = load_pages(args.document)[0]

    for name, pre_opts, overrides in VARIANTS:
        print("\n" + "=" * 110)
        print(f"### {name}   prétraitement={pre_opts or 'défaut'}   mise en page={overrides or 'config'}")
        print("=" * 110)
        try:
            image, steps = preprocess(page, **pre_opts)
            print(f"prétraitement : {steps}")
            blocks = run(image, cfg, overrides)
        except Exception as exc:
            print(f"ÉCHEC : {type(exc).__name__}: {exc}")
            continue

        tables = [b for b in blocks if b.get("block_label") == "table"]
        print(f"{len(blocks)} bloc(s), dont {len(tables)} table(s)\n")

        # Le tableau de bornes est le juge de paix : on l'affiche en entier.
        for t in tables:
            bbox = t.get("block_bbox")
            box = "[" + ", ".join(f"{v:.0f}" for v in bbox) + "]" if bbox else ""
            print(f"  TABLE {box}")
            print(f"    {flatten(t.get('block_content'), 400)}\n")

        if args.all:
            for i, b in enumerate(blocks):
                print(f"{i:>2} {b.get('block_label'):<16} {flatten(b.get('block_content'), 80)}")
        else:
            labels = [b.get("block_label") for b in blocks]
            print("  labels :", ", ".join(labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
