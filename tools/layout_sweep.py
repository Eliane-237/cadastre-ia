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

from app.core.config import from_env                      # noqa: E402
from model.paddle_reader import reading_from_result    # noqa: E402
from model.preprocess import load_pages, preprocess    # noqa: E402

_TAGS = re.compile(r"<[^>]+>")

# (nom, options de prétraitement, surcharges de mise en page).
#
# Pourquoi ces variantes-là. Sur plusieurs documents, la mise en page ne rend qu'un seul
# bloc — un « seal » sur 54987580 — alors que la page entière est lisible. Le suspect
# principal est `layout_merge_bboxes_mode="small"` : quand deux boîtes se recouvrent, ce
# mode garde la petite. Or le cadre du plan recouvre à peu près tout, et le cachet est la
# plus petite boîte de la feuille. On balaye donc les modes de fusion avant de toucher au
# reste. Le dernier essai sonde un seuil de détection : si le pipeline ne l'accepte pas,
# la variante échoue proprement et on le saura.
VARIANTS: list[tuple[str, dict, dict]] = [
    ("référence (config actuelle)", {}, {}),
    ("fusion = large", {}, {"layout_merge_bboxes_mode": "large"}),
    ("fusion = union", {}, {"layout_merge_bboxes_mode": "union"}),
    ("merge_layout_blocks = True", {}, {"merge_layout_blocks": True}),
    ("dépliage UVDoc", {}, {"use_doc_unwarping": True}),
    ("deskew = True", {"deskew": True}, {}),
    ("seuil de détection 0.3", {}, {"layout_threshold": 0.3}),
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
    return res.json.get("res", res.json)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare prétraitement et mise en page.")
    ap.add_argument("document", type=Path)
    ap.add_argument("--all", action="store_true", help="Affiche tous les blocs, pas le résumé.")
    args = ap.parse_args()

    cfg = from_env()
    page = load_pages(args.document)[0]
    recap: list[tuple[str, str]] = []

    for name, pre_opts, overrides in VARIANTS:
        print("\n" + "=" * 110)
        print(f"### {name}   prétraitement={pre_opts or 'défaut'}   mise en page={overrides or 'config'}")
        print("=" * 110)
        try:
            image, steps = preprocess(page, **pre_opts)
            print(f"prétraitement : {steps}")
            data = run(image, cfg, overrides)
            blocks = data.get("parsing_res_list") or []
        except Exception as exc:
            print(f"ÉCHEC : {type(exc).__name__}: {exc}")
            recap.append((name, f"ÉCHEC {type(exc).__name__}"))
            continue

        tables = [b for b in blocks if b.get("block_label") == "table"]
        # Le vrai juge : ce que le lecteur tire de cette mise en page.
        lecture = reading_from_result(data)
        champs = sum(1 for v in lecture.fields.values() if v.raw)
        complets = sum(1 for p in lecture.points if p.x and p.y)
        print(f"{len(blocks)} bloc(s), dont {len(tables)} table(s) — "
              f"{champs} champ(s) lu(s), {complets} sommet(s) complet(s)\n")
        recap.append((name, f"{len(blocks):>3} blocs  {len(tables):>2} tables  "
                            f"{champs:>2} champs  {complets:>2} sommets"))

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

    print("\n" + "=" * 110)
    print(f"RÉCAPITULATIF — {args.document.name}")
    print("=" * 110)
    for name, resultat in recap:
        print(f"  {name:<32} {resultat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
