"""Évalue le modèle sur tous les titres annotés.

Usage : python -m eval.run_baseline data/raw data/annotations [reports]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from eval.metrics import (
    DocReport, Outcome, calibration_table, compare, summarize, threshold_for,
)
from model.annotation import load_all
from model.extract import extract


def main(raw: Path, ann_dir: Path, out: Path) -> None:
    reports: list[DocReport] = []
    for ann in load_all(ann_dir):
        img = raw / ann.image
        if not img.is_file():
            print(f"{ann.document_id}: image introuvable ({img.name}), ignoré")
            continue
        rep = compare(extract(img, ann.document_id), ann)
        reports.append(rep)
        n = len(rep.results)
        c = {o: sum(1 for r in rep.results if r.outcome is o) for o in Outcome}
        print(f"{ann.document_id}: exact {c[Outcome.EXACT]}/{n}  near {c[Outcome.NEAR]}  "
              f"wrong {c[Outcome.WRONG]}  missing {c[Outcome.MISSING]}  en trop {rep.spurious_points}")
    if not reports:
        sys.exit("Aucun document évalué.")

    s = summarize(reports)
    results = [r for rep in reports for r in rep.results]
    print(f"\nGlobal : {s['exact']}/{s['values']} exacts ({s['exact_rate']:.0%}), "
          f"near {s['near']}, wrong {s['wrong']}, missing {s['missing']}, points en trop {s['spurious_points']}")
    print("\nCalibration (taux d'erreur par tranche de confiance) :")
    for row in calibration_table(results):
        rate = "—" if row["error_rate"] is None else f"{row['error_rate']:.0%}"
        print(f"  {row['from']:.2f}–{row['to']:.2f}  n={row['n']:3d}  erreurs={row['errors']:3d}  taux={rate}")
    t = threshold_for(results, 0.01)
    print(f"\nSeuil pour <=1 % d'erreurs parmi les valeurs acceptées : {t}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline.json").write_text(json.dumps(
        {"summary": s, "calibration": calibration_table(results), "threshold": t,
         "results": [{**r.__dict__, "outcome": r.outcome.value} for r in results]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRapport écrit dans {out / 'baseline.json'}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3] if len(sys.argv) == 4 else "reports"))
