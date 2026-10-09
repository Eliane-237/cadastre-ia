"""Évalue le modèle sur tous les titres annotés : chaque lecteur seul, puis la fusion.

Usage :
    python -m eval.run_baseline data/raw data/annotations [reports]         # modèles (GPU)
    python -m eval.run_baseline data/raw data/annotations reports --replay  # fusion seule

Les lectures brutes des deux modèles sont enregistrées dans reports/readings/ : `--replay`
les relit pour rejouer la fusion et la validation (réglages, seuils) sans GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.metrics import (
    DocReport,
    Outcome,
    calibration_table,
    compare,
    summarize,
    threshold_for,
)
from model.annotation import load_all
from model.config import Settings, from_env
from model.extract import run
from model.fusion import fuse
from model.readers.base import DocumentReading, reading_from_dict, reading_to_dict
from model.schema import Extraction, Source
from model.validate import validate


def fused(
    doc_id: str, paddle: DocumentReading | None, qwen: DocumentReading | None, cfg: Settings
) -> Extraction:
    ex, report = fuse(doc_id, paddle, qwen, cfg)
    ex.issues = [*report.issues, *validate(ex, cfg)]
    return ex


def _line(name: str, rep: DocReport) -> str:
    n = len(rep.results)
    c = {o: sum(1 for r in rep.results if r.outcome is o) for o in Outcome}
    return (
        f"  {name:7s} exact {c[Outcome.EXACT]}/{n}  near {c[Outcome.NEAR]}  "
        f"wrong {c[Outcome.WRONG]}  missing {c[Outcome.MISSING]}  en trop {rep.spurious_points}"
    )


def main(raw: Path, ann_dir: Path, out: Path, replay: bool) -> None:
    cfg = from_env()
    cache = out / "readings"
    cache.mkdir(parents=True, exist_ok=True)
    reports: dict[str, list[DocReport]] = {"paddle": [], "qwen": [], "fusion": []}
    review = 0
    for ann in load_all(ann_dir):
        cached = cache / f"{ann.document_id}.json"
        if replay:
            if not cached.is_file():
                print(f"{ann.document_id}: pas de lectures enregistrées, ignoré")
                continue
            data = json.loads(cached.read_text(encoding="utf-8"))
            readings = {k: reading_from_dict(v) for k, v in data.items()}
        else:
            img = raw / ann.image
            if not img.is_file():
                print(f"{ann.document_id}: image introuvable ({img.name}), ignoré")
                continue
            readings = run(img, ann.document_id, cfg=cfg).readings
            cached.write_text(
                json.dumps(
                    {k: reading_to_dict(v) for k, v in readings.items()},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        p = readings.get(Source.PADDLE_VL.value)
        q = readings.get(Source.QWEN_VL.value)
        both = fused(ann.document_id, p, q, cfg)
        review += both.needs_review()
        print(f"{ann.document_id}:{' (revue)' if both.needs_review() else ''}")
        for name, ex in (
            ("paddle", fused(ann.document_id, p, None, cfg) if p else None),
            ("qwen", fused(ann.document_id, None, q, cfg) if q else None),
            ("fusion", both),
        ):
            if ex is None:
                continue
            rep = compare(ex, ann)
            reports[name].append(rep)
            print(_line(name, rep))
    if not reports["fusion"]:
        sys.exit("Aucun document évalué.")

    print("\nGlobal :")
    summary = {}
    for name, reps in reports.items():
        if reps:
            s = summarize(reps)
            summary[name] = s
            print(
                f"  {name:7s} {s['exact']}/{s['values']} exacts ({s['exact_rate']:.0%}), "
                f"near {s['near']}, wrong {s['wrong']}, missing {s['missing']}"
            )
    print(f"  documents envoyés en revue : {review}/{len(reports['fusion'])}")

    results = [r for rep in reports["fusion"] for r in rep.results]
    calib = calibration_table(results)
    print("\nCalibration de la fusion (taux d'erreur par tranche de confiance) :")
    for row in calib:
        rate = "—" if row["error_rate"] is None else f"{row['error_rate']:.0%}"
        print(
            f"  {row['from']:.2f}–{row['to']:.2f}  n={row['n']:3d}  "
            f"erreurs={row['errors']:3d}  taux={rate}"
        )
    t = threshold_for(results, 0.01)
    print(f"\nSeuil pour <=1 % d'erreurs parmi les valeurs acceptées : {t}")

    (out / "evaluation.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "review": review,
                "calibration": calib,
                "threshold": t,
                "results": [{**r.__dict__, "outcome": r.outcome.value} for r in results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nRapport écrit dans {out / 'evaluation.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Évaluation : chaque lecteur seul, puis la fusion.")
    ap.add_argument("raw", type=Path)
    ap.add_argument("annotations", type=Path)
    ap.add_argument("out", type=Path, nargs="?", default=Path("reports"))
    ap.add_argument(
        "--replay",
        action="store_true",
        help="Rejouer la fusion depuis les lectures enregistrées (sans GPU).",
    )
    a = ap.parse_args()
    main(a.raw, a.annotations, a.out, a.replay)
