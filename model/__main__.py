"""Usage : python -m model DOCUMENT [--id D01] [--out resultat.json] [--readings lectures.json]
                           [--only paddle|qwen]

Les serveurs des modèles se règlent par variables d'environnement (voir model/config.py) :
CADASTRE_PADDLE_URL, CADASTRE_QWEN_URL, CADASTRE_*_BACKEND=server|local.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model.config import from_env
from model.extract import run
from model.readers.base import reading_to_dict
from model.schema import Severity

_ICON = {Severity.ERROR: "✗", Severity.WARNING: "!", Severity.INFO: "i"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extraction d'un titre foncier / plan de bornage.")
    ap.add_argument("document", type=Path)
    ap.add_argument("--id", dest="document_id")
    ap.add_argument("--out", type=Path, help="Extraction en JSON (défaut : sortie standard).")
    ap.add_argument("--readings", type=Path, help="Lectures brutes des deux modèles, pour l'audit.")
    ap.add_argument("--only", choices=["paddle", "qwen"], help="Un seul lecteur (diagnostic).")
    args = ap.parse_args(argv)

    result = run(
        args.document,
        args.document_id,
        cfg=from_env(),
        use_paddle=args.only != "qwen",
        use_qwen=args.only != "paddle",
    )
    ex = result.extraction
    payload = ex.model_dump_json(indent=2)
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload)
    if args.readings:
        args.readings.write_text(
            json.dumps(
                {k: reading_to_dict(v) for k, v in result.readings.items()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        f"\n{ex.document_id} : {len(ex.points)} points, "
        f"{'REVUE HUMAINE' if ex.needs_review() else 'aucune revue requise'}",
        file=sys.stderr,
    )
    for issue in ex.issues:
        print(f"  {_ICON[issue.severity]} [{issue.code}] {issue.message}", file=sys.stderr)
    low = ex.low_confidence_fields()
    if low:
        print(f"  champs à relire : {', '.join(low)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
