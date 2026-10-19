"""Mesure l'inclinaison estimée de chaque image du corpus.

Aucune OCR, aucun modèle : on appelle seulement l'estimateur pour savoir
quels documents sont réellement inclinés. Sans un document incliné, tester
`deskew=True` ne mesure rien.

Usage :
    python tools/skew_census.py
    python tools/skew_census.py data/raw
"""

from __future__ import annotations

import sys
from pathlib import Path

from model.preprocess import estimate_skew, load_image

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".jfif", ".bmp", ".tif", ".tiff"}


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    if not root.exists():
        print(f"Dossier introuvable : {root}")
        return 1

    paths = sorted(p for p in root.iterdir() if p.suffix.lower() in EXTS)
    if not paths:
        print(f"Aucune image dans {root}")
        return 1

    width = max(len(p.name) for p in paths)
    print(f"{'document'.ljust(width)}   inclinaison   verdict")
    print("-" * (width + 32))

    tilted: list[tuple[Path, float]] = []
    for path in paths:
        try:
            image = load_image(path)
        except Exception as exc:  # une image illisible ne doit pas arrêter le recensement
            print(f"{path.name.ljust(width)}   {'—'.rjust(11)}   illisible : {exc}")
            continue

        try:
            angle = estimate_skew(image)
        except Exception as exc:
            print(f"{path.name.ljust(width)}   {'—'.rjust(11)}   échec : {exc}")
            continue

        if angle is None:
            print(f"{path.name.ljust(width)}   {'—'.rjust(11)}   pas d'estimation")
            continue

        verdict = "DROIT (sous le seuil)" if abs(angle) < 0.3 else "INCLINÉ"
        if abs(angle) >= 0.3:
            tilted.append((path, angle))
        print(f"{path.name.ljust(width)}   {angle:+8.2f}°     {verdict}")

    print()
    if tilted:
        tilted.sort(key=lambda t: abs(t[1]), reverse=True)
        print("Candidats pour le test de redressement, du plus incliné au moins :")
        for path, angle in tilted:
            print(f"  python tools/layout_sweep.py \"{path.as_posix()}\"   # {angle:+.2f}°")
    else:
        print("Aucun document du corpus n'est incliné au-dessus du seuil.")
        print("Le redressement ne s'activerait sur aucun d'eux : le laisser désactivé")
        print("ne change rien aujourd'hui, et l'activer ne gagnerait rien non plus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
