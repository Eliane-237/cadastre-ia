"""Localise le tableau de bornage par ce qu'il CONTIENT, non par sa forme ni sa position.

Un tableau de bornage se reconnait a sa REGULARITE : plusieurs nombres decimaux de meme
gabarit, groupes au meme endroit. On ne code pas les plages senegalaises en dur — un
gabarit trop strict ne reconnait que les coordonnees deja bien lues, et echoue donc
precisement la ou la localisation serait utile (« 38210.57 » pour « 381210.57 »), ou sur
un document exprime dans un autre referentiel (X a cinq chiffres).

    python tools/table_by_content.py d05_readings.json
    python tools/table_by_content.py reports/readings/
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TAGS = re.compile(r"<[^>]+>")
# Un nombre decimal « long » : 4 a 8 chiffres avant le separateur, 2 ou 3 apres. Exclut
# les distances du plan (« 17.00 », « 46.83 ») et les echelles (« 1 : 500 »).
NUMBER = re.compile(r"(?<![\d.,])(\d{4,8})\s*[.,]\s*(\d{2,3})(?![\d])")

MIN_VALUES = 4        # un tableau de bornage a au moins deux sommets, donc quatre valeurs
MAX_SHAPES = 3        # X et Y, plus une tolerance pour une colonne de distances


@dataclass
class Candidate:
    block_index: int
    label: str | None
    bbox: list[float] | None
    values: int
    shapes: list[tuple[int, int]]
    sample: str

    @property
    def score(self) -> float:
        """Un tableau est d'autant plus credible qu'il porte beaucoup de valeurs de peu
        de gabarits differents."""
        return self.values / max(1, len(self.shapes))


def plain_text(content: object) -> str:
    """Contenu d'un bloc en texte : les balises deviennent des espaces, afin qu'une
    coordonnee isolee dans sa cellule reste reperable."""
    return re.sub(r"\s+", " ", html.unescape(TAGS.sub(" ", str(content or "")))).strip()


def long_numbers(text: str) -> list[tuple[str, tuple[int, int]]]:
    """Nombres decimaux longs du texte, avec leur gabarit (chiffres entiers, decimales)."""
    out = []
    for entier, dec in NUMBER.findall(text):
        out.append((f"{entier}.{dec}", (len(entier), len(dec))))
    return out


def inspect_block(text: str) -> tuple[int, list[tuple[int, int]], list[str]] | None:
    """(nombre de valeurs retenues, gabarits dominants, echantillon) ou None."""
    found = long_numbers(text)
    if len(found) < MIN_VALUES:
        return None

    counts = Counter(shape for _, shape in found)
    # On ne garde que les gabarits representes au moins deux fois : une valeur isolee
    # d'un format unique est du bruit, pas une colonne.
    dominant = [shape for shape, n in counts.most_common(MAX_SHAPES) if n >= 2]
    if not dominant:
        return None

    kept = [v for v, shape in found if shape in dominant]
    if len(kept) < MIN_VALUES:
        return None
    return len(kept), dominant, kept[:6]


def candidates(data: dict) -> list[Candidate]:
    blocks = data.get("parsing_res_list") or []
    found: list[Candidate] = []
    for i, block in enumerate(blocks):
        result = inspect_block(plain_text(block.get("block_content")))
        if result is None:
            continue
        values, shapes, sample = result
        found.append(
            Candidate(
                block_index=i,
                label=block.get("block_label"),
                bbox=block.get("block_bbox"),
                values=values,
                shapes=shapes,
                sample=" ".join(sample),
            )
        )
    found.sort(key=lambda c: c.score, reverse=True)
    return found


def from_readings(path: Path) -> dict[str, list[Candidate]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for reader, reading in data.items():
        raw = (reading or {}).get("raw_output") or {}
        out[reader] = candidates(raw)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Localise le tableau de bornage par son contenu.")
    ap.add_argument("source", type=Path, help="Un *_readings.json, ou un dossier.")
    args = ap.parse_args()

    files = (
        sorted(args.source.glob("*_readings.json"))
        if args.source.is_dir() else [args.source]
    )
    if not files:
        print("Aucun fichier de lectures trouve.")
        return 1

    localises = 0
    for path in files:
        print(f"\n=== {path.name} ===")
        try:
            per_reader = from_readings(path)
        except Exception as exc:
            print(f"  ERREUR {type(exc).__name__}: {exc}")
            continue

        trouve = False
        for reader, cands in per_reader.items():
            if not cands:
                # Qwen rend un JSON structure, pas des blocs : son silence est normal.
                print(f"  {reader:<12} pas de bloc localisable")
                continue
            trouve = True
            best = cands[0]
            box = (
                "[" + ", ".join(f"{v:.0f}" for v in best.bbox) + "]"
                if isinstance(best.bbox, (list, tuple)) else "bbox absente"
            )
            shapes = ", ".join(f"{a}+{b}" for a, b in best.shapes)
            print(f"  {reader:<12} bloc #{best.block_index} ({best.label}) "
                  f"{best.values} valeurs, gabarits {shapes}  {box}")
            print(f"  {'':<12} echantillon : {best.sample}")
        localises += trouve

    print(f"\n{localises}/{len(files)} document(s) avec un tableau localise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
