"""Générateur de plans de bornage synthétiques avec leur vérité terrain.

Sert à tester la chaîne de bout en bout et à mesurer la robustesse (rotation, flou,
bruit, compression) sans exposer de données réelles, qui ne sont jamais versionnées.
Les documents générés sont fictifs ; leur mise en page reste à rapprocher des modèles
réels utilisés par le cadastre.

Usage : python -m eval.synth --out data/synth --n 20 --seed 0 [--hard]
puis   python -m eval.run_baseline data/synth/raw data/synth/annotations reports/synth
"""

from __future__ import annotations

import argparse
import io
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from model.annotation import AnnotatedPoint, Annotation, save_annotation
from model.schema import Area
from model.validate import shoelace_area

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "DejaVuSans.ttf",
    "arial.ttf",
]
OWNERS = ["Mamadou DIOP", "Awa NDIAYE", "Ousmane FALL", "Fatou SARR", "Cheikh SECK", "Aïssatou BA"]
COMMUNES = ["Sangalkam", "Diamniadio", "Bambilor", "Rufisque Est", "Yène", "Keur Massar"]
REGIONS = ["DK", "DG", "R", "TH", "MB"]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def fr_number(value: float, decimals: int = 2) -> str:
    """245100.5 → '245 100,50'."""
    s = f"{value:,.{decimals}f}"
    return s.replace(",", " ").replace(".", ",")


def area_ha_a_ca(m2: float) -> str:
    ca_total = round(m2)
    ha, rest = divmod(ca_total, 10_000)
    a, ca = divmod(rest, 100)
    parts = ([f"{ha} ha"] if ha else []) + [f"{a:02d} a", f"{ca:02d} ca"]
    return " ".join(parts)


@dataclass
class SynthDoc:
    image: Image.Image
    truth: Annotation


def random_parcel(rng: random.Random) -> list[tuple[float, float]]:
    """Polygone convexe de 4 à 7 sommets, quelque part dans la région de Dakar/Thiès."""
    cx = rng.uniform(230_000, 300_000)
    cy = rng.uniform(1_620_000, 1_660_000)
    n = rng.randint(4, 7)
    radius = rng.uniform(10, 60)
    # Angles strictement croissants sur moins d'un tour : polygone simple (étoilé),
    # sans côtés minuscules.
    angles = [2 * math.pi * (i + rng.uniform(-0.3, 0.3)) / n for i in range(n)]
    pts = [
        (
            round(cx + radius * rng.uniform(0.7, 1.0) * math.cos(a), 2),
            round(cy + radius * rng.uniform(0.7, 1.0) * math.sin(a), 2),
        )
        for a in angles
    ]
    # Sens horaire, comme souvent sur les plans (aire signée négative).
    pairs = zip(pts, pts[1:] + pts[:1], strict=True)
    signed = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in pairs)
    if signed > 0:
        pts.reverse()
    return pts


def render(rng: random.Random, width: int = 1654, height: int = 2339) -> SynthDoc:
    pts = random_parcel(rng)
    labels = [f"B{i + 1}" for i in range(len(pts))]
    sides = [round(math.dist(pts[i], pts[(i + 1) % len(pts)]), 2) for i in range(len(pts))]
    area = round(abs(shoelace_area(pts)))
    tf = f"{rng.randint(100, 25_000)}/{rng.choice(REGIONS)}"
    nicad = "".join(str(rng.randint(0, 9)) for _ in range(14))
    owner = rng.choice(OWNERS)
    commune = rng.choice(COMMUNES)
    area_txt = area_ha_a_ca(area) if rng.random() < 0.6 else f"{fr_number(area, 0)} m²"

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    big, mid, small = _font(46), _font(30), _font(28)
    m = 120
    y = 110
    draw.text((width / 2, y), "RÉPUBLIQUE DU SÉNÉGAL", font=mid, fill="black", anchor="mm")
    y += 70
    draw.text((width / 2, y), "PLAN DE BORNAGE", font=big, fill="black", anchor="mm")
    y += 110
    for line in (
        f"Titre Foncier : TF {tf}",
        f"NICAD : {nicad}",
        f"Propriétaire : {owner}",
        f"Commune de {commune}",
    ):
        draw.text((m, y), line, font=mid, fill="black")
        y += 55

    # Tableau des bornes.
    y += 40
    cols = [m, m + 180, m + 560, m + 980, width - m]
    row_h = 58
    headers = ["Borne", "X (m)", "Y (m)", "Distance (m)"]
    n_rows = len(pts) + 1
    for i in range(n_rows + 1):
        draw.line([(cols[0], y + i * row_h), (cols[-1], y + i * row_h)], fill="black", width=2)
    for x in cols:
        draw.line([(x, y), (x, y + n_rows * row_h)], fill="black", width=2)
    for j, h in enumerate(headers):
        draw.text((cols[j] + 15, y + 14), h, font=small, fill="black")
    for i, ((px, py), lab, d) in enumerate(zip(pts, labels, sides, strict=True)):
        ry = y + (i + 1) * row_h + 14
        for j, cell in enumerate((lab, fr_number(px), fr_number(py), fr_number(d))):
            draw.text((cols[j] + 15, ry), cell, font=small, fill="black")
    y += n_rows * row_h + 60
    draw.text((m, y), f"Superficie : {area_txt}", font=mid, fill="black")

    # Croquis de la parcelle.
    y += 90
    box = (m + 200, y, width - m - 200, height - 200)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    scale = min((box[2] - box[0]) / (max(xs) - min(xs)), (box[3] - box[1]) / (max(ys) - min(ys)))
    sketch = [(box[0] + (px - min(xs)) * scale, box[3] - (py - min(ys)) * scale) for px, py in pts]
    draw.polygon(sketch, outline="black", width=3)
    for (sx, sy), lab in zip(sketch, labels, strict=True):
        draw.ellipse((sx - 6, sy - 6, sx + 6, sy + 6), fill="black")
        draw.text((sx + 10, sy - 30), lab, font=small, fill="black")

    ha, rest = divmod(area, 10_000)
    truth = Annotation(
        document_id="",
        image="",
        title_number=f"TF {tf}",
        nicad=nicad,
        locality=commune,
        printed_area=Area(ha=ha, a=rest // 100, ca=rest % 100),
        points=[
            AnnotatedPoint(label=lab, x=f"{x:.2f}", y=f"{y:.2f}")
            for lab, (x, y) in zip(labels, pts, strict=True)
        ],
        decimals=2,
        annotator="synthetique",
        verified=True,  # vérité connue par construction
        notes=f"propriétaire : {owner} ; document fictif",
    )
    return SynthDoc(image=img, truth=truth)


def degrade(
    img: Image.Image,
    rng: random.Random,
    *,
    max_rotation: float = 3.0,
    blur: float = 0.8,
    noise: float = 8.0,
    jpeg_quality: int = 60,
) -> Image.Image:
    """Simule un scan ou une photo médiocre."""
    out = img.rotate(rng.uniform(-max_rotation, max_rotation), expand=True, fillcolor="white")
    if blur:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0, blur)))
    if noise:
        arr = np.asarray(out).astype(np.float32)
        arr += np.random.default_rng(rng.randint(0, 2**31)).normal(0, noise, arr.shape)
        out = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if jpeg_quality:
        buf = io.BytesIO()
        out.save(buf, "JPEG", quality=jpeg_quality)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
    return out


HARD = {"max_rotation": 8.0, "blur": 1.5, "noise": 15.0, "jpeg_quality": 35}


def generate(
    out_dir: Path, n: int, seed: int = 0, degraded: bool = True, hard: bool = False
) -> None:
    images, annotations = out_dir / "raw", out_dir / "annotations"
    images.mkdir(parents=True, exist_ok=True)
    annotations.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for i in range(n):
        doc = render(rng)
        img = degrade(doc.image, rng, **(HARD if hard else {})) if degraded else doc.image
        name = f"synth_{i:04d}"
        img.save(images / f"{name}.png")
        save_annotation(
            doc.truth.model_copy(update={"document_id": name, "image": f"{name}.png"}), annotations
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/synth"))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean", action="store_true", help="Sans dégradation.")
    parser.add_argument("--hard", action="store_true", help="Dégradation forte (photo médiocre).")
    args = parser.parse_args()
    generate(args.out, args.n, args.seed, degraded=not args.clean, hard=args.hard)
    print(f"{args.n} documents écrits dans {args.out}")


if __name__ == "__main__":
    main()
