"""Confronte des parcelles entre elles : recouvrement, et superposition procustéenne.

    python tools/confronter.py data/annotations --all
    python tools/confronter.py A.json B.json --procruste
    python tools/confronter.py data/annotations --all --png reports/confrontations

Le système ne déclare jamais une fraude. Il constate un recouvrement et le qualifie ;
dire si c'est une fraude, une erreur de levé ou un morcellement relève de l'humain.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.geospatial.overlap import NonComparable, compare  # noqa: E402
from app.geospatial.parcel import GeometryRefusee, Parcel, parcel  # noqa: E402
from app.geospatial.procrustes import RecalageImpossible, concordent, superpose  # noqa: E402
from tools.parcelle import charge  # noqa: E402

TOLERANCE_PCT = 1.0


def charge_parcelles(fichiers: list[Path]) -> list[Parcel]:
    out = []
    for f in fichiers:
        try:
            out.append(parcel(charge(f)))
        except GeometryRefusee as exc:
            print(f"  {f.stem:<34} ecarte : {exc}")
        except Exception as exc:                        # noqa: BLE001
            print(f"  {f.stem:<34} illisible : {type(exc).__name__}: {exc}")
    return out


def sommets(p: Parcel) -> list[tuple[float, float]]:
    """Les sommets du polygone, sans la repetition du premier a la fin."""
    return list(p.polygon.exterior.coords)[:-1]


def recouvrements(parcelles: list[Parcel]) -> int:
    conflits = 0
    for a, b in itertools.combinations(parcelles, 2):
        nom = f"{a.title_number or a.document_id} / {b.title_number or b.document_id}"
        try:
            r = compare(a, b)
        except NonComparable as exc:
            print(f"  {nom:<44} non confrontable : {exc}")
            continue
        if r is None:
            print(f"  {nom:<44} disjointes")
            continue
        conflits += r.conflit
        print(f"  {nom:<44} {'CONFLIT  ' if r.conflit else '         '}{r.message}")
    return conflits


def procruste(a: Parcel, b: Parcel) -> None:
    """Les deux releves decrivent-ils le meme terrain, a un deplacement pres ?"""
    try:
        r = superpose(sommets(a), sommets(b))
    except RecalageImpossible as exc:
        print(f"\n  superposition impossible : {exc}")
        return
    print(f"\n  superposition de {b.document_id} sur {a.document_id} :")
    print(f"    echelle      {r.echelle:.6f}")
    print(f"    rotation     {r.rotation_deg:+.3f}°")
    print(f"    residu       {r.residu:.4f} m  ({r.residu_pct:.3f} % de la taille)")
    print(f"    sommets      {r.n}")
    if r.reflexion:
        print(f"    {r.message}")
    elif concordent(r, TOLERANCE_PCT):
        print(f"    => MEME TERRAIN a {TOLERANCE_PCT} % pres, deplace et pivote.")
    else:
        print("    => terrains distincts : la superposition laisse trop de residu.")


def dessine(parcelles: list[Parcel], dossier: Path) -> None:
    """Toutes les parcelles sur un meme plan, avec les recouvrements hachures."""
    try:
        import matplotlib
    except ImportError:
        print("\n  matplotlib n'est pas installe : python -m pip install matplotlib")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dossier.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    for p in parcelles:
        xs, ys = zip(*list(p.polygon.exterior.coords), strict=False)
        ax.plot(xs, ys, "-", linewidth=1.4)
        ax.fill(xs, ys, alpha=0.12)
        cx, cy = p.polygon.centroid.x, p.polygon.centroid.y
        ax.annotate(p.title_number or p.document_id, (cx, cy),
                    ha="center", fontsize=8)
    for a, b in itertools.combinations(parcelles, 2):
        try:
            commun = a.polygon.intersection(b.polygon)
        except Exception:                               # noqa: BLE001
            continue
        if not commun.is_empty and commun.area > 0:
            xs, ys = zip(*list(commun.exterior.coords), strict=False)
            ax.fill(xs, ys, color="#c0392b", alpha=0.45, hatch="//", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("Est (m)")
    ax.set_ylabel("Nord (m)")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.ticklabel_format(style="plain", useOffset=False)
    fig.tight_layout()
    chemin = dossier / "confrontation.png"
    fig.savefig(chemin, dpi=150)
    plt.close(fig)
    print(f"\n  {chemin}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sources", type=Path, nargs="+")
    ap.add_argument("--all", action="store_true", help="Tous les JSON du dossier.")
    ap.add_argument("--procruste", action="store_true",
                    help="Superpose les deux parcelles (exactement deux).")
    ap.add_argument("--png", type=Path, nargs="?", const=Path("reports/confrontations"),
                    metavar="DOSSIER", help="Dessine les parcelles et leurs recouvrements.")
    args = ap.parse_args()

    fichiers: list[Path] = []
    for s in args.sources:
        fichiers += (sorted(f for f in s.glob("*.json") if not f.stem.endswith("_readings"))
                     if s.is_dir() else [s])
    parcelles = charge_parcelles(fichiers)
    if len(parcelles) < 2:
        print(f"{len(parcelles)} parcelle(s) exploitable(s) : il en faut au moins deux.")
        return 1

    print(f"\n{len(parcelles)} parcelles, {len(parcelles) * (len(parcelles) - 1) // 2} paire(s) :")
    conflits = recouvrements(parcelles)
    print(f"\n  {conflits} conflit(s) de recouvrement")

    if args.procruste:
        if len(parcelles) != 2:
            print("\n  --procruste demande exactement deux parcelles.")
        else:
            procruste(*parcelles)
    if args.png:
        dessine(parcelles, args.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
