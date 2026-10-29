"""Construit le polygone d'un document et le donne à voir.

    python tools/parcelle.py reports/readings/tf.json
    python tools/parcelle.py reports/readings/tf.json --geojson reports/tf.geojson
    python tools/parcelle.py reports/readings --all

Le GeoJSON s'ouvre dans QGIS, ou sur geojson.io si `pyproj` est installé : la norme
GeoJSON impose le WGS84, et sans reprojection les coordonnées restent en UTM 28N.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.schema import (  # noqa: E402
    Extraction,
)
from app.geospatial.export import to_geojson  # noqa: E402
from app.geospatial.parcel import GeometryRefusee, Parcel, parcel  # noqa: E402
from app.geospatial.render import RenduIndisponible, apercu_png  # noqa: E402
from app.geospatial.render import libelles_imprimes as etiquettes  # noqa: E402


def charge(chemin: Path) -> Extraction:
    """Une extraction, telle que les modèles l'ont produite."""
    donnees = json.loads(chemin.read_text(encoding="utf-8"))
    if "document_id" not in donnees:
        # Un fichier de lectures brutes (`--readings`) depose parmi les extractions :
        # il a les lecteurs en cles de premier niveau, pas de document.
        raise ValueError("ce n'est pas une extraction (lectures brutes ?)")
    return Extraction.model_validate(donnees)


def montre(p: Parcel) -> None:
    print(f"\n=== {p.title_number or p.document_id} ===")
    print(f"  projection  {p.crs}")
    print(f"  datum       {p.datum}" + ("" if p.comparable else "   ← incomparable"))
    sommets = list(p.polygon.exterior.coords)[:-1]
    print(f"  sommets     {len(sommets)}, du sud-ouest, dans le sens direct")
    for i, (x, y) in enumerate(sommets, 1):
        print(f"     {i:>2}  {x:>14.3f}  {y:>15.3f}")
    print(f"  aire        {p.area_m2:>14.2f} m²")
    if p.printed_m2:
        ecart = abs(p.area_m2 - p.printed_m2) / p.printed_m2 * 100
        print(f"  imprimée    {p.printed_m2:>14.0f} m²   écart {ecart:.2f} %")
    else:
        print("  imprimée    non lue sur le document")
    x0, y0, x1, y1 = p.polygon.bounds
    print(f"  emprise     {x1 - x0:.1f} m est-ouest, {y1 - y0:.1f} m nord-sud")


def geojson(parcelles: list[Parcel], chemin: Path) -> None:
    """Écrit la FeatureCollection produite par `app.geospatial.export`."""
    sortie = to_geojson(parcelles)
    chemin.write_text(json.dumps(sortie, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{chemin} — {len(sortie['features'])} parcelle(s)")
    print(f"  {sortie['note']}")


def dessine(parcelles: list[Parcel], dossier: Path,
            libelles: dict[str, dict[tuple[float, float], str]] | None = None) -> None:
    """Un PNG par parcelle, dessiné par `app.geospatial.render`."""
    dossier.mkdir(parents=True, exist_ok=True)
    for p in parcelles:
        try:
            octets = apercu_png(p, (libelles or {}).get(p.document_id, {}))
        except RenduIndisponible as exc:
            print(f"  {exc}")
            return
        chemin = dossier / f"{p.document_id.replace('/', '_')}.png"
        chemin.write_bytes(octets)
        print(f"  {chemin}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path,
                    help="Un JSON d'extraction, ou un dossier qui en contient.")
    ap.add_argument("--all", action="store_true", help="Tous les JSON du dossier.")
    ap.add_argument("--geojson", type=Path, help="Écrit les polygones pour QGIS.")
    ap.add_argument("--png", type=Path, nargs="?", const=Path("reports/parcelles"),
                    metavar="DOSSIER",
                    help="Dessine chaque parcelle (défaut : reports/parcelles).")
    args = ap.parse_args()

    fichiers = (sorted(f for f in args.source.glob("*.json")
                       if not f.stem.endswith("_readings"))
                if args.all or args.source.is_dir() else [args.source])
    if not fichiers:
        print(f"Aucun JSON dans {args.source}.")
        return 1

    parcelles: list[Parcel] = []
    libelles: dict[str, dict[tuple[float, float], str]] = {}
    for f in fichiers:
        try:
            extraction = charge(f)
            p = parcel(extraction)
            libelles[p.document_id] = etiquettes(extraction)
        except GeometryRefusee as exc:
            print(f"\n=== {f.stem} ===\n  REFUS : {exc}")
            continue
        except Exception as exc:                      # noqa: BLE001
            print(f"\n=== {f.stem} ===\n  illisible : {type(exc).__name__}: {exc}")
            continue
        montre(p)
        parcelles.append(p)

    if args.png and parcelles:
        print(f"\n{len(parcelles)} tracé(s) :")
        dessine(parcelles, args.png, libelles)
    if args.geojson and parcelles:
        geojson(parcelles, args.geojson)
    return 0 if parcelles else 1


if __name__ == "__main__":
    raise SystemExit(main())
