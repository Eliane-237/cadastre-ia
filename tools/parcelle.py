"""Construit le polygone d'un document et le donne à voir.

    python tools/parcelle.py data/annotations/D10.json
    python tools/parcelle.py reports/readings/tf.json --geojson reports/tf.geojson
    python tools/parcelle.py data/annotations --all

Le GeoJSON s'ouvre dans QGIS, ou sur geojson.io si `pyproj` est installé : la norme
GeoJSON impose le WGS84, et sans reprojection les coordonnées restent en UTM 28N.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.geospatial.parcel import GeometryRefusee, Parcel, parcel                 # noqa: E402
from app.core.schema import (                                                 # noqa: E402
    DATUM_RRS04,
    Area,
    BoundaryPoint,
    Extracted,
    Extraction,
)


def _depuis_annotation(donnees: dict) -> Extraction:
    pts = [
        BoundaryPoint(
            label=p["label"], row_index=i,
            x=Extracted[float](value=float(p["x"]), confidence=1.0, raw_text=p["x"]),
            y=Extracted[float](value=float(p["y"]), confidence=1.0, raw_text=p["y"]),
        )
        for i, p in enumerate(donnees["points"])
    ]
    aire = donnees.get("printed_area")
    return Extraction(
        document_id=donnees["document_id"], points=pts, datum=DATUM_RRS04,
        printed_area=Extracted[Area](value=Area(**aire), confidence=1.0)
        if aire else Extracted[Area](),
        title_number=Extracted[str](value=donnees["title_number"], confidence=1.0)
        if donnees.get("title_number") else Extracted[str](),
    )


def charge(chemin: Path) -> Extraction:
    """Une annotation ou une extraction, indifféremment."""
    donnees = json.loads(chemin.read_text(encoding="utf-8"))
    if "annotation_version" in donnees:
        return _depuis_annotation(donnees)
    if "document_id" not in donnees:
        # Un fichier de lectures brutes (`--readings`) depose parmi les extractions :
        # il a les lecteurs en cles de premier niveau, pas de document.
        raise ValueError("ni annotation ni extraction (lectures brutes ?)")
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
    """Écrit les parcelles, reprojetées en WGS84 si `pyproj` est là."""
    try:
        from pyproj import Transformer

        vers_wgs84 = Transformer.from_crs(parcelles[0].crs, "EPSG:4326", always_xy=True)
        projete = vers_wgs84.transform
        note = ("Reprojection UTM 28N → WGS84 sans transformation de datum : "
                "l'écart RRS04/WGS84 est de l'ordre du mètre, suffisant pour regarder, "
                "pas pour mesurer.")
    except ImportError:
        projete = None
        note = (f"Coordonnées laissées en {parcelles[0].crs} : pyproj n'est pas installé. "
                "QGIS les lira ; geojson.io attend du WGS84.")

    traits = []
    for p in parcelles:
        anneau = [(projete(x, y) if projete else (x, y))
                  for x, y in p.polygon.exterior.coords]
        traits.append({
            "type": "Feature",
            "properties": {
                "document": p.document_id,
                "titre": p.title_number,
                "titre_mere": p.mother_title,
                "aire_m2": round(p.area_m2, 2),
                "aire_imprimee_m2": p.printed_m2,
                "datum": p.datum,
            },
            "geometry": {"type": "Polygon", "coordinates": [[list(c) for c in anneau]]},
        })

    sortie = {"type": "FeatureCollection", "note": note, "features": traits}
    if projete is None:
        sortie["crs"] = {"type": "name",
                         "properties": {"name": parcelles[0].crs}}   # lu par QGIS
    chemin.write_text(json.dumps(sortie, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{chemin} — {len(traits)} parcelle(s)")
    print(f"  {note}")


def etiquettes(extraction: Extraction) -> dict[tuple[float, float], str]:
    """Position d'un sommet -> son libelle imprime (B1, B2...).

    `parcel()` reordonne les sommets pour que deux parcelles soient comparables : le
    premier point du polygone n'est pas le premier du tableau. Pour comparer le trace au
    plan du document, il faut les libelles du papier, pas l'ordre canonique.
    """
    return {(p.x.value, p.y.value): p.label
            for p in extraction.points if p.complete}


def dessine(parcelles: list[Parcel], dossier: Path,
            libelles: dict[str, dict[tuple[float, float], str]] | None = None) -> None:
    """Un PNG par parcelle : la forme, ses sommets numerotes, son echelle.

    C'est le controle que l'oeil fait mieux que n'importe quel seuil : si la forme tracee
    ne ressemble pas au plan imprime sur le document, les coordonnees sont fausses, meme
    quand l'aire tombe juste. Le nord est en haut, l'est a droite, et les proportions sont
    respectees — un polygone etire par l'affichage ne se compare a rien.
    """
    try:
        import matplotlib
    except ImportError:
        print("  matplotlib n'est pas installé dans cet environnement.\n"
              "  Installe-le : python -m pip install matplotlib\n"
              "  Le reste de l'outil fonctionne sans lui (--geojson, affichage texte).")
        return
    matplotlib.use("Agg")                     # aucun affichage : on ecrit des fichiers
    import matplotlib.pyplot as plt

    dossier.mkdir(parents=True, exist_ok=True)
    for p in parcelles:
        sommets = list(p.polygon.exterior.coords)
        xs, ys = zip(*sommets)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(xs, ys, "-", linewidth=1.6, color="#1a4d7a")
        ax.fill(xs, ys, alpha=0.10, color="#1a4d7a")
        ax.plot(xs[:-1], ys[:-1], "o", markersize=4, color="#1a4d7a")
        noms = (libelles or {}).get(p.document_id, {})
        for i, (x, y) in enumerate(sommets[:-1], 1):
            ax.annotate(noms.get((x, y), str(i)), (x, y),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

        ecart = ""
        if p.printed_m2:
            ecart = f" — imprimee {p.printed_m2:.0f} m² (ecart {abs(p.area_m2 - p.printed_m2) / p.printed_m2 * 100:.2f} %)"
        ax.set_title(f"{p.title_number or p.document_id}\n{p.area_m2:.0f} m²{ecart}", fontsize=10)
        ax.set_xlabel("Est (m)")
        ax.set_ylabel("Nord (m)")
        ax.set_aspect("equal")                # sans cela, la forme est deformee
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.ticklabel_format(style="plain", useOffset=False)
        fig.tight_layout()
        chemin = dossier / f"{p.document_id.replace('/', '_')}.png"
        fig.savefig(chemin, dpi=150)
        plt.close(fig)
        print(f"  {chemin}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="Un JSON d'annotation ou d'extraction, ou un dossier.")
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
