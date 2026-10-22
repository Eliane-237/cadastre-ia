"""Une parcelle vers ce qui s'affiche : GeoJSON en WGS84.

Ce code vivait dans `tools/parcelle.py`, donc l'API ne pouvait pas s'en servir. Le
dupliquer aurait voulu dire deux reprojections qui dérivent l'une de l'autre, et un jour
deux cartes qui ne montrent pas la même parcelle.

La réserve qui accompagne chaque sortie n'est pas une formalité : la norme GeoJSON impose
le WGS84, et on y va sans transformation de datum. L'écart RRS04/WGS84 est de l'ordre du
mètre — sur une parcelle aux côtés de quinze mètres, cela fait 7 %. Bon pour regarder,
pas pour mesurer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.geospatial.parcel import Parcel

WGS84 = "EPSG:4326"

SANS_DATUM = ("Reprojection UTM 28N → WGS84 sans transformation de datum : l'écart "
              "RRS04/WGS84 est de l'ordre du mètre, suffisant pour regarder, pas pour mesurer.")


def _reprojecteur(crs: str) -> tuple[Callable[[float, float], tuple[float, float]] | None, str]:
    """La fonction de reprojection vers WGS84, et ce qu'il faut en dire. None sans pyproj."""
    try:
        from pyproj import Transformer
    except ImportError:
        return None, (f"Coordonnées laissées en {crs} : pyproj n'est pas installé. "
                      "QGIS les lira ; une carte web attend du WGS84.")
    return Transformer.from_crs(crs, WGS84, always_xy=True).transform, SANS_DATUM


def _proprietes(p: Parcel) -> dict[str, Any]:
    return {
        "document": p.document_id,
        "titre": p.title_number,
        "titre_mere": p.mother_title,
        "aire_m2": round(p.area_m2, 2),
        "aire_imprimee_m2": p.printed_m2,
        # La qualité voyage avec la géométrie : une carte qui montre un polygone sans
        # dire ce qu'il vaut laisse croire qu'il vaut le pixel. C'est ce que le Kadaster
        # publie sur son Informatiekaart, et ce que QGIS lira dans la table attributaire.
        "precision_m": round(p.precision_m, 3) if p.precision_m is not None else None,
        "qualite": p.qualite,
        "datum": p.datum,
        "comparable": p.comparable,
    }


def to_geojson(parcelles: list[Parcel]) -> dict[str, Any]:
    """Les parcelles en FeatureCollection, reprojetées quand c'est possible.

    La note est dans la sortie, pas dans un commentaire de code : qui ouvre le fichier
    dans six mois doit savoir ce qu'il regarde.
    """
    if not parcelles:
        return {"type": "FeatureCollection", "features": [], "note": "aucune parcelle"}

    projete, note = _reprojecteur(parcelles[0].crs)
    traits = [
        {
            "type": "Feature",
            "properties": _proprietes(p),
            "geometry": {
                "type": "Polygon",
                "coordinates": [[list(projete(x, y) if projete else (x, y))
                                 for x, y in p.polygon.exterior.coords]],
            },
        }
        for p in parcelles
    ]
    sortie: dict[str, Any] = {"type": "FeatureCollection", "note": note, "features": traits}
    if projete is None:
        # Lu par QGIS ; la norme GeoJSON ne connaît que le WGS84, on le signale donc.
        sortie["crs"] = {"type": "name", "properties": {"name": parcelles[0].crs}}
    return sortie


def centre_wgs84(p: Parcel) -> tuple[float, float] | None:
    """Longitude et latitude du centroïde, pour poser un repère sur une carte."""
    projete, _ = _reprojecteur(p.crs)
    if projete is None:
        return None
    c = p.polygon.centroid
    lon, lat = projete(c.x, c.y)
    return round(lon, 7), round(lat, 7)
