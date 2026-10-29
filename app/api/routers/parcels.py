"""Modules 4 et 5 — la base de parcelles, et la confrontation.

L'ordre des deux premières routes dit la règle : on confronte AVANT d'enregistrer.
Enregistrer une parcelle qui chevauche une autre sans l'avoir vu, c'est fabriquer le
conflit qu'on prétend détecter.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from app.api.deps import depot
from app.api.schemas import (
    EnregistrementIn,
    JugementOut,
    ParcelleOut,
    RecouvrementOut,
)
from app.geospatial.export import centre_wgs84, to_geojson
from app.geospatial.overlap import NonComparable, scan
from app.geospatial.parcel import GeometryRefusee, parcel
from app.geospatial.render import RenduIndisponible, apercu_png, libelles_imprimes
from app.storage.repository import NonEnregistrable
from app.validation.verdict import juger

router = APIRouter(prefix="/parcelles", tags=["parcelle"])




def _parcelle(extraction):
    try:
        return parcel(extraction)
    except GeometryRefusee as exc:
        raise HTTPException(422, str(exc)) from exc


def _png(parcelle, libelles) -> Response:
    try:
        return Response(apercu_png(parcelle, libelles), media_type="image/png")
    except RenduIndisponible as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/apercu", summary="Tracer une extraction, sans la lire ni l'enregistrer",
             response_class=Response,
             responses={200: {"content": {"image/png": {}}, "description": "Le tracé"}})
def apercu(corps: EnregistrementIn) -> Response:
    """Le polygone dessiné, en PNG. Aucun modèle n'est appelé : c'est du calcul pur.

    C'est ce qu'un agent regarde avant de décider — le tracé à côté du plan imprimé.
    Séparé de `/documents` pour qu'un changement d'avis ne coûte pas une seconde lecture.
    """
    return _png(_parcelle(corps.extraction), libelles_imprimes(corps.extraction))


@router.get("/geojson", summary="Toute la base en GeoJSON, pour une carte")
def carte() -> dict:
    """La couche à poser sur une carte web, ou à ouvrir dans QGIS."""
    return to_geojson(depot().all())


@router.post("/confrontation", summary="Confronter une extraction à la base, sans enregistrer")
def confronter(corps: EnregistrementIn) -> dict:
    """Ce que cette parcelle recouvrirait si on l'enregistrait. Aucune écriture."""
    p = _parcelle(corps.extraction)
    voisines = depot().near(p.polygon.bounds)
    try:
        trouves = scan(p, voisines)
    except NonComparable as exc:
        raise HTTPException(409, str(exc)) from exc
    return {
        "parcelle": ParcelleOut.de(p),
        "centre": centre_wgs84(p),
        "recouvrements": [RecouvrementOut.de(r) for r in trouves],
        "conflits": sum(1 for r in trouves if r.conflit),
        # La nouvelle parcelle et ses voisines d'enveloppe sur la même couche : c'est en
        # les superposant qu'on voit ce qui se chevauche.
        "geojson": to_geojson([p, *voisines]),
    }


@router.post("", status_code=201, summary="Enregistrer une parcelle exploitable")
def enregistrer(corps: EnregistrementIn) -> dict:
    """La seule porte d'entrée de la base. Un verdict autre qu'« exploitable » la ferme."""
    extraction = corps.extraction
    p = _parcelle(extraction)
    jugement = juger(extraction)
    try:
        enregistree = depot().save(p, extraction, jugement)
    except NonEnregistrable as exc:
        raise HTTPException(409, str(exc)) from exc
    return {
        "document_id": enregistree.document_id,
        "enregistree_le": enregistree.enregistree_le.isoformat(),
        "jugement": JugementOut.de(jugement),
        "parcelle": ParcelleOut.de(p),
    }


@router.get("", summary="Inventaire de la base")
def inventaire() -> dict:
    parcelles = depot().all()
    return {"total": len(parcelles), "parcelles": [ParcelleOut.de(p) for p in parcelles]}


@router.get("/{document_id:path}/recouvrements", summary="Voisines et conflits")
def recouvrements(document_id: str) -> dict:
    enregistree = _trouve(document_id)
    p = enregistree.parcelle
    try:
        trouves = scan(p, depot().near(p.polygon.bounds))
    except NonComparable as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"document_id": document_id,
            "recouvrements": [RecouvrementOut.de(r) for r in trouves],
            "conflits": sum(1 for r in trouves if r.conflit)}


@router.get("/{document_id:path}/apercu", summary="Le tracé d'une parcelle enregistrée",
            response_class=Response,
            responses={200: {"content": {"image/png": {}}, "description": "Le tracé"}})
def apercu_enregistree(document_id: str) -> Response:
    enregistree = _trouve(document_id)
    return _png(enregistree.parcelle, libelles_imprimes(enregistree.extraction))


@router.get("/{document_id:path}/geojson", summary="Une parcelle en GeoJSON")
def geojson_enregistree(document_id: str) -> dict:
    return to_geojson([_trouve(document_id).parcelle])


@router.get("/{document_id:path}", summary="Une parcelle enregistrée")
def lire(document_id: str) -> dict:
    enregistree = _trouve(document_id)
    return {
        "document_id": enregistree.document_id,
        "enregistree_le": enregistree.enregistree_le.isoformat(),
        "jugement": JugementOut.de(enregistree.jugement),
        "parcelle": ParcelleOut.de(enregistree.parcelle),
        "extraction": enregistree.extraction,
    }


@router.delete("/{document_id:path}", status_code=204, summary="Retirer une parcelle")
def retirer(document_id: str) -> None:
    if not depot().delete(document_id):
        raise HTTPException(404, f"{document_id} n'est pas en base")


def _trouve(document_id: str):
    enregistree = depot().get(document_id)
    if enregistree is None:
        raise HTTPException(404, f"{document_id} n'est pas en base")
    return enregistree
