"""Modules 4 et 5 — la base de parcelles, et la confrontation.

L'ordre des deux premières routes dit la règle : on confronte AVANT d'enregistrer.
Enregistrer une parcelle qui chevauche une autre sans l'avoir vu, c'est fabriquer le
conflit qu'on prétend détecter.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from app.api.deps import depot
from app.api.schemas import (
    ConcordanceOut,
    EnregistrementIn,
    JugementOut,
    ParcelleOut,
    RapprochementOut,
    RecouvrementOut,
)
from app.geospatial.export import centre_wgs84, to_geojson
from app.geospatial.overlap import NonComparable, compare, scan
from app.geospatial.parcel import GeometryRefusee, parcel
from app.geospatial.render import RenduIndisponible, apercu_png, libelles_imprimes
from app.storage.repository import NonEnregistrable
from app.validation.parties import Lien, concordance, lien, rapprocher
from app.validation.verdict import juger

router = APIRouter(prefix="/parcelles", tags=["parcelle"])


def _ecritures(sauf: str | None = None) -> list[tuple[str, str | None]]:
    """Les noms de propriétaires déjà en base, avec le document où ils figurent."""
    return [(p.document_id, p.owner) for p in depot().all()
            if p.document_id != sauf and p.owner]


def _concordances(p, voisines) -> list:
    """Les titres liés à celui-ci, et ce que leurs requérants en disent.

    Le lien d'abord, le nom ensuite : on ne confronte deux identités que lorsque le
    NICAD, une filiation déclarée ou le terrain lui-même rattachent déjà les deux
    titres. Un nom qui se ressemble, à lui seul, ne fait pas un dossier.
    """
    recouvrantes = {v.document_id for v in voisines
                    if (r := compare(p, v)) is not None and r.conflit}
    trouves = []
    for autre in depot().all():
        if autre.document_id == p.document_id:
            continue
        nature = lien(p, autre) or (Lien.GEOMETRIE
                                    if autre.document_id in recouvrantes else None)
        if nature is None:
            continue
        if (c := concordance(p, autre, nature)) is not None:
            trouves.append(c)
    # Ce qui demande une instruction d'abord.
    trouves.sort(key=lambda c: not c.a_instruire)
    return trouves


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
    # Deux usages distincts, et l'ordre compte. `concordances` vérifie l'identité sur
    # des titres DÉJÀ liés (NICAD, filiation, même terrain) : c'est là qu'un dossier se
    # forme. `rapprochements` ne fait que suggérer des homonymes dans la base, et peut
    # en rendre plusieurs sans que cela veuille dire quoi que ce soit.
    accords = _concordances(p, voisines)
    proches = rapprocher(corps.extraction.owner.value, _ecritures(p.document_id))
    return {
        "parcelle": ParcelleOut.de(p),
        "centre": centre_wgs84(p),
        "recouvrements": [RecouvrementOut.de(r) for r in trouves],
        "conflits": sum(1 for r in trouves if r.conflit),
        "concordances": [ConcordanceOut.de(c) for c in accords],
        "a_instruire": sum(1 for c in accords if c.a_instruire),
        "rapprochements": [RapprochementOut.de(r) for r in proches],
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


@router.get("/{document_id:path}/homonymes",
            summary="Les titres dont le propriétaire s'écrit presque pareil")
def homonymes(document_id: str) -> dict:
    """Des pistes pour un agent, jamais une identité établie : un même nom désigne
    souvent des personnes différentes."""
    enregistree = _trouve(document_id)
    nom = enregistree.extraction.owner
    proches = rapprocher(nom.value if nom.present else None, _ecritures(document_id))
    return {"document_id": document_id,
            "nom": nom.value if nom.present else None,
            "rapprochements": [RapprochementOut.de(r) for r in proches]}


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
