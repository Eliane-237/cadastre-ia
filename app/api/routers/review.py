"""Module 5 — ce que l'humain a tranché, et qui doit rester.

En écriture seule : on consigne, on ne reprend pas. Une décision réécrite après coup ne
vaut rien devant un requérant qui demande pourquoi son titre a été bloqué.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from app.api.deps import historique
from app.api.schemas import DecisionIn
from app.review.history import Action, Decision

router = APIRouter(prefix="/revue", tags=["revue"])


@router.post("/{document_id:path}", status_code=201, summary="Consigner une décision humaine")
def consigner(document_id: str, corps: DecisionIn) -> dict:
    try:
        action = Action(corps.action)
    except ValueError as exc:
        raise HTTPException(
            422, f"action « {corps.action} » inconnue ; attendu : "
                 + ", ".join(a.value for a in Action)) from exc
    if not corps.auteur.strip():
        raise HTTPException(422, "une décision sans auteur ne vaut rien : renseignez `auteur`")
    decision = historique().consigne(
        Decision(document_id, action, corps.auteur.strip(), corps.motif, corps.detail)
    )
    return asdict(decision) | {"action": decision.action.value}


@router.get("/{document_id:path}", summary="L'historique d'un document")
def journal(document_id: str) -> dict:
    decisions = historique().pour(document_id)
    return {
        "document_id": document_id,
        "total": len(decisions),
        "decisions": [asdict(d) | {"action": d.action.value} for d in decisions],
    }
