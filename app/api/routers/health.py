"""Module 0 — l'API est-elle en état de travailler ?"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import depot, settings

router = APIRouter(tags=["santé"])


@router.get("/health")
def sante() -> dict:
    """Vivant, et avec combien de parcelles en base."""
    return {"statut": "ok", "parcelles": len(depot().all())}


@router.get("/health/lecteurs")
def lecteurs() -> dict:
    """Les deux serveurs de modèles répondent-ils ? Sans eux, aucune lecture n'est possible."""
    import httpx

    s = settings()
    etats = {}
    for nom, url in (("paddle_vl", s.paddle.server_url), ("qwen_vl", s.qwen.server_url)):
        try:
            with httpx.Client(timeout=2.0) as c:
                etats[nom] = {"url": url, "joignable": c.get(f"{url}/models").status_code < 500}
        except Exception as exc:                        # noqa: BLE001
            etats[nom] = {"url": url, "joignable": False, "erreur": type(exc).__name__}
    return {"lecteurs": etats, "les_deux": all(e["joignable"] for e in etats.values())}
