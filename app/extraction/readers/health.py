"""Vérifie que les deux serveurs llama.cpp répondent, avant de lancer une extraction.

Usage : python -m model.check_servers
"""

from __future__ import annotations

import sys

import httpx

from app.core.config import Settings, from_env


def check(name: str, url: str, client: httpx.Client) -> bool:
    """Interroge /health de llama-server (l'URL configurée finit par /v1)."""
    base = url.rstrip("/").removesuffix("/v1")
    try:
        r = client.get(f"{base}/health", timeout=5.0)
    except httpx.HTTPError as exc:
        print(f"✗ {name} injoignable sur {base} ({type(exc).__name__}). Le serveur est-il lancé ?")
        return False
    if r.status_code == 200:
        print(f"✓ {name} prêt sur {base}")
        return True
    if r.status_code == 503:
        print(f"… {name} démarre encore sur {base} (modèle en cours de chargement)")
    else:
        print(f"✗ {name} répond {r.status_code} sur {base}")
    return False


def main(cfg: Settings | None = None, client: httpx.Client | None = None) -> int:
    cfg = cfg or from_env()
    client = client or httpx.Client()
    ok = check("PaddleOCR-VL", cfg.paddle.server_url, client)
    ok &= check("Qwen3-VL", cfg.qwen.server_url, client)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
