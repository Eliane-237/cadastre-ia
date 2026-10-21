"""L'API expose ce que le métier décide, et n'en décide rien de plus.

FastAPI est une dépendance du projet, pas des tests : sans elle, ces tests-ci sont
inapplicables, mais les 274 autres restent valables. `importorskip` les met de côté
au lieu d'interrompre toute la collecte.
"""

import pytest

pytest.importorskip(
    "fastapi",
    reason="FastAPI absent de cet environnement : python -m pip install -r requirements.txt",
)

from fastapi.testclient import TestClient  # noqa: E402

from app.api import deps
from tests.test_verdict import _extraction


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CADASTRE_DATA_DIR", str(tmp_path))
    deps.reinitialise()
    from main import app

    with TestClient(app) as c:
        yield c
    deps.reinitialise()


def _corps(document_id="D"):
    ex = _extraction()
    ex.document_id = document_id
    return {"extraction": ex.model_dump(mode="json")}


def test_toutes_les_routes_sont_publiees(client):
    chemins = client.get("/openapi.json").json()["paths"]
    for attendu in ("/health", "/documents", "/parcelles", "/parcelles/confrontation",
                    "/revue/{document_id}"):
        assert attendu in chemins, f"{attendu} absent de l'OpenAPI"


def test_base_vide(client):
    assert client.get("/health").json() == {"statut": "ok", "parcelles": 0}
    assert client.get("/parcelles").json()["total"] == 0


def test_enregistrer_puis_relire(client):
    r = client.post("/parcelles", json=_corps())
    assert r.status_code == 201, r.text
    assert r.json()["jugement"]["entre_en_base"] is True

    lu = client.get("/parcelles/D").json()
    assert lu["document_id"] == "D"
    assert lu["parcelle"]["comparable"] is True
    assert len(lu["parcelle"]["sommets"]) == 5
    assert client.get("/health").json()["parcelles"] == 1


def test_une_extraction_a_relire_est_refusee(client):
    corps = _corps()
    corps["extraction"]["datum"] = "inconnu"          # datum absent : revue obligatoire
    r = client.post("/parcelles", json=corps)
    assert r.status_code == 409
    assert "exploitable" in r.json()["detail"]
    assert client.get("/parcelles").json()["total"] == 0


def test_confrontation_nenregistre_rien(client):
    client.post("/parcelles", json=_corps("VOISIN"))
    r = client.post("/parcelles/confrontation", json=_corps("NOUVEAU"))
    assert r.status_code == 200
    # le même terrain exactement : inclusion totale, donc conflit
    assert r.json()["conflits"] == 1
    assert client.get("/parcelles").json()["total"] == 1      # rien n'a été écrit


def test_recouvrements_dune_parcelle_enregistree(client):
    client.post("/parcelles", json=_corps("A"))
    client.post("/parcelles", json=_corps("B"))
    r = client.get("/parcelles/A/recouvrements").json()
    assert r["conflits"] == 1 and r["recouvrements"][0]["b"] in ("TF 1738/KL", "B")


def test_document_inconnu(client):
    assert client.get("/parcelles/FANTOME").status_code == 404
    assert client.delete("/parcelles/FANTOME").status_code == 404


def test_suppression(client):
    client.post("/parcelles", json=_corps())
    assert client.delete("/parcelles/D").status_code == 204
    assert client.get("/parcelles/D").status_code == 404


def test_format_de_fichier_refuse(client):
    r = client.post("/documents", files={"fichier": ("notes.txt", b"rien", "text/plain")})
    assert r.status_code == 415 and ".pdf" in r.json()["detail"]


# ------------------------------------------------------------------ revue


def test_consigner_et_relire_une_decision(client):
    r = client.post("/revue/D", json={"action": "corrigee", "auteur": "mea",
                                      "motif": "B4 : papier 282405, lu 282408"})
    assert r.status_code == 201 and r.json()["horodatage"]
    client.post("/revue/D", json={"action": "acceptee", "auteur": "mea"})

    journal = client.get("/revue/D").json()
    assert journal["total"] == 2
    assert [d["action"] for d in journal["decisions"]] == ["corrigee", "acceptee"]


def test_action_inconnue_refusee(client):
    r = client.post("/revue/D", json={"action": "supprimee", "auteur": "mea"})
    assert r.status_code == 422 and "acceptee" in r.json()["detail"]


def test_decision_sans_auteur_refusee(client):
    r = client.post("/revue/D", json={"action": "acceptee", "auteur": "   "})
    assert r.status_code == 422
