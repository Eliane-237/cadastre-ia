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


def test_un_doute_nempeche_pas_lenregistrement(client):
    """Sans systeme de coordonnees imprime, le datum reste inconnu : la parcelle part
    « a relire », mais l'agent peut l'inscrire. Elle entre en base marquee non
    comparable — la refuser definitivement ne servirait personne."""
    corps = _corps()
    corps["extraction"]["datum"] = "inconnu"
    r = client.post("/parcelles", json=corps)
    assert r.status_code == 201, r.text
    assert r.json()["jugement"]["decision"] == "a_relire"
    assert r.json()["jugement"]["entre_en_base"] is True
    assert client.get("/parcelles/D").json()["parcelle"]["comparable"] is False


def test_une_geometrie_impossible_reste_refusee(client):
    """La seule porte qui reste fermee : sans polygone, il n'y a rien a inscrire."""
    corps = _corps()
    corps["extraction"]["points"] = corps["extraction"]["points"][:2]
    r = client.post("/parcelles", json=corps)
    assert r.status_code == 422
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


# ------------------------------------------------- ce qui se voit : tracé et carte


def test_tracer_une_extraction_sans_la_lire_ni_lenregistrer(client):
    r = client.post("/parcelles/apercu", json=_corps())
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:4] == b"\x89PNG"
    assert client.get("/parcelles").json()["total"] == 0      # rien n'a été écrit


def test_le_trace_dune_parcelle_enregistree(client):
    client.post("/parcelles", json=_corps())
    r = client.get("/parcelles/D/apercu")
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"


def test_geojson_dune_parcelle(client):
    client.post("/parcelles", json=_corps())
    g = client.get("/parcelles/D/geojson").json()
    assert g["type"] == "FeatureCollection" and len(g["features"]) == 1
    lon, lat = g["features"][0]["geometry"]["coordinates"][0][0]
    assert -18 < lon < -11 and 12 < lat < 17            # le Sénégal, en WGS84
    assert "datum" in g["note"]                        # la réserve accompagne la sortie


def test_la_carte_de_toute_la_base(client):
    assert client.get("/parcelles/geojson").json()["features"] == []
    client.post("/parcelles", json=_corps("A"))
    client.post("/parcelles", json=_corps("B"))
    assert len(client.get("/parcelles/geojson").json()["features"]) == 2


def test_la_confrontation_rend_de_quoi_lafficher(client):
    client.post("/parcelles", json=_corps("VOISIN"))
    r = client.post("/parcelles/confrontation", json=_corps("NOUVEAU")).json()
    assert r["centre"] is not None
    # la nouvelle parcelle et sa voisine sur la même couche
    assert len(r["geojson"]["features"]) == 2


def test_une_geometrie_irrecevable_ne_se_trace_pas(client):
    corps = _corps()
    corps["extraction"]["points"] = corps["extraction"]["points"][:2]   # 2 sommets
    assert client.post("/parcelles/apercu", json=corps).status_code == 422
