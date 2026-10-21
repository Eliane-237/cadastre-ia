"""La persistance : ce qui entre, ce qui est refusé, et ce qu'on retrouve."""

import json

import pytest

from app.geospatial.parcel import parcel
from app.review.history import Action, Historique
from app.review.history import Decision as DecisionHumaine
from app.storage.json_store import JsonParcelStore
from app.storage.repository import NonEnregistrable, ParcelRepository
from app.validation.verdict import Decision, Jugement, juger
from tests.test_verdict import _extraction


@pytest.fixture
def base(tmp_path):
    return JsonParcelStore(tmp_path / "parcelles")


def _couple(document_id="D"):
    ex = _extraction()
    ex.document_id = document_id
    return parcel(ex), ex, juger(ex)


def test_le_magasin_json_respecte_le_contrat(base):
    assert isinstance(base, ParcelRepository)


def test_aller_retour(base):
    p, ex, j = _couple()
    base.save(p, ex, j)
    relue = base.get("D")
    assert relue is not None
    assert relue.parcelle.polygon.equals(p.polygon)
    assert relue.extraction.points[0].x.raw_text == ex.points[0].x.raw_text
    assert relue.jugement.decision is Decision.EXPLOITABLE


def test_une_parcelle_a_relire_est_refusee(base):
    p, ex, _ = _couple()
    with pytest.raises(NonEnregistrable, match="exploitable"):
        base.save(p, ex, Jugement(Decision.A_RELIRE, ["datum inconnu"]))
    assert base.all() == []


def test_un_refus_est_refuse(base):
    p, ex, _ = _couple()
    with pytest.raises(NonEnregistrable):
        base.save(p, ex, Jugement(Decision.REFUSE, ["AREA_MISMATCH"]))


def test_parcelle_et_extraction_doivent_parler_du_meme_document(base):
    p, ex, j = _couple("D")
    autre = _extraction()
    autre.document_id = "AUTRE"
    with pytest.raises(NonEnregistrable, match="on n'enregistre pas"):
        base.save(p, autre, j)


def test_identifiant_avec_une_barre_oblique(base):
    p, ex, j = _couple("6652/TH")
    base.save(p, ex, j)
    assert base.get("6652/TH") is not None     # « / » n'est pas un dossier


def test_near_ne_rend_que_ce_qui_touche(base):
    p, ex, j = _couple()
    base.save(p, ex, j)
    x0, y0, x1, y1 = p.polygon.bounds
    assert len(base.near((x0 - 5, y0 - 5, x1 + 5, y1 + 5))) == 1
    assert base.near((x0 + 10_000, y0, x1 + 10_000, y1)) == []


def test_le_polygone_est_reconstruit_et_non_relu(base):
    """Un polygone enregistré qui cesserait de suivre son extraction serait un mensonge."""
    p, ex, j = _couple()
    base.save(p, ex, j)
    brut = json.loads((base.dossier / "D.json").read_text(encoding="utf-8"))
    assert "polygon" not in brut and "coordinates" not in json.dumps(brut["jugement"])
    assert "extraction" in brut and "emprise" in brut


def test_un_fichier_abime_nefface_pas_la_base(base):
    p, ex, j = _couple("BON")
    base.save(p, ex, j)
    (base.dossier / "ABIME.json").write_text("{ pas du json", encoding="utf-8")
    assert [x.document_id for x in base.all()] == ["BON"]
    assert len(base.near((-1e9, -1e9, 1e9, 1e9))) == 1


def test_suppression(base):
    p, ex, j = _couple()
    base.save(p, ex, j)
    assert base.delete("D") is True
    assert base.delete("D") is False and base.get("D") is None


# --------------------------------------------------------------- historique


def test_historique_ajoute_et_ne_reprend_jamais(tmp_path):
    h = Historique(tmp_path / "h.jsonl")
    h.consigne(DecisionHumaine("D", Action.CORRIGEE, "mea", "B4 lu 282408, papier 282405"))
    h.consigne(DecisionHumaine("D", Action.ACCEPTEE, "mea"))
    h.consigne(DecisionHumaine("AUTRE", Action.REJETEE, "autre", "cachet absent"))

    assert [d.action for d in h.pour("D")] == [Action.CORRIGEE, Action.ACCEPTEE]
    assert len(h.tout()) == 3
    assert h.pour("D")[0].horodatage                      # horodaté d'office
    assert h.chemin.read_text(encoding="utf-8").count("\n") == 3   # une ligne par décision


def test_une_ligne_abimee_nefface_pas_les_autres(tmp_path):
    h = Historique(tmp_path / "h.jsonl")
    h.consigne(DecisionHumaine("D", Action.ACCEPTEE, "mea"))
    with h.chemin.open("a", encoding="utf-8") as f:
        f.write("{ pas du json\n")
    h.consigne(DecisionHumaine("D", Action.COMMENTEE, "mea", "revu"))
    assert len(h.pour("D")) == 2
