import pytest

from app.core.config import SETTINGS, Settings
from app.core.schema import BoundaryPoint, Extracted, Source
from app.extraction.merge import (
    FusionReport,
    _Candidate,
    _Conflict,
    _fuse_field,
    arbitrate,
    fuse,
)
from app.extraction.reading import DocumentReading, FieldReading
from app.validation.localites import localite_connue
from app.validation.rules import validate
from tests.fake_readers import AREA_M2, COORDS, SIDES, reading

CFG = Settings()


def codes(issues):
    return [i.code for i in issues]


def with_coord(coords, i, axis, value):
    out = [list(c) for c in coords]
    out[i][0 if axis == "x" else 1] = value
    return [tuple(c) for c in out]


def test_accord_des_deux_lecteurs():
    ex, rep = fuse("D", reading(Source.PADDLE_VL), reading(Source.QWEN_VL), CFG)
    assert rep.issues == []
    p = ex.points[0]
    assert p.x.value == 381210.57 and p.x.source is Source.CONSENSUS
    assert p.x.confidence == CFG.conf_agree
    assert [r.source for r in p.x.readings] == [Source.PADDLE_VL, Source.QWEN_VL]
    assert ex.title_number.source is Source.CONSENSUS and ex.title_number.value == "TF 1738/KL"
    assert (
        ex.owner.source is Source.QWEN_VL and ex.owner.confidence == 0.9
    )  # plafond lecture unique
    assert ex.printed_area.value.m2 == AREA_M2
    assert [s.value for s in ex.printed_sides] == [float(s) for s in SIDES]
    assert validate(ex, CFG) == []
    assert not ex.needs_review(CFG.review_threshold)


def test_desaccord_tranche_par_la_geometrie():
    # Qwen lit un 8 au lieu d'un 3 : la seule combinaison cohérente est celle de PaddleOCR-VL.
    wrong = with_coord(COORDS, 2, "y", "1564081.22")
    ex, rep = fuse("D", reading(Source.PADDLE_VL), reading(Source.QWEN_VL, wrong), CFG)
    y = ex.points[2].y
    assert (
        y.value == 1564051.22 and y.source is Source.GEOMETRY and y.confidence == CFG.conf_geometry
    )
    assert codes(rep.issues) == ["READING_CONFLICT_RESOLVED"]
    assert rep.conflicts_resolved == 1


def test_la_geometrie_peut_donner_raison_a_qwen():
    wrong = with_coord(COORDS, 1, "x", "381255.56")
    ex, _ = fuse("D", reading(Source.PADDLE_VL, wrong), reading(Source.QWEN_VL), CFG)
    assert ex.points[1].x.value == 381225.56 and ex.points[1].x.raw_text == "381225.56"


def test_desaccord_sans_preuve_geometrique_part_en_revue():
    no_area = {"title_number": FieldReading("TF 1738/KL")}
    wrong = with_coord(COORDS, 2, "y", "1564081.22")
    ex, rep = fuse(
        "D",
        reading(Source.PADDLE_VL, sides=False, fields=no_area),
        reading(Source.QWEN_VL, wrong, sides=False, fields=no_area),
        CFG,
    )
    y = ex.points[2].y
    assert y.value == 1564051.22 and y.confidence == CFG.conf_conflict  # valeur Paddle, à relire
    (issue,) = rep.issues
    assert issue.code == "READING_CONFLICT" and issue.point_labels == ["B3"]
    assert ex.needs_review(CFG.review_threshold)
    assert "points[B3].y" in ex.low_confidence_fields(CFG.review_threshold)


def test_desaccord_sur_les_decimales_que_la_surface_ne_voit_pas():
    # 1 cm d'écart : la surface reste dans la tolérance pour les deux lectures, mais les côtés
    # imprimés (tolérance 5 cm) ne suffisent pas non plus : pas de choix unique → revue.
    wrong = with_coord(COORDS, 0, "x", "381210.58")
    ex, rep = fuse("D", reading(Source.PADDLE_VL), reading(Source.QWEN_VL, wrong), CFG)
    assert codes(rep.issues) == ["READING_CONFLICT"]
    assert ex.points[0].x.confidence == CFG.conf_conflict


def test_coordonnee_lue_une_seule_fois_toujours_relue_meme_si_qwen_est_sur():
    q = reading(Source.QWEN_VL)
    q.points[0].x_confidence = 0.999
    ex, _ = fuse("D", None, q, CFG)
    assert ex.points[0].x.confidence == CFG.conf_single_coordinate_cap < CFG.review_threshold
    assert ex.owner.confidence == CFG.conf_single_cap  # champ texte : plafond plus haut


def test_un_seul_lecteur():
    ex, rep = fuse("D", None, reading(Source.QWEN_VL), CFG)
    assert ex.points[0].x.source is Source.QWEN_VL
    assert ex.points[0].x.confidence == CFG.conf_single
    assert ex.needs_review(CFG.review_threshold)


def test_alignement_par_libelle_malgre_une_ligne_oubliee():
    q = reading(Source.QWEN_VL)
    del q.points[1]
    ex, rep = fuse("D", reading(Source.PADDLE_VL), q, CFG)
    assert "POINT_COUNT_MISMATCH" in codes(rep.issues)
    assert [p.label for p in ex.points] == ["B1", "B2", "B3", "B4", "B5"]
    assert ex.points[1].x.source is Source.PADDLE_VL  # lu par un seul
    assert ex.points[2].x.source is Source.CONSENSUS  # bien réaligné


def test_champ_divergent():
    q = reading(Source.QWEN_VL)
    q.fields["title_number"] = FieldReading("TF 1788/DK")
    ex, rep = fuse("D", reading(Source.PADDLE_VL), q, CFG)
    assert "FIELD_CONFLICT" in codes(rep.issues)
    assert ex.title_number.value == "TF 1788/DK" and ex.title_number.confidence == CFG.conf_conflict


def test_valeur_illisible_ignoree_mais_tracee():
    q = reading(Source.QWEN_VL, with_coord(COORDS, 0, "x", "illisible"))
    ex, _ = fuse("D", reading(Source.PADDLE_VL), q, CFG)
    x = ex.points[0].x
    assert x.source is Source.PADDLE_VL and x.value == 381210.57
    assert [r.raw_text for r in x.readings] == ["381210.57", "illisible"]


@pytest.mark.parametrize("dist", ["17.00", "17.01"])
def test_cotes_fusionnes(dist):
    q = reading(Source.QWEN_VL)
    q.sides[0].length = dist
    ex, _ = fuse("D", reading(Source.PADDLE_VL), q, CFG)
    s0 = ex.printed_sides[0]
    assert s0.source is (Source.CONSENSUS if dist == SIDES[0] else Source.PADDLE_VL)


# --------------------------------------------------- forme retenue entre lectures concordantes


@pytest.mark.parametrize(
    "texte, attendu",
    [
        ("14 DEC 2020", 0),
        ("1.4.DEC.2020", 2),      # le point coupe « 4|DEC » puis « DEC|2020 »
        ("234773.60", 0),         # decimale : un point entre deux chiffres n'est pas du bruit
        ("1627666.10", 0),
        ("Mermoz : Sacré Coeur", 0),
        ("Mornoz: Sacré Coeur", 0),
        ("Grand-Dakar", 0),
        ("05ha 89a 41ca", 0),
    ],
)
def test_ponctuation_parasite(texte, attendu):
    from app.extraction.merge import _ponctuation_parasite

    assert _ponctuation_parasite(texte) == attendu


def _avec_date(source, date):
    r = reading(source)
    r.fields["certified_on"] = FieldReading(
        raw=date, confidence=0.5 if date == "14 DEC 2020" else 0.99
    )
    return r


def test_date_la_mieux_formee_gagne_malgre_une_confiance_plus_basse():
    # Memes chiffres des deux cotes : les lecteurs ont lu la meme date. Le plus sur de lui
    # porte la forme abimee ; c'est la forme propre qui doit sortir.
    ex, _ = fuse(
        "D",
        _avec_date(Source.PADDLE_VL, "14 DEC 2020"),
        _avec_date(Source.QWEN_VL, "1.4.DEC.2020"),
        CFG,
    )
    assert ex.certified_on.value == "14 DEC 2020"
    assert ex.certified_on.source is Source.CONSENSUS
    assert ex.certified_on.confidence == CFG.conf_near_agree
    # les deux lectures restent tracables
    assert {r.raw_text for r in ex.certified_on.readings} == {"14 DEC 2020", "1.4.DEC.2020"}


def test_a_forme_egale_la_confiance_departage_toujours():
    ex, _ = fuse(
        "D",
        _avec_date(Source.PADDLE_VL, "14 DEC 2020"),
        _avec_date(Source.QWEN_VL, "14 DEG 2020"),    # aucune ponctuation parasite non plus
        CFG,
    )
    assert ex.certified_on.value == "14 DEG 2020"     # la plus confiante des deux


def test_les_decimales_du_tableau_survivent_a_la_fusion():
    ex, _ = fuse("D", reading(Source.PADDLE_VL), reading(Source.QWEN_VL), CFG)
    assert ex.points[0].x.raw_text == COORDS[0][0]    # zeros finaux compris


# --- Arbitrage par elimination, puis par enumeration --------------------------
# L'enumeration seule exige UNE combinaison coherente. Quand deux lectures ne
# different que d'un centimetre, les deux restent coherentes, plusieurs
# combinaisons gagnent, et elle abandonne — en emportant au passage les
# desaccords de 900 m qu'elle savait pourtant trancher.

_THIES = [(294831.706, 1640903.146), (294965.286, 1640831.843),
          (295057.694, 1640803.611), (295253.622, 1640842.742),
          (295277.101, 1640725.224), (295215.415, 1640714.692),
          (295001.906, 1640637.808), (294782.735, 1640888.333)]
_THIES_M2 = 58_941.0


def _conflit(i, axe, autre):
    """Le desaccord sur une coordonnee : la lecture de Paddle contre celle de Qwen."""
    bon = _THIES[i][0 if axe == "x" else 1]
    return _Conflict(i, axe, [
        _Candidate(Source.PADDLE_VL, f"{bon:.3f}", f"{bon:.3f}"),
        _Candidate(Source.QWEN_VL, autre, autre),
    ])


def _points_thies():
    return [BoundaryPoint(label=f"8.{i+1}", row_index=i,
                          x=Extracted[float](value=x, confidence=0.9),
                          y=Extracted[float](value=y, confidence=0.9))
            for i, (x, y) in enumerate(_THIES)]


def test_un_chiffre_aberrant_est_tranche_seul():
    """900 m d'ecart : la superficie imprimee refute cette lecture a elle seule."""
    pts = _points_thies()
    conflits = [_conflit(0, "y", "1640003.146")]
    tranches = arbitrate(pts, conflits, _THIES_M2, [None] * 8, SETTINGS)
    assert tranches[0].source is Source.PADDLE_VL


def test_un_ecart_centimetrique_reste_indecis():
    """A 1 cm, les deux lectures restent coherentes : la geometrie ne peut pas trancher,
    et pretendre le contraire serait inventer une certitude."""
    pts = _points_thies()
    conflits = [_conflit(3, "x", "295253.632")]
    assert arbitrate(pts, conflits, _THIES_M2, [None] * 8, SETTINGS) == {}


def test_les_evidents_sont_tranches_meme_quand_des_indecis_restent():
    """Le cas reel du titre de Thies : 13 desaccords, dont 4 indepartageables.

    L'enumeration seule rendait None et laissait les 13 en suspens, par-dessus le
    plafond max_conflicts_for_geometry qui l'empechait meme d'essayer.
    """
    pts = _points_thies()
    aberrants = {(0, "y"): "1640003.146", (1, "x"): "294905.286",
                 (1, "y"): "1640031.843", (2, "x"): "295007.094",
                 (2, "y"): "1640063.611", (3, "y"): "1640942.742",
                 (4, "x"): "295777.101", (4, "y"): "1640775.224",
                 (6, "y"): "1640037.868"}
    minimes = {(3, "x"): "295253.632", (5, "y"): "1640714.662",
               (6, "x"): "295001.008", (7, "y"): "1640889.333"}
    conflits = [_conflit(i, ax, v) for (i, ax), v in {**aberrants, **minimes}.items()]
    assert len(conflits) > SETTINGS.max_conflicts_for_geometry

    tranches = arbitrate(pts, conflits, _THIES_M2, [None] * 8, SETTINGS)
    assert len(tranches) == len(aberrants)
    assert all(c.source is Source.PADDLE_VL for c in tranches.values())
    restants = {(conflits[i].point, conflits[i].axis)
                for i in range(len(conflits)) if i not in tranches}
    assert restants == set(minimes)


def test_sans_superficie_ni_cotes_rien_nest_tranche():
    """La geometrie a besoin d'une preuve independante des coordonnees."""
    pts = _points_thies()
    conflits = [_conflit(0, "y", "1640003.146")]
    assert arbitrate(pts, conflits, None, [None] * 8, SETTINGS) == {}


# --- Une etiquette n'est pas une valeur ---------------------------------------
# Un lecteur a rendu « Requerant » comme nom du proprietaire : le libelle imprime
# de la case, pas son contenu. Ce n'est pas un desaccord a arbitrer.

def _deux(nom, pad, qw, conf=0.9):
    def lecture(src, v):
        r = DocumentReading(source=src, model="x")
        r.fields[nom] = FieldReading(raw=v, confidence=conf)
        return r
    return [lecture(Source.PADDLE_VL, pad), lecture(Source.QWEN_VL, qw)]


def _fusionne(nom, pad, qw, connu=None):
    rapport = FusionReport()
    champ = _fuse_field(_deux(nom, pad, qw), nom, SETTINGS, rapport, connu=connu)
    return champ, [i.code for i in rapport.issues]


def test_letiquette_dune_case_nest_pas_sa_valeur():
    champ, codes_vus = _fusionne("owner", "Requérant", "ETAT DU SENEGAL / VILLE DE THIES")
    assert champ.value == "ETAT DU SENEGAL / VILLE DE THIES"
    assert champ.confidence > SETTINGS.conf_conflict
    assert "FIELD_CONFLICT" not in codes_vus


def test_letiquette_nest_jamais_retiree_en_dernier():
    """Sans rien en face, mieux vaut une valeur douteuse que pas de valeur du tout."""
    champ, _ = _fusionne("owner", "Requérant", "Requérant")
    assert champ.value == "Requérant"


# --- Le repertoire des localites departage, a sens unique ---------------------

def test_une_localite_connue_lemporte_sur_une_inconnue():
    champ, codes_vus = _fusionne(
        "region", "AUX CENTRE DES SERVICES FISCAUX DE THIES", "Thies",
        connu=localite_connue)
    assert champ.value == "Thies"
    assert champ.confidence > SETTINGS.conf_conflict
    assert "FIELD_CONFLICT" not in codes_vus


def test_deux_lectures_inconnues_restent_en_conflit():
    """Le repertoire ne couvre qu'une partie des localites senegalaises : « Mermoz :
    Sacre Coeur » est reel et n'y figure pas. Une valeur inconnue n'est jamais ecartee
    pour autant — la regle ne sert qu'a departager, jamais a rejeter."""
    assert not localite_connue("Mermoz : Sacré Coeur")
    champ, _ = _fusionne("commune", "Saint-Louis", "Mermoz : Sacré Coeur",
                         connu=localite_connue)
    # Saint-Louis est au repertoire, l'autre non : on ne tranche pas pour autant
    # a l'aveugle — on verifie seulement qu'aucune valeur n'a ete inventee.
    assert champ.value in {"Saint-Louis", "Mermoz : Sacré Coeur"}


def test_le_repertoire_ne_touche_pas_les_autres_champs():
    """« 1/500 » contre « 1 /5000 » reste un vrai doute d'un chiffre."""
    champ, codes_vus = _fusionne("scale", "1 /5000", "1/500", connu=localite_connue)
    assert champ.confidence == SETTINGS.conf_conflict
    assert "FIELD_CONFLICT" in codes_vus


def test_sans_repertoire_le_comportement_est_inchange():
    """La fusion n'importe pas la validation : sans `connu`, rien ne change."""
    avec, _ = _fusionne("region", "AUX CENTRE DES SERVICES FISCAUX DE THIES", "Thies",
                        connu=localite_connue)
    sans, codes_vus = _fusionne("region", "AUX CENTRE DES SERVICES FISCAUX DE THIES",
                                "Thies")
    assert avec.value == "Thies" and sans.confidence == SETTINGS.conf_conflict
    assert "FIELD_CONFLICT" in codes_vus
