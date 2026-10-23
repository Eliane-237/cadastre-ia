import pytest

from app.core.schema import (
    DATUM_INCONNU,
    DATUM_RRS04,
    Area,
    BoundaryPoint,
    Extracted,
    Extraction,
    Severity,
)
from app.validation.rules import (
    datum_for,
    shoelace_area,
    side_length,
    unreliable_sides,
    validate,
    validated,
)

BASE_X, BASE_Y = 300_000.0, 1_600_000.0


def ex(v):
    return Extracted[float](value=v, confidence=0.95)


def pt(i, x, y):
    return BoundaryPoint(label=f"B{i + 1}", row_index=i,
                         x=ex(x) if x is not None else Extracted[float](),
                         y=ex(y) if y is not None else Extracted[float]())


def square(side=100.0, area=None, dx=0.0):
    pts = [pt(0, BASE_X, BASE_Y), pt(1, BASE_X + side + dx, BASE_Y),
           pt(2, BASE_X + side, BASE_Y + side), pt(3, BASE_X, BASE_Y + side)]
    ca = int(area if area is not None else side * side)
    printed = Extracted[Area](value=Area(ca=ca), confidence=0.9)
    return Extraction(document_id="D", points=pts, printed_area=printed)


def codes(issues):
    return [i.code for i in issues]


def test_formule_de_gauss_et_precision():
    assert shoelace_area([(BASE_X, BASE_Y), (BASE_X + 10, BASE_Y), (BASE_X + 10, BASE_Y + 10),
                          (BASE_X, BASE_Y + 10)]) == pytest.approx(100.0)
    assert side_length((0, 0), (3, 4)) == 5.0


def test_parcelle_coherente_sans_anomalie():
    assert validate(square()) == []


def test_ecart_de_superficie():
    e = square(side=100, area=11_580)      # ~13,7 % d'écart
    assert "AREA_MISMATCH" in codes(validate(e))


def test_tolerance_de_superficie():
    assert validate(square(area=10_040)) == []                       # 0,4 % : accepté
    assert "AREA_MISMATCH" in codes(validate(square(area=10_100)))   # 1 % : refusé


def test_superficie_non_lue_est_signalee_sans_erreur():
    e = square()
    e.printed_area = Extracted[Area]()
    (i,) = validate(e)
    assert i.code == "AREA_UNAVAILABLE" and i.severity is Severity.INFO


def test_coordonnee_hors_plage_bloque_la_geometrie():
    e = square()
    e.points[1] = pt(1, 84_300_000.0, BASE_Y)
    c = codes(validate(e))
    assert "COORD_OUT_OF_RANGE" in c and "GEOMETRY_SKIPPED" in c and "AREA_MISMATCH" not in c


def test_valeur_aberrante_dans_la_plage():
    e = square()
    # plausible seule, absurde pour cette parcelle
    e.points[2] = pt(2, BASE_X + 50_000, BASE_Y + 100)
    assert "COORD_OUTLIER" in codes(validate(e))


def test_point_incomplet():
    e = square()
    e.points[1] = pt(1, BASE_X + 100, None)
    assert "POINT_INCOMPLETE" in codes(validate(e))


def test_trop_peu_de_points():
    e = Extraction(document_id="D", points=[pt(0, BASE_X, BASE_Y), pt(1, BASE_X + 5, BASE_Y)])
    assert "TOO_FEW_POINTS" in codes(validate(e))


def test_polygone_croise():
    pts = [pt(0, BASE_X, BASE_Y), pt(1, BASE_X + 100, BASE_Y + 100),
           pt(2, BASE_X + 100, BASE_Y), pt(3, BASE_X, BASE_Y + 100)]   # papillon
    e = Extraction(document_id="D", points=pts,
                   printed_area=Extracted[Area](value=Area(ca=5000), confidence=0.9))
    assert "POLYGON_INVALID" in codes(validate(e))


def test_point_de_fermeture_repete_est_ignore():
    e = square()
    e.points.append(pt(4, BASE_X, BASE_Y))
    assert validate(e) == []


def test_cotes_coherents():
    e = square()
    e.printed_sides = [ex(100.0)] * 4
    assert validate(e) == []


def test_sommet_suspect_localise_par_deux_cotes_adjacents():
    e = square(dx=10, area=10_500)         # B2 mal lu : côtés B1-B2 et B2-B3 faux
    e.printed_sides = [ex(100.0)] * 4
    issues = validate(e)
    suspects = [i for i in issues if i.code == "POINT_SUSPECT"]
    assert [s.point_labels for s in suspects] == [["B2"]]
    assert sum(1 for i in issues if i.code == "SIDE_MISMATCH") == 2


def test_un_seul_cote_faux_ne_designe_pas_de_suspect():
    e = square()
    e.printed_sides = [ex(100.0), ex(100.0), ex(100.0), ex(97.0)]
    c = codes(validate(e))
    assert c == ["SIDE_MISMATCH"]


def test_validated_ajoute_sans_toucher_loriginal():
    e = square(area=11_580)
    v = validated(e)
    assert e.issues == [] and "AREA_MISMATCH" in codes(v.issues) and v.needs_review()


# --------------------------------------------------------- colonne des côtés imprimés

_NGUEKOKH = [21.64, 13.68, 19.32, 12.44, 12.44, 21.05, 30.94, 19.15, 30.94,
             12.44, 12.44, 12.44, 12.44, 12.44, 12.44]


@pytest.mark.parametrize(
    "printed, printed_m2, motif",
    [
        # Colonnes réellement lues sur le corpus, et pourquoi elles sont fausses.
        ([11.6] * 4, 221.0, "colonne constante"),            # tf kaolack : côtés réels ~15 m
        ([15.0] * 4, 300.0, "colonne constante"),            # images.jfif : parcelle 15 x 20
        ([16.0] * 15, 649.0, "colonne constante"),           # images (7)
        ([46.83, 101.4, 0.17, 0.29, 0.37, 0.47, 0.45, 101.4], 4489.0, "découpage"),  # images (4)
        (_NGUEKOKH, 9397.0, "périmètre"),                    # côtés partiellement recopiés
        # Inégalité polygonale : un côté ne peut dépasser la somme des autres.
        ([100.0, 10.0, 10.0], 500.0, "somme des autres"),
    ],
)
def test_cotes_imprimes_ecartes(printed, printed_m2, motif):
    raison = unreliable_sides(printed, printed_m2)
    assert raison is not None and motif in raison


@pytest.mark.parametrize(
    "printed, printed_m2",
    [
        ([16.67, 15.35, 16.68, 15.26], 255.0),                       # tf.jpg
        ([157.42, 96.82, 190.8, 113.94, 62.58, 225.5, 232.8, None], 58941.0),  # 54987580
        ([39.73, 61.5, 22.2, 47.0, 20.1], 2100.0),                   # tf mbour
        ([30.59, 27.34, 29.47, 26.43], None),                        # superficie non lue
        ([10.0, 15.0, 10.0, 15.0], 150.0),                           # lot 10 x 15 : côtés vrais
        ([15.0] * 4, 225.0),                 # lot carré : colonne constante légitime
        ([None] * 4, 255.0),                                         # colonne vide
    ],
)
def test_cotes_imprimes_conserves(printed, printed_m2):
    assert unreliable_sides(printed, printed_m2) is None


# ------------------------------------------- etiquettes, doublons et datum

def renomme(extraction, labels):
    pts = [p.model_copy(update={"label": lab})
           for p, lab in zip(extraction.points, labels, strict=False)]
    return extraction.model_copy(update={"points": pts})


def test_etiquette_repetee():
    issues = validate(renomme(square(), ["B1", "B2", "B2", "B4"]))
    assert "LABEL_DUPLICATE" in codes(issues)
    # Une etiquette repetee est une erreur : la geometrie n'est pas calculee derriere.
    assert "GEOMETRY_SKIPPED" in codes(issues)


def test_trou_dans_la_suite_des_sommets():
    issues = validate(renomme(square(), ["B1", "B2", "B4", "B5"]))
    assert "LABEL_SEQUENCE" in codes(issues)
    assert "3" in next(i.message for i in issues if i.code == "LABEL_SEQUENCE")


def test_prefixes_melanges():
    assert "LABEL_PREFIX_MIXTE" in codes(validate(renomme(square(), ["B1", "B2", "3", "B4"])))


def test_suite_complete_sans_anomalie():
    assert validate(renomme(square(), ["B.1", "b-2", "B03", "B4"])) == []


def test_sommets_aux_memes_coordonnees():
    carre = square()
    pts = list(carre.points)
    pts[2] = pts[2].model_copy(update={"x": ex(BASE_X), "y": ex(BASE_Y)})
    issues = validate(carre.model_copy(update={"points": pts}))
    assert "POINT_DUPLICATE" in codes(issues)


def test_colonne_de_coordonnees_recopiee():
    carre = square()
    pts = [p.model_copy(update={"x": ex(BASE_X)}) for p in carre.points]
    issues = validate(carre.model_copy(update={"points": pts}))
    assert "COORD_COLUMN_SUSPECT" in codes(issues)


def test_parcelle_rectangulaire_ne_declenche_pas_le_doublon():
    # Deux sommets partagent legitimement une abscisse : rien ne doit etre signale.
    assert "COORD_COLUMN_SUSPECT" not in codes(validate(square()))


@pytest.mark.parametrize(
    "declare, datum",
    [
        ("RSS 04", DATUM_RRS04),
        ("RRS04", DATUM_RRS04),
        ("Système RRS 04", DATUM_RRS04),
        ("HATT", DATUM_INCONNU),
        ("WGS 84", DATUM_INCONNU),      # non reconnu : surtout ne rien supposer
        (None, DATUM_INCONNU),
    ],
)
def test_datum_declare(declare, datum):
    assert datum_for(declare) == datum


def test_validated_renseigne_le_datum():
    carre = square()
    assert carre.datum == DATUM_INCONNU          # rien n'est suppose par defaut
    lu = carre.model_copy(
        update={"coordinate_system": Extracted[str](value="RSS 04", confidence=0.9)}
    )
    assert validated(lu).datum == DATUM_RRS04
    assert validated(carre).datum == DATUM_INCONNU


def test_sans_point_aucun_plantage():
    # Un document dont le tableau n'a pas ete lu ne doit pas faire tomber la validation.
    vide = Extraction(document_id="D", points=[])
    assert "TOO_FEW_POINTS" in codes(validate(vide))


def test_systeme_non_reconnu_signale():
    # « RGS24 » lu sur 54987580 la ou le document imprime « RRS04 » : le datum reste
    # inconnu, mais l'operateur doit le savoir au lieu de ne rien voir.
    carre = square().model_copy(
        update={"coordinate_system": Extracted[str](value="RGS24", confidence=0.8)}
    )
    issues = validate(carre)
    assert "CRS_INCONNU" in codes(issues)
    assert validated(carre).datum == DATUM_INCONNU


# ------------------------------------------------- regularite des colonnes

def colonne(xs, ys):
    pts = [BoundaryPoint(label=f"B{i+1}", row_index=i,
                         x=Extracted[float](value=float(x), confidence=1.0, raw_text=x),
                         y=Extracted[float](value=float(y), confidence=1.0, raw_text=y))
           for i, (x, y) in enumerate(zip(xs, ys, strict=False))]
    modifiee = square().model_copy(
        update={"points": pts, "printed_area": Extracted[Area]()})
    return codes(validate(modifiee))


Y_KAOLACK = ["1564042.96", "1564034.94", "1564023.49", "1564031.51"]


def test_coordonnee_amputee_de_son_prefixe():
    # tf kaolack : « 38210.57 » au lieu de « 381210.57 ». Repere sans aucune geometrie,
    # ce qui compte quand la superficie imprimee manque.
    trouves = colonne(["38210.57", "381225.56", "381219.40", "381204.41"], Y_KAOLACK)
    assert "COLUMN_LENGTH" in trouves
    assert "COLUMN_PREFIX" in trouves


def test_decimales_irregulieres():
    # Grand Dakar : « 236867.4 » a une seule decimale la ou la colonne en a deux.
    trouves = colonne(["236869.09", "236899.63", "236896.76", "236867.4"],
                      ["1628753.88", "1628752.08", "1628724.89", "1628727.50"])
    assert "COLUMN_DECIMALS" in trouves


def test_colonne_reguliere_sans_bruit():
    trouves = colonne(["381210.57", "381225.56", "381219.40", "381204.41"], Y_KAOLACK)
    assert not {"COLUMN_LENGTH", "COLUMN_DECIMALS", "COLUMN_PREFIX"} & set(trouves)


# ---------------------------------------------- coherence de la localite

KAOLACK = [(381210.57, 1564042.96), (381225.56, 1564034.94),
           (381236.10, 1564051.22), (381208.44, 1564060.13)]


def parcelle_a(coords, **champs):
    pts = [BoundaryPoint(label=f"B{i+1}", row_index=i, x=ex(x), y=ex(y))
           for i, (x, y) in enumerate(coords)]
    maj = {"points": pts, "printed_area": Extracted[Area]()}
    maj.update({k: Extracted[str](value=v, confidence=0.9) for k, v in champs.items()})
    return square().model_copy(update=maj)


def test_parcelle_chez_elle():
    assert "LOCALITE_INCOHERENTE" not in codes(validate(parcelle_a(KAOLACK, commune="Kaolack")))


def test_parcelle_loin_de_sa_commune():
    # Les coordonnees sont a Kaolack, le document annonce Dakar : 157 km.
    issues = validate(parcelle_a(KAOLACK, commune="Dakar"))
    assert "LOCALITE_INCOHERENTE" in codes(issues)
    assert "Dakar" in next(i.message for i in issues if i.code == "LOCALITE_INCOHERENTE")


def test_region_vaste_toleree():
    # Une region est grande : Kaolack reste dans la region de Kaolack sans alerte,
    # alors que le meme ecart a l'echelle d'une commune serait signale.
    assert "LOCALITE_INCOHERENTE" not in codes(validate(parcelle_a(KAOLACK, region="Kaolack")))


def test_localite_inconnue_ne_dit_rien():
    # La base ne couvre que les localites principales : un nom inconnu n'accuse personne.
    assert "LOCALITE_INCOHERENTE" not in codes(
        validate(parcelle_a(KAOLACK, commune="Ngaparou")))


def test_niveau_le_plus_precis_retenu():
    # La commune est connue et fausse : on ne se rabat pas sur la region pour l'excuser.
    assert "LOCALITE_INCOHERENTE" in codes(
        validate(parcelle_a(KAOLACK, commune="Dakar", region="Kaolack")))


# ------------------------------------------- un titre n'est pas son propre titre mere


def _titres(titre, mere):
    from app.core.schema import Extracted

    return Extraction(
        document_id="D",
        title_number=Extracted[str](value=titre, confidence=0.9) if titre else Extracted[str](),
        mother_title=Extracted[str](value=mere, confidence=0.9) if mere else Extracted[str](),
    )


@pytest.mark.parametrize(
    "titre, mere, signale",
    [
        ("1738/DK", "1738/DK", True),                  # le cas reel de titre-foncier
        ("1738/DK", "1738/DK (ex 3016/DG)", True),     # meme numero, ancien rappel en plus
        ("TF 1738/DK", "1738/DK", True),               # le prefixe ne distingue pas
        ("TF 1738/DK", "TM 1738/DK", True),
        ("22 665/GR", "976/GR(ex 1 306/DG)", False),   # deux titres differents : rien a dire
        ("6652/TH", "6652", False),                    # prudence : on ne signale pas un prefixe
        (None, "1738/DK", False),                      # une seule case lue : rien a comparer
        ("1738/DK", None, False),
    ],
)
def test_titre_egal_mere(titre, mere, signale):
    from app.validation.rules import _check_titre_mere

    codes = [i.code for i in _check_titre_mere(_titres(titre, mere))]
    assert ("TITRE_EGAL_MERE" in codes) is signale


def test_titre_egal_mere_envoie_en_revue_sans_bloquer_la_geometrie():
    from app.core.schema import Severity
    from app.validation.rules import _check_titre_mere

    ex = _titres("1738/DK", "1738/DK")
    ex.issues = _check_titre_mere(ex)
    assert ex.issues[0].severity is Severity.WARNING
    assert not ex.has_error          # les coordonnees restent exploitables
    assert ex.needs_review()         # mais un humain doit trancher


# --- Rapprochement des identités de propriétaires ----------------------------
# Le même requérant ne s'écrit pas deux fois pareil d'un titre à l'autre. Ce que
# ces tests gardent surtout, c'est ce que le module refuse de rapprocher.

from app.validation.parties import (  # noqa: E402
    Identite,
    Lien,
    Niveau,
    concordance,
    jetons,
    lien,
    patronyme,
    rapprocher,
    similarite,
)


def test_jetons_ignorent_civilite_ordre_et_ponctuation():
    assert jetons("M. Mamadou DIOP") == jetons("Diop, Mamadou") == ("diop", "mamadou")


def test_jetons_recollent_les_sigles():
    """« S.C.I. » doit devenir un jeton, pas trois lettres isolées aussitôt écartées."""
    assert jetons("S.C.I. Les Baobabs") == ("baobabs", "sci")


@pytest.mark.parametrize("a, b", [
    ("SOCIETE IMMOBILIERE DU CAP-VERT", "Sté Immobilière du Cap Vert"),
    ("DIOP Mamadou", "Mamadou DIOP"),
    ("El Hadj Ousmane NDIAYE", "Ousmane Ndiaye"),
    ("Ets SALL et Frères", "ETABLISSEMENTS SALL ET FRERES"),
    ("G.I.E. NDAR", "GIE Ndar"),
])
def test_memes_parties_ecrites_autrement(a, b):
    assert similarite(a, b) == pytest.approx(1.0)


@pytest.mark.parametrize("a, b", [
    ("Fatou SALL", "Fatou FALL"),          # deux patronymes sénégalais distincts
    ("Mamadou DIOP", "Mamadou DIOUF"),
    ("Mamadou DIOP", "Aminata DIOP"),
    ("Mamadou DIOP", "SOCIETE IMMOBILIERE DU CAP-VERT"),
])
def test_parties_distinctes_ne_se_rapprochent_pas(a, b):
    """Le seuil qui repêcherait une coquille d'OCR confondrait SALL et FALL.

    `diop`/`diof` et `sall`/`fall` donnent le même rapport, 0,750 : aucun réglage ne
    sépare la coquille de la famille voisine. Un nom mal lu se corrige à l'extraction.
    """
    assert similarite(a, b) < 0.78
    assert rapprocher(a, [("D99", b)]) == []


def test_prenom_supplementaire_part_en_verification_pas_en_certitude():
    [r] = rapprocher("Mamadou Lamine DIOP", [("D99", "Mamadou DIOP")])
    assert r.niveau is Niveau.A_VERIFIER
    assert r.document == "D99"
    assert "ne suffit pas à établir une identité" in r.message


def test_rapprochement_sans_nom_ne_rend_rien():
    assert rapprocher(None, [("D99", "Mamadou DIOP")]) == []
    assert rapprocher("M.", [("D99", "Mamadou DIOP")]) == []


def test_meilleurs_scores_en_tete():
    base = [("D1", "Mamadou Lamine DIOP"), ("D2", "DIOP Mamadou")]
    scores = [r.score for r in rapprocher("Mamadou DIOP", base)]
    assert scores == sorted(scores, reverse=True)


# --- La casse dit lequel des mots est le patronyme ---------------------------
# Les titres écrivent le patronyme en capitales et le prénom en bas de casse.
# Trier les jetons sans lire la casse rendait « Mamadou DIOP » et « MAMADOU Diop »
# identiques : deux personnes sans rapport, rapprochées à 100 %.

@pytest.mark.parametrize("nom, attendu", [
    ("DIOP Mamadou", ("diop",)),
    ("Mamadou DIOP", ("diop",)),
    ("M. Mamadou DIOP", ("diop",)),
    ("El Hadj Ousmane NDIAYE", ("ndiaye",)),
    ("MAMADOU Diop", ("mamadou",)),
])
def test_patronyme_se_lit_a_la_casse(nom, attendu):
    assert patronyme(nom) == attendu


@pytest.mark.parametrize("nom", [
    "MAMADOU DIOP",                      # tout en capitales : l'OCR en produit souvent
    "mamadou diop",                      # tout en bas de casse
    "SOCIETE IMMOBILIERE DU CAP-VERT",   # personne morale : la convention ne s'applique pas
    "Sté Immobilière du Cap Vert",
    "S.C.I. Les Baobabs",
])
def test_patronyme_vide_quand_le_document_ne_distingue_pas(nom):
    """Un signal absent ne se devine pas."""
    assert patronyme(nom) == ()


def test_patronymes_differents_ecartent_deux_noms_pourtant_identiques():
    """Le cas que le tri des jetons rendait invisible."""
    assert similarite("Mamadou DIOP", "MAMADOU Diop") == 0.0
    assert rapprocher("Mamadou DIOP", [("D99", "MAMADOU Diop")]) == []


def test_sans_signal_de_casse_on_retombe_sur_les_jetons():
    assert similarite("MAMADOU DIOP", "Mamadou Diop") == pytest.approx(1.0)


def test_la_casse_ecarte_mais_ne_rapproche_jamais():
    """Un même patronyme ne suffit pas : le prénom diffère, le score reste bas."""
    assert patronyme("Mamadou DIOP") == patronyme("Aminata DIOP") == ("diop",)
    assert similarite("Mamadou DIOP", "Aminata DIOP") == pytest.approx(0.5)


def test_personne_morale_a_la_casse_incoherente_reste_rapprochee():
    """« STE Immobilière » et « Sté IMMOBILIERE » : la règle du patronyme ne s'applique
    pas aux personnes morales, dont la casse ne suit aucune convention."""
    assert similarite("STE Immobilière du Cap-Vert",
                      "Sté IMMOBILIERE du Cap-Vert") == pytest.approx(1.0)


# --- Le lien d'abord, le nom ensuite ------------------------------------------
# Un nom seul ne rattache pas deux titres : des milliers de personnes portent le
# meme au Senegal. On ne confronte deux identites que sur un lien deja etabli.

def _parcelle_test(doc, titre=None, mere=None, nicad=None, owner=None):
    from shapely.geometry import Polygon

    from app.geospatial.parcel import Parcel
    return Parcel(document_id=doc, polygon=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                  crs="EPSG:32628", datum=DATUM_RRS04, title_number=titre,
                  mother_title=mere, nicad=nicad, owner=owner)


def test_nicad_identique_lie_deux_titres():
    a = _parcelle_test("A", nicad="0113 0134")
    assert lien(a, _parcelle_test("B", nicad="01130134")) is Lien.NICAD


def test_filiation_declaree_lie_deux_titres():
    a = _parcelle_test("A", titre="22 665/GR")
    assert lien(a, _parcelle_test("B", mere="TF n° 22.665/GR")) is Lien.TITRE_MERE


def test_meme_nom_sans_lien_ne_donne_aucune_concordance():
    """Le cœur de la correction : un homonyme n'est pas un dossier."""
    a = _parcelle_test("A", titre="1/DK", owner="Mamadou DIOP")
    b = _parcelle_test("B", titre="999/DK", owner="DIOP Mamadou")
    assert lien(a, b) is None
    assert concordance(a, b) is None


def test_meme_parcelle_deux_requerants_est_a_instruire():
    a = _parcelle_test("A", nicad="01130134", owner="Mamadou DIOP")
    b = _parcelle_test("B", nicad="01130134", owner="NDIAYE Ousmane")
    c = concordance(a, b)
    assert c.identite is Identite.DIVERGENTE
    assert c.a_instruire is True
    assert "à instruire" in c.message


def test_morcellement_cede_a_un_tiers_n_est_pas_a_instruire():
    """Un lot détaché et vendu porte normalement un autre nom que le titre mère."""
    a = _parcelle_test("A", titre="22 665/GR", owner="Mamadou DIOP")
    b = _parcelle_test("B", mere="22 665/GR", owner="NDIAYE Ousmane")
    c = concordance(a, b)
    assert c.lien is Lien.TITRE_MERE
    assert c.identite is Identite.DIVERGENTE
    assert c.a_instruire is False
    assert "normal pour un morcellement" in c.message


def test_requerant_non_lu_laisse_l_identite_indeterminee():
    """Le cas le plus fréquent du corpus : le requérant n'est lu que sur un titre sur cinq."""
    a = _parcelle_test("A", nicad="01130134", owner="Mamadou DIOP")
    c = concordance(a, _parcelle_test("B", nicad="01130134", owner=None))
    assert c.identite is Identite.INDETERMINEE
    assert c.a_instruire is False
    assert "ne peut pas être vérifiée" in c.message


# --- Copropriete : le seul cas ou deux titres partagent un NICAD --------------
# Decret sur l'identification cadastrale : le NICAD est unique par parcelle
# (art. 1 et 2) et tout morcellement annule les anciens (art. 4). En copropriete
# (art. 6), il designe la parcelle d'assise et le numero de lot distingue les biens.

def _copro(doc, nicad="0113013401200430", lot=None, owner=None):
    from shapely.geometry import Polygon

    from app.geospatial.parcel import Parcel
    return Parcel(document_id=doc, polygon=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                  crs="EPSG:32628", datum=DATUM_RRS04, nicad=nicad, lot=lot, owner=owner)


def test_meme_nicad_lots_distincts_est_une_copropriete():
    a = _copro("A", nicad="0113 0134 0120 0430", lot="Lot n° 12", owner="DIOP Mamadou")
    b = _copro("B", lot="LOT 7", owner="NDIAYE Ousmane")
    c = concordance(a, b)
    assert c.lien is Lien.COPROPRIETE
    assert c.a_instruire is False
    assert "normal en copropriété" in c.message


def test_meme_nicad_meme_lot_deux_requerants_est_a_instruire():
    a = _copro("A", lot="Lot n° 12", owner="DIOP Mamadou")
    c = concordance(a, _copro("B", lot="lot 12", owner="NDIAYE Ousmane"))
    assert c.lien is Lien.NICAD
    assert c.a_instruire is True
    assert c.reserve is None


def test_lot_non_lu_signale_sa_reserve():
    """Sans numéro de lot, une copropriété ne peut pas être écartée : l'agent doit le savoir."""
    a = _copro("A", lot="Lot n° 12", owner="DIOP Mamadou")
    c = concordance(a, _copro("B", lot=None, owner="NDIAYE Ousmane"))
    assert c.a_instruire is True
    assert "copropriété ne peut pas être écartée" in c.message
