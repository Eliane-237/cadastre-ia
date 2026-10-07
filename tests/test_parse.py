from model.ocr import OcrBox
from model.parse import normalize, parse, parse_area, parse_points


def box(text, x, y, conf=0.9, w=80, h=14):
    return OcrBox(text, conf, x, y, x + w, y + h)


def table(rows, x_label=20, x_x=100, x_y=220, dy=30):
    """rows : [(label, x_text, y_text)] -> boîtes OCR comme le moteur les rendrait."""
    boxes = []
    for i, (lab, xt, yt) in enumerate(rows):
        y = 100 + i * dy
        boxes.append(box(lab, x_label, y, w=30))
        if xt:
            boxes.append(box(xt, x_x, y))
        if yt:
            boxes.append(box(yt, x_y, y))
    return boxes


def test_normalize():
    assert normalize("381210,57") == "381210.57"
    assert normalize("381210 . 57") == "381210.57"


def test_tableau_simple_avec_etiquettes():
    pts = parse_points(table([("B1", "381210.57", "1564042.96"), ("B2", "381225.56", "1564034.94")]))
    assert [p.label for p in pts] == ["B1", "B2"]
    assert [p.row_index for p in pts] == [0, 1]
    assert pts[1].x.value == 381225.56 and pts[1].y.value == 1564034.94
    assert pts[0].x.bbox is not None


def test_virgule_decimale():
    (p,) = parse_points(table([("B1", "381210,57", "1564042,96")]))
    assert p.x.value == 381210.57


def test_deux_nombres_dans_une_seule_boite():
    boxes = [box("B1", 20, 100, w=30), box("381210.57 1564042.96", 100, 100, w=200)]
    boxes += [box("B2", 20, 130, w=30), box("381225.56", 100, 130), box("1564034.94", 220, 130)]
    pts = parse_points(boxes)
    assert pts[0].x.value == 381210.57 and pts[0].y.value == 1564042.96
    assert pts[1].x.value == 381225.56


def test_valeur_isolee_rattachee_a_la_bonne_colonne():
    pts = parse_points(table([("B1", "381210.57", "1564042.96"), ("B2", None, "1564034.94"),
                              ("B3", "381219.40", None)]))
    assert pts[1].x.value is None and pts[1].y.value == 1564034.94   # X manquant, Y en bonne colonne
    assert pts[2].x.value == 381219.40 and pts[2].y.value is None


def test_nombre_mal_lu_reste_dans_sa_colonne():
    # chiffre en trop : 8 chiffres avant le point, mais positionné dans la colonne Est
    pts = parse_points(table([("B1", "381210.57", "1564042.96"), ("B2", "84381225.56", "1564034.94")]))
    assert pts[1].x.value == 84381225.56      # on garde ce qui est lu : la validation le rejettera
    assert pts[1].y.value == 1564034.94


def test_etiquette_absente_donne_un_nom_par_defaut():
    boxes = [box("381210.57", 100, 100), box("1564042.96", 220, 100)]
    assert parse_points(boxes)[0].label == "P1"


def test_aucun_nombre_aucune_ligne():
    assert parse_points([box("TITRE FONCIER", 10, 10)]) == []


def test_superficie_complete_et_courte():
    full = parse_area([box("Superficie", 10, 10), box("00 ha 06 a 49 ca", 120, 10, conf=0.8)])
    assert full.value.m2 == 649 and full.confidence == 0.8
    short = parse_area([box("21a 00ca", 10, 10)])
    assert short.value.m2 == 2100


def test_superficie_absente():
    assert not parse_area([box("rien ici", 10, 10)]).present


def test_parse_complet():
    e = parse(table([("B1", "381210.57", "1564042.96")]), "D01")
    assert e.document_id == "D01" and len(e.points) == 1 and not e.title_number.present
