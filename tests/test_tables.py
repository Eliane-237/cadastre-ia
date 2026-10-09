from model.tables import (
    best_coordinate_table,
    html_to_grid,
    parse_coordinate_table,
    points_from_text,
)

HTML = """<table><tr><td>Borne</td><td>X</td><td>Y</td><td>Distance (m)</td></tr>
<tr><td>B1</td><td>381210.57</td><td>1564042.96</td><td>17.00</td></tr>
<tr><td>B2</td><td>381225.56</td><td>1564034.94</td><td>19.38</td></tr>
<tr><td>B3</td><td>381236.10</td><td>1564051.22</td><td>19.73</td></tr></table>"""


def test_grille_avec_fusions():
    grid = html_to_grid(
        '<table><tr><td rowspan="2">A</td><td colspan="2">B</td></tr>'
        "<tr><td>c</td><td>d</td></tr></table>"
    )
    assert grid == [["A", "B", "B"], ["A", "c", "d"]]


def test_tableau_avec_entetes():
    t = parse_coordinate_table(html_to_grid(HTML))
    assert [(p.label, p.x, p.y) for p in t.points][:2] == [
        ("B1", "381210.57", "1564042.96"),
        ("B2", "381225.56", "1564034.94"),
    ]
    assert [(s.row_index, s.length) for s in t.sides] == [(0, "17.00"), (1, "19.38"), (2, "19.73")]


def test_colonnes_reconnues_par_les_valeurs_meme_en_desordre():
    html = """<table><tr><td>1564042.96</td><td>P1</td><td>381210.57</td></tr>
    <tr><td>1564034.94</td><td>P2</td><td>381225.56</td></tr></table>"""
    t = parse_coordinate_table(html_to_grid(html))
    assert [(p.label, p.x, p.y) for p in t.points] == [
        ("P1", "381210.57", "1564042.96"),
        ("P2", "381225.56", "1564034.94"),
    ]


def test_entete_inverse_corrige_par_les_valeurs():
    html = HTML.replace("<td>X</td><td>Y</td>", "<td>Y</td><td>X</td>")
    t = parse_coordinate_table(html_to_grid(html))
    assert t.points[0].x == "381210.57"


def test_cotes_entre_libelles():
    html = """<table><tr><th>Côté</th><th>Distance</th><th>X</th><th>Y</th></tr>
    <tr><td>B1-B2</td><td>17.00</td><td>381210.57</td><td>1564042.96</td></tr>
    <tr><td>B2-B3</td><td>19.38</td><td>381225.56</td><td>1564034.94</td></tr></table>"""
    t = parse_coordinate_table(html_to_grid(html))
    assert [(s.from_label, s.to_label, s.length) for s in t.sides] == [
        ("B1", "B2", "17.00"),
        ("B2", "B3", "19.38"),
    ]


def test_x_et_y_fusionnes_dans_une_cellule():
    html = """<table><tr><td>B1</td><td>381210.57 1564042.96</td></tr>
    <tr><td>B2</td><td>381225.56 1564034.94</td></tr></table>"""
    t = parse_coordinate_table(html_to_grid(html))
    assert [(p.label, p.x, p.y) for p in t.points][1] == ("B2", "381225.56", "1564034.94")


def test_tableau_sans_coordonnees_ignore():
    assert best_coordinate_table(["<table><tr><td>Nom</td><td>Date</td></tr></table>"]) is None
    assert (
        best_coordinate_table(["<table><tr><td>a</td></tr></table>", HTML]).points[2].label == "B3"
    )


def test_repli_texte():
    pts = points_from_text("Tableau\nB1 381210.57 1564042.96\nB2  381 225,56  1 564 034,94\nfin")
    assert [(p.label, p.x) for p in pts] == [("B1", "381210.57"), ("B2", "381 225,56")]
