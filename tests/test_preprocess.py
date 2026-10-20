import cv2
import numpy as np

from model.preprocess import (
    find_document_quad,
    flatten_illumination,
    illumination_unevenness,
    load_image,
    preprocess,
)


def page(h=1100, w=800):
    img = np.full((h, w, 3), 245, np.uint8)
    for i in range(12):
        cv2.putText(
            img,
            f"B{i} 381210.57 1564042.96",
            (60, 100 + 70 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (20, 20, 20),
            2,
        )
    return img


def test_scan_propre_laisse_intact():
    img = page()
    out, steps = preprocess(img)
    assert steps["perspective"] is None and steps["illumination_corrected"] is False
    assert out.shape == img.shape and np.array_equal(out, img)


def test_ombre_corrigee():
    img = page().astype(np.float32)
    gradient = np.linspace(0.35, 1.0, img.shape[1])[None, :, None]  # ombre sur la gauche
    shaded = (img * gradient).astype(np.uint8)
    assert illumination_unevenness(shaded) > illumination_unevenness(page())
    flat = flatten_illumination(shaded)
    assert illumination_unevenness(flat) < illumination_unevenness(shaded) / 2
    _, steps = preprocess(shaded)
    assert steps["illumination_corrected"] is True


def test_photo_feuille_sur_table_recadree():
    sheet = cv2.resize(page(), (400, 550))
    canvas = np.full((850, 700, 3), 40, np.uint8)
    canvas[150:700, 150:550] = sheet
    quad = find_document_quad(canvas)
    assert quad is not None and abs(quad[0][0] - 150) < 8 and abs(quad[2][1] - 700) < 8
    out, steps = preprocess(canvas)
    assert steps["perspective"] is not None and abs(out.shape[0] - 550) < 10


def test_chemin_non_ascii(tmp_path):
    path = tmp_path / "titre_médina.png"
    cv2.imencode(".png", page(50, 40))[1].tofile(str(path))
    assert load_image(path).shape == (50, 40, 3)


# ------------------------------------------------- cadre imprime vs bord de la feuille


def _scan_avec_cadre():
    """Un scan : la feuille remplit l'image, et un cadre est imprime dessus."""
    img = np.full((600, 800, 3), 245, np.uint8)
    cv2.putText(img, "TITRE N", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.rectangle(img, (40, 120), (760, 560), (0, 0, 0), 3)   # le cadre du plan
    return img


def _photo_sur_fond_sombre():
    """Une photo : feuille claire posee sur un fond sombre, legerement de travers."""
    img = np.full((600, 800, 3), 35, np.uint8)
    feuille = np.array([[120, 90], [690, 70], [710, 520], [140, 540]], np.int32)
    cv2.fillConvexPoly(img, feuille, (242, 242, 242))
    cv2.putText(img, "TITRE N", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img


def test_cadre_imprime_refuse_et_image_intacte():
    img = _scan_avec_cadre()
    out, steps = preprocess(img, illumination=False)
    assert steps.get("perspective_refusee")           # le cadre n'est pas le bord
    assert out.shape == img.shape                     # rien n'a ete ampute


def test_feuille_sur_fond_sombre_bien_recadree():
    img = _photo_sur_fond_sombre()
    out, steps = preprocess(img, illumination=False)
    assert steps["perspective"] is not None
    assert "perspective_refusee" not in steps         # la correction reste possible
    assert out.shape[0] < img.shape[0]                # le fond sombre a ete retire


def test_le_juge_est_la_luminosite_du_dehors():
    from model.preprocess import _quad_borde_la_feuille, find_document_quad

    photo = _photo_sur_fond_sombre()
    assert _quad_borde_la_feuille(photo, find_document_quad(photo)) is True

    scan = _scan_avec_cadre()
    assert _quad_borde_la_feuille(scan, find_document_quad(scan)) is False
