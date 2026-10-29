"""Parcelle de référence et lecteurs factices (aucun GPU nécessaire)."""

from __future__ import annotations

import math

from app.core.schema import Source
from app.extraction.reading import DocumentReading, FieldReading, PointReading, SideReading
from app.geospatial.measures import shoelace_area

# Pentagone réaliste, coordonnées imprimées avec 2 décimales (Kaolack, UTM 28N).
# La commune et le numéro de titre s'accordent à ces coordonnées : le contrôle de
# localité vérifie que la parcelle tombe bien près de la commune qu'elle déclare.
COORDS = [
    ("381210.57", "1564042.96"),
    ("381225.56", "1564034.94"),
    ("381236.10", "1564051.22"),
    ("381224.03", "1564066.85"),
    ("381208.44", "1564060.13"),
]
LABELS = [f"B{i + 1}" for i in range(len(COORDS))]
FLOATS = [(float(x), float(y)) for x, y in COORDS]
AREA_M2 = round(shoelace_area(FLOATS))
SIDES = [f"{math.dist(FLOATS[i], FLOATS[(i + 1) % 5]):.2f}" for i in range(5)]


def area_text(m2: int = AREA_M2) -> str:
    ha, rest = divmod(m2, 10_000)
    return f"{ha:02d} ha {rest // 100:02d} a {rest % 100:02d} ca"


def reading(
    source: Source, coords=COORDS, *, sides=True, fields=None, labels=LABELS
) -> DocumentReading:
    f = {
        "title_number": FieldReading("TF 1738/KL"),
        "nicad": FieldReading("0123456789012"),
        "area": FieldReading(area_text()),
    }
    if source is Source.QWEN_VL:
        f.update(
            owner=FieldReading("Awa NDIAYE", 0.95), commune=FieldReading("Kaolack", 0.9)
        )
    if fields is not None:
        f = fields
    pts = [
        PointReading(label=lab, x=x, y=y, row_index=i)
        for i, (lab, (x, y)) in enumerate(zip(labels, coords, strict=True))
    ]
    s = [SideReading(length=d, row_index=i) for i, d in enumerate(SIDES)] if sides else []
    return DocumentReading(source=source, model=source.value, fields=f, points=pts, sides=s)


class FakeReader:
    def __init__(self, result: DocumentReading | Exception) -> None:
        self.result = result
        self.source = result.source if isinstance(result, DocumentReading) else Source.QWEN_VL
        self.model_id = "fake"
        self.seen = []

    def read(self, image):
        self.seen.append(image.shape)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


# --------------------------------------------------- parcelles réelles

# Coordonnées issues de `reports/readings` : ce que PaddleOCR-VL et Qwen3-VL ont
# réellement lu sur de vrais titres. Elles remplacent le corpus annoté, qui comparait le
# système à une vérité terrain saisie à la main — les tests portent désormais sur ce que
# le système produit, et non sur ce qu'on aurait voulu qu'il lise.
#
# L'écart entre l'aire calculée (formule de Gauss) et l'aire imprimée sur le document est
# rappelé pour chacune : c'est lui qui atteste que les sommets ont été lus juste, sans
# avoir besoin d'une annotation pour arbitrer.

REELLES: dict[str, dict] = {
    "mermoz": {                     # 4 bornes, Dakar — écart 0,093 %
        "titre": "22 665/GR",
        "printed_m2": 255,
        "points": (
            ("B1", 234773.600, 1627666.100),
            ("B2", 234787.390, 1627656.730),
            ("B3", 234778.760, 1627644.030),
            ("B4", 234765.020, 1627653.480),
        ),
    },
    "grand_dakar": {                # 4 bornes — écart 0,030 %
        "titre": "TF 807/DK",
        "printed_m2": 807,
        "points": (
            ("B1", 236869.090, 1628753.880),
            ("B2", 236899.630, 1628752.080),
            ("B3", 236896.760, 1628724.890),
            ("B4", 236867.400, 1628727.500),
        ),
    },
    "thies": {                      # 8 bornes, Thiès — écart 0,033 %
        "titre": "TF 6652/TH",
        "printed_m2": 58_941,
        "points": (
            ("8.1", 294831.706, 1640903.146),
            ("8.2", 294965.286, 1640831.843),
            ("8.3", 295057.694, 1640803.611),
            ("8.4", 295253.622, 1640842.742),
            ("8.5", 295277.101, 1640725.224),
            ("8.6", 295215.415, 1640714.692),
            ("8.7", 295001.906, 1640637.808),
            ("8.8", 294782.735, 1640888.333),
        ),
    },
    "nguekhokh": {                  # 15 bornes, Nguékhokh — écart 0,044 %
        "titre": "TF 7742/MB",
        "printed_m2": 9_397,
        "points": (
            ("B1", 282331.830, 1604661.570),
            ("B2", 282416.370, 1604658.140),
            ("B3", 282417.040, 1604654.900),
            ("B4", 282405.310, 1604647.930),
            ("B5", 282392.670, 1604633.310),
            ("B6", 282362.240, 1604587.960),
            ("B7", 282354.580, 1604578.160),
            ("B8", 282340.170, 1604555.270),
            ("B9", 282314.580, 1604537.880),
            ("B10", 282300.070, 1604574.250),
            ("B11", 282280.510, 1604625.970),
            ("B12", 282268.230, 1604637.800),
            ("B13", 282285.650, 1604651.170),
            ("B14", 282297.690, 1604659.600),
            ("B15", 282316.250, 1604661.420),
        ),
    },
    "dakar": {                      # 12 bornes — écart 0,033 %
        "titre": "TF 649/DK",
        "printed_m2": 649,
        "points": (
            ("B1", 235906.620, 1624690.747),
            ("B2", 235911.823, 1624686.990),
            ("B3", 235919.165, 1624683.100),
            ("B4", 235895.652, 1624658.995),
            ("B5", 235880.994, 1624672.884),
            ("B6", 235881.571, 1624673.492),
            ("B7", 235877.383, 1624677.333),
            ("B8", 235883.969, 1624684.920),
            ("B9", 235886.148, 1624683.028),
            ("B10", 235897.914, 1624685.403),
            ("B11", 235899.680, 1624685.954),
            ("B12", 235900.745, 1624686.707),
        ),
    },
}


def sommets(nom: str) -> list[tuple[float, float]]:
    """Les coordonnées seules d'une parcelle réelle."""
    return [(x, y) for _, x, y in REELLES[nom]["points"]]


def extraction_reelle(nom: str, datum: str | None = None):
    """Une parcelle réelle sous la forme d'une `Extraction`, prête pour `parcel()`."""
    from app.core.schema import (
        DATUM_RRS04,
        Area,
        BoundaryPoint,
        Extracted,
        Extraction,
    )

    donnees = REELLES[nom]
    m2 = donnees["printed_m2"]
    return Extraction(
        document_id=nom,
        datum=datum or DATUM_RRS04,
        points=[
            BoundaryPoint(label=lab, row_index=i,
                          x=Extracted[float](value=x, confidence=1.0),
                          y=Extracted[float](value=y, confidence=1.0))
            for i, (lab, x, y) in enumerate(donnees["points"])
        ],
        printed_area=Extracted[Area](
            value=Area(ha=m2 // 10_000, a=(m2 % 10_000) // 100, ca=m2 % 100),
            confidence=1.0),
        title_number=Extracted[str](value=donnees["titre"], confidence=1.0),
    )
