"""Le tracé d'une parcelle, en PNG.

C'est le contrôle que l'œil fait mieux que n'importe quel seuil : une aire juste ne
garantit pas une forme juste. Deux coordonnées interverties peuvent laisser l'écart de
superficie dans la tolérance tout en donnant un terrain qui n'a rien à voir avec le plan
imprimé. On pose le tracé à côté du document, et on voit.

Trois choix rendent la comparaison possible, et aucun n'est cosmétique :

- les sommets portent leurs **libellés imprimés** (B1, B2…) et non l'ordre du polygone.
  `parcel()` réordonne les sommets pour que deux parcelles soient comparables entre
  elles : le premier point du polygone n'est pas le premier du tableau. Afficher
  « 1, 2, 3 » ferait désigner B8 par « 1 » — inutilisable pour vérifier quoi que ce soit.
- les **proportions sont respectées** : un polygone étiré par l'affichage ne se compare
  à rien.
- nord en haut, est à droite, mètres sans notation scientifique.

La fonction rend des octets. La ligne de commande les écrit dans un fichier, l'API les
sert telles quelles : un seul dessin, pas deux qui divergent.
"""

from __future__ import annotations

import io

from app.core.schema import Extraction
from app.geospatial.parcel import Parcel

TRAIT = "#1a4d7a"


class RenduIndisponible(Exception):
    """matplotlib n'est pas installé dans cet environnement."""


def libelles_imprimes(extraction: Extraction) -> dict[tuple[float, float], str]:
    """Position d'un sommet → son libellé tel qu'imprimé sur le document."""
    return {(p.x.value, p.y.value): p.label for p in extraction.points if p.complete}


def apercu_png(parcelle: Parcel,
               libelles: dict[tuple[float, float], str] | None = None,
               dpi: int = 150) -> bytes:
    """Le tracé de la parcelle en PNG. Lève `RenduIndisponible` sans matplotlib."""
    try:
        import matplotlib
    except ImportError as exc:
        raise RenduIndisponible(
            "matplotlib n'est pas installé : python -m pip install matplotlib"
        ) from exc
    matplotlib.use("Agg")                      # aucun affichage : on produit des octets
    import matplotlib.pyplot as plt

    sommets = list(parcelle.polygon.exterior.coords)
    xs, ys = zip(*sommets, strict=True)
    figure, axes = plt.subplots(figsize=(7, 7))
    try:
        axes.plot(xs, ys, "-", linewidth=1.6, color=TRAIT)
        axes.fill(xs, ys, alpha=0.10, color=TRAIT)
        axes.plot(xs[:-1], ys[:-1], "o", markersize=4, color=TRAIT)

        noms = libelles or {}
        for rang, (x, y) in enumerate(sommets[:-1], 1):
            axes.annotate(noms.get((x, y), str(rang)), (x, y),
                          textcoords="offset points", xytext=(5, 5), fontsize=8)

        ecart = ""
        if parcelle.printed_m2:
            pct = abs(parcelle.area_m2 - parcelle.printed_m2) / parcelle.printed_m2 * 100
            ecart = f" — imprimée {parcelle.printed_m2:.0f} m² (écart {pct:.2f} %)"
        titre = parcelle.title_number or parcelle.document_id
        axes.set_title(f"{titre}\n{parcelle.area_m2:.0f} m²{ecart}", fontsize=10)
        axes.set_xlabel("Est (m)")
        axes.set_ylabel("Nord (m)")
        axes.set_aspect("equal")               # sans cela, la forme est déformée
        axes.grid(True, linewidth=0.3, alpha=0.5)
        axes.ticklabel_format(style="plain", useOffset=False)
        figure.tight_layout()

        tampon = io.BytesIO()
        figure.savefig(tampon, format="png", dpi=dpi)
        return tampon.getvalue()
    finally:
        plt.close(figure)
