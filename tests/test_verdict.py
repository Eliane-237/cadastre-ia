"""Le verdict traduit les contrôles en une décision. Il n'en ajoute aucun."""


from app.core.schema import DATUM_INCONNU, DATUM_RRS04, Extracted, Extraction, Issue, Severity
from app.validation.verdict import Decision, juger
from tests.fake_readers import COORDS


def _extraction(datum=DATUM_RRS04, issues=(), systeme="RRS04", confiance=1.0):
    from app.core.schema import BoundaryPoint

    pts = [
        BoundaryPoint(
            label=f"B{i+1}", row_index=i,
            x=Extracted[float](value=float(x), confidence=confiance, raw_text=x),
            y=Extracted[float](value=float(y), confidence=confiance, raw_text=y),
        )
        for i, (x, y) in enumerate(COORDS)
    ]
    from app.core.schema import Area

    # Les champs cœur doivent être présents : absents, ils partent d'office en revue,
    # et le jeu d'essai ne testerait plus ce qu'il prétend tester.
    return Extraction(
        document_id="D", datum=datum, points=pts, issues=list(issues),
        title_number=Extracted[str](value="TF 1738/KL", confidence=confiance),
        nicad=Extracted[str](value="0123456789012", confidence=confiance),
        commune=Extracted[str](value="Kaolack", confidence=confiance),
        owner=Extracted[str](value="Awa NDIAYE", confidence=confiance),
        printed_area=Extracted[Area](value=Area(a=2, ca=21), confidence=confiance),
        coordinate_system=Extracted[str](value=systeme, confidence=confiance)
        if systeme else Extracted[str](),
    )


def test_tout_va_bien():
    j = juger(_extraction())
    assert j.decision is Decision.EXPLOITABLE and j.entre_en_base and j.raisons == []


def test_une_erreur_refuse():
    j = juger(_extraction(issues=[Issue(code="AREA_MISMATCH", severity=Severity.ERROR,
                                        message="écart de 95 %")]))
    assert j.decision is Decision.REFUSE and not j.entre_en_base
    assert "AREA_MISMATCH" in str(j)


def test_sans_sommet_refuse():
    ex = _extraction()
    ex.points = []
    assert juger(ex).decision is Decision.REFUSE


def test_datum_inconnu_envoie_en_revue_mais_ne_refuse_pas():
    """La parcelle existe ; c'est la confrontation qui n'a pas de sens."""
    j = juger(_extraction(datum=DATUM_INCONNU, systeme="UTM zone 28N"))
    assert j.decision is Decision.A_RELIRE
    assert "datum inconnu" in str(j) and "UTM zone 28N" in str(j)


def test_aucun_systeme_imprime_est_dit_comme_tel():
    j = juger(_extraction(datum=DATUM_INCONNU, systeme=None))
    assert "aucun système de coordonnées imprimé" in str(j)


def test_une_alerte_envoie_en_revue():
    j = juger(_extraction(issues=[Issue(code="TITRE_EGAL_MERE", severity=Severity.WARNING,
                                        message="même numéro")]))
    assert j.decision is Decision.A_RELIRE and "TITRE_EGAL_MERE" in str(j)


def test_une_information_ne_change_rien():
    j = juger(_extraction(issues=[Issue(code="READING_CONFLICT_RESOLVED",
                                        severity=Severity.INFO, message="tranché")]))
    assert j.decision is Decision.EXPLOITABLE


def test_confiance_basse_envoie_en_revue():
    j = juger(_extraction(confiance=0.2))
    assert j.decision is Decision.A_RELIRE and "à relire" in str(j)


def test_le_refus_prime_sur_le_doute():
    j = juger(_extraction(
        datum=DATUM_INCONNU, confiance=0.2,
        issues=[Issue(code="POLYGON_INVALID", severity=Severity.ERROR, message="se recoupe"),
                Issue(code="COLUMN_DECIMALS", severity=Severity.WARNING, message="B4")],
    ))
    assert j.decision is Decision.REFUSE
    assert j.raisons == ["anomalies bloquantes : POLYGON_INVALID"]   # une seule raison : la bonne
