"""Ce que chaque modele a lu, et ce que la fusion a retenu. Aucune verite terrain ici.

    python tools/trace.py reports/readings/tf.json
    python tools/trace.py reports/readings --tableau      # que le tableau de coordonnees
    python tools/trace.py reports/readings/tf.json --brut # sortie brute des modeles

L'outil ne juge pas : il montre. Les lectures brutes sont a cote de l'extraction, dans un
fichier « <document>_readings.json » ecrit par `python -m model ... --readings`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.schema import TEXT_FIELDS                                       # noqa: E402

# Les lecteurs nomment la superficie « area », le schema « printed_area ».
CHAMPS = (*TEXT_FIELDS, "printed_area")
ALIAS = {"printed_area": "area"}
LARGEUR = 30


def _coupe(v: object, n: int = LARGEUR) -> str:
    s = "-" if v is None else str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def _aire(v: object) -> str:
    if not isinstance(v, dict):
        return _coupe(v)
    ha, a, ca = v.get("ha", 0), v.get("a", 0), v.get("ca", 0)
    return f"{ha}ha {a}a {ca}ca = {ha * 10000 + a * 100 + ca} m²"


def _texte_retenu(bloc: object) -> str:
    """Ce qui a ete retenu, tel que lu : la chaine, pas le flottant qui en derive."""
    if not isinstance(bloc, dict):
        return "-"
    brut = bloc.get("raw_text")
    if brut is not None:
        return str(brut)
    v = bloc.get("value")
    return "-" if v is None else str(v)


def _marque(gauche: object, droite: object) -> str:
    if gauche is None and droite is None:
        return " "
    if gauche is None or droite is None:
        return "·"                      # un seul lecteur a vu quelque chose
    return "=" if str(gauche) == str(droite) else "≠"


def champs(ex: dict, lectures: dict) -> int:
    p = lectures.get("paddle_vl", {}).get("fields", {})
    q = lectures.get("qwen_vl", {}).get("fields", {})
    print(f"\n  {'champ':<18} {'':1} {'paddle':<{LARGEUR}} {'qwen':<{LARGEUR}} "
          f"{'retenu':<{LARGEUR}} source")
    print("  " + "-" * (20 + 3 * LARGEUR + 12))
    divergences = 0
    for nom in CHAMPS:
        cle = ALIAS.get(nom, nom)
        pv = p.get(cle, {}).get("raw")
        qv = q.get(cle, {}).get("raw")
        bloc = ex.get(nom) or {}
        valeur = _aire(bloc.get("value")) if nom == "printed_area" else bloc.get("value")
        m = _marque(pv, qv)
        divergences += m == "≠"
        if pv is None and qv is None and valeur is None:
            continue                     # rien a montrer sur ce champ
        src = bloc.get("source", "-")
        conf = bloc.get("confidence")
        conf = f" {conf:.2f}" if isinstance(conf, (int, float)) and conf else ""
        print(f"  {nom:<18} {m} {_coupe(pv):<{LARGEUR}} {_coupe(qv):<{LARGEUR}} "
              f"{_coupe(valeur):<{LARGEUR}} {src}{conf}")
    return divergences


def tableau(ex: dict, lectures: dict) -> int:
    p = {x["row_index"]: x for x in lectures.get("paddle_vl", {}).get("points", [])}
    q = {x["row_index"]: x for x in lectures.get("qwen_vl", {}).get("points", [])}
    pts = ex.get("points", [])
    lignes = sorted(set(p) | set(q) | {i for i, _ in enumerate(pts)})
    if not lignes:
        print("\n  aucun point lu")
        return 0
    print(f"\n  {'ligne':<6} {'':1} {'paddle':<26} {'qwen':<26} {'retenu':<26} conf")
    print("  " + "-" * 96)
    divergences = 0
    for i in lignes:
        a, b = p.get(i, {}), q.get(i, {})
        pxy = f"{a.get('x') or '-'} / {a.get('y') or '-'}"
        qxy = f"{b.get('x') or '-'} / {b.get('y') or '-'}"
        pt = pts[i] if i < len(pts) else {}
        # `value` est un flottant : il perd les zeros finaux, qui font partie de ce qui
        # est imprime. `raw_text` garde la chaine retenue telle que lue.
        rx = _texte_retenu(pt.get("x"))
        ry = _texte_retenu(pt.get("y"))
        rxy = f"{rx} / {ry}"
        m = " "
        if a or b:
            mx = _marque(a.get("x"), b.get("x"))
            my = _marque(a.get("y"), b.get("y"))
            m = "≠" if "≠" in (mx, my) else ("·" if "·" in (mx, my) else "=")
        divergences += m == "≠"
        conf = min(((pt.get(c) or {}).get("confidence") or 1.0) for c in ("x", "y")) if pt else 0.0
        label = pt.get("label") or a.get("label") or b.get("label") or f"#{i}"
        print(f"  {label:<6} {m} {pxy:<26} {qxy:<26} {rxy:<26} {conf:.2f}")
    return divergences


def _est_extraction(chemin: Path) -> bool:
    """Ecarte les annotations et les fichiers de lectures brutes deposes la par erreur."""
    if chemin.stem.endswith("_readings"):
        return False
    try:
        d = json.loads(chemin.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(d, dict) and "document_id" in d and "annotation_version" not in d


def resume(chemin: Path) -> str:
    ex = json.loads(chemin.read_text(encoding="utf-8"))
    brut = chemin.with_name(f"{chemin.stem}_readings.json")
    lectures = json.loads(brut.read_text(encoding="utf-8")) if brut.exists() else {}
    p = lectures.get("paddle_vl", {})
    q = lectures.get("qwen_vl", {})
    npad = sum(1 for v in p.get("fields", {}).values() if v.get("raw") is not None)
    nqwen = sum(1 for v in q.get("fields", {}).values() if v.get("raw") is not None)
    retenus = sum(1 for n in CHAMPS if (ex.get(n) or {}).get("value") is not None)
    err = sum(1 for i in ex.get("issues", []) if i["severity"] == "error")
    warn = sum(1 for i in ex.get("issues", []) if i["severity"] == "warning")
    return (f"  {chemin.stem:<34} "
            f"{npad:>6} {nqwen:>5} {retenus:>7} {len(p.get('points', [])):>4}"
            f"{len(q.get('points', [])):>5} {len(ex.get('points', [])):>7} {err:>4}{warn:>5}")


def montre(chemin: Path, args) -> None:
    ex = json.loads(chemin.read_text(encoding="utf-8"))
    brut = chemin.with_name(f"{chemin.stem}_readings.json")
    lectures = json.loads(brut.read_text(encoding="utf-8")) if brut.exists() else {}

    print(f"\n{'=' * 100}\n{ex.get('document_id', chemin.stem)}")
    if not lectures:
        print(f"  (pas de lectures brutes : {brut.name} absent — relancer avec --readings)")
    for cle, r in lectures.items():
        o = r.get("orientation_deg")
        print(f"  {cle:<12} {r.get('model', '?')}"
              + (f"  (image pivotee de {o}°)" if o else ""))

    d = 0
    if not args.tableau:
        d += champs(ex, lectures)
    if not args.champs:
        d += tableau(ex, lectures)

    issues = ex.get("issues", [])
    if issues:
        print(f"\n  {len(issues)} signalement(s) :")
        for i in issues:
            print(f"    [{i['severity']:<7}] {i['code']:<24} {i['message']}")

    print(f"\n  {d} divergence(s) entre les deux lecteurs")

    if args.brut:
        for cle, r in lectures.items():
            print(f"\n--- sortie brute {cle} " + "-" * 60)
            print(json.dumps(r.get("raw_output"), ensure_ascii=False, indent=2)[: args.brut])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path, help="Une extraction JSON, ou un dossier.")
    ap.add_argument("--champs", action="store_true", help="Que les champs texte.")
    ap.add_argument("--tableau", action="store_true", help="Que le tableau de coordonnees.")
    ap.add_argument("--brut", nargs="?", type=int, const=4000, default=0,
                    metavar="N", help="Sortie brute des modeles, N caracteres (defaut 4000).")
    ap.add_argument("--resume", action="store_true", help="Une ligne par document.")
    args = ap.parse_args()

    if args.source.is_dir():
        chemins = [p for p in sorted(args.source.glob("*.json")) if _est_extraction(p)]
    elif args.source.exists():
        chemins = [args.source]
    else:
        # Un nom tronque ou mal recopie : on montre ce qui existe de proche plutot que
        # de renvoyer une trace d'erreur.
        dossier = args.source.parent
        proches = [p.name for p in sorted(dossier.glob("*.json"))
                   if _est_extraction(p) and p.stem.startswith(args.source.stem[:8])]
        print(f"{args.source} n'existe pas.", file=sys.stderr)
        if proches:
            print("Vouliez-vous :", file=sys.stderr)
            for n in proches:
                print(f"    {dossier / n}", file=sys.stderr)
        return 2
    if not chemins:
        print(f"Aucune extraction dans {args.source}", file=sys.stderr)
        return 2

    if args.resume:
        print(f"\n  {'fichier':<34} {'paddle':>6} {'qwen':>5} {'retenu':>7} "
              f"{'pts p':>4}{'pts q':>5} {'retenu':>7} {'err':>4}{'warn':>5}")
        print("  " + "-" * 86)
        for c in chemins:
            try:
                print(resume(c))
            except Exception as e:
                print(f"  {c.stem:<34} illisible — {type(e).__name__}")
        print("\n  champs lus par chaque lecteur, points lus, et signalements")
        return 0

    for c in chemins:
        try:
            montre(c, args)
        except Exception as e:
            print(f"\n{c.stem} : illisible — {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
