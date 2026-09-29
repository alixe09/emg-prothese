"""Génère la matrice de traçabilité exigences -> vérification -> résultat.

Les chiffres sont lus dans les résultats enregistrés (models/, models/db3/) et
les exigences vérifiées par test sont évaluées en lançant la suite pytest :
rien n'est recopié à la main dans le dossier.

Usage : python src/verification_report.py
Sortie : docs/dispositif-medical/03-matrice-tracabilite.md
"""

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUT = ROOT / "docs" / "dispositif-medical" / "03-matrice-tracabilite.md"


def load_stream(db_dir):
    files = sorted(db_dir.glob("stream_s*.json"), key=lambda p: int(p.stem[8:]))
    return [json.loads(p.read_text()) for p in files]


def stat(runs, model, key, scale=100):
    v = np.array([r[model]["post_traite"][key] for r in runs]) * scale
    return v.mean(), v.max(), v.min()


def run_tests():
    """Lance pytest et renvoie {nom_du_test: réussi}."""
    with tempfile.TemporaryDirectory() as d:
        xml = Path(d) / "junit.xml"
        subprocess.run([sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q",
                        "-p", "no:warnings", f"--junitxml={xml}"], capture_output=True)
        tree = ET.parse(xml)
    return {c.get("name"): c.find("failure") is None and c.find("error") is None
            for c in tree.iter("testcase")}


def tests_status(tests, prefix):
    names = [n for n in tests if n.startswith(prefix)]
    ok = all(tests[n] for n in names) and names
    return ok, names


def status(ok):
    return "✅ conforme" if ok else "❌ non conforme"


def main():
    db2, db3 = load_stream(MODELS), load_stream(MODELS / "db3")
    tests = run_tests()
    tflite = json.loads((MODELS / "tflite_results.json").read_text())
    transfer2 = {(e["methode"], e["repetitions_calibration"]): e
                 for e in json.loads((MODELS / "transfer_summary.json").read_text())}
    transfer3 = {(e["methode"], e["repetitions_calibration"]): e
                 for e in json.loads((MODELS / "db3" / "transfer_summary.json").read_text())}
    rows = []

    # EX-01 délai
    d2 = max(r[m]["post_traite"]["delai_ms"] for r in db2 for m in ("lda", "cnn"))
    d3 = max(r[m]["post_traite"]["delai_ms"] for r in db3 for m in ("lda", "cnn"))
    t_ok, t_names = tests_status(tests, "test_EX01")
    rows.append(("EX-01", "Délai total de décision (fenêtre + vote + calcul)", "≤ 300 ms",
                 "R3", f"Mesure sur flux continu ; tests `{'`, `'.join(t_names)}`",
                 f"max {d2:.0f} ms", f"max {d3:.0f} ms", status(d2 <= 300 and d3 <= 300 and t_ok)))

    # EX-02 faux gestes au repos (CNN, intra-sujet, seuil réglé sur validation)
    m2, x2, _ = stat(db2, "cnn", "faux_gestes_au_repos")
    m3, x3, _ = stat(db3, "cnn", "faux_gestes_au_repos")
    rows.append(("EX-02", "Faux gestes pendant le repos (CNN)", "moyenne ≤ 5 %, chaque sujet ≤ 10 %",
                 "R1", "Flux continu, répétitions de test 2 et 5 (`src/evaluate_stream.py`)",
                 f"moy. {m2:.1f} %, max {x2:.1f} %", f"moy. {m3:.1f} %, max {x3:.1f} %",
                 f"DB2 {status(m2 <= 5 and x2 <= 10)} · DB3 {status(m3 <= 5 and x3 <= 10)}"))

    # EX-03 précision
    a2, _, n2 = stat(db2, "cnn", "balanced_accuracy")
    a3, _, n3 = stat(db3, "cnn", "balanced_accuracy")
    rows.append(("EX-03", "Précision équilibrée, 17 gestes + repos (CNN)", "moyenne ≥ 50 %",
                 "R2", "Flux continu, répétitions de test 2 et 5",
                 f"moy. {a2:.1f} % (min {n2:.1f} %)", f"moy. {a3:.1f} % (min {n3:.1f} %)",
                 f"DB2 {status(a2 >= 50)} · DB3 {status(a3 >= 50)}"))

    # EX-04 taille
    size = max(p.stat().st_size for p in (ROOT / "app" / "model").glob("*_int8.tflite")) / 1024
    t_ok, t_names = tests_status(tests, "test_EX04")
    rows.append(("EX-04", "Taille du modèle embarqué (int8)", "≤ 256 Ko", "—",
                 f"Test `{t_names[0]}`", f"{size:.0f} Ko", f"{size:.0f} Ko", status(size <= 256 and t_ok)))

    # EX-05 quantification
    diffs = np.array([(v["float32"]["flux"]["balanced_accuracy"] - v["int8"]["flux"]["balanced_accuracy"]) * 100
                      for v in tflite.values()])
    t_ok, t_names = tests_status(tests, "test_R6")
    rows.append(("EX-05", "Perte de précision due à la quantification int8",
                 "moyenne ≤ 2 points, chaque sujet ≤ 5 points", "R6",
                 f"`src/export_tflite.py` (5 sujets DB2) ; test de non-régression `{t_names[0]}`",
                 f"moy. {diffs.mean():.1f} pt, max {diffs.max():.1f} pt", "sujet 8 : pas de perte",
                 status(diffs.mean() <= 2 and diffs.max() <= 5 and t_ok)))

    # EX-06 état sûr
    t_ok, t_names = tests_status(tests, "test_EX06")
    rows.append(("EX-06", "État sûr : sans signal ou sans décision fiable, la prothèse reste au repos",
                 "décision = repos", "R1, R4", "Tests " + ", ".join(f"`{n}`" for n in t_names),
                 "vérifié par test", "vérifié par test", status(t_ok)))

    # EX-07 signal aberrant
    t_ok, t_names = tests_status(tests, "test_EX07")
    rows.append(("EX-07", "Électrode saturée ou bruit fort : pas de geste déclenché",
                 "décision = repos, fausses alarmes < 0,1 %", "R4",
                 "Tests " + ", ".join(f"`{n}`" for n in t_names) + " (`src/signal_quality.py`)",
                 "vérifié par test", "vérifié par test", status(t_ok)))

    # EX-08 calibration nouveau porteur
    c2 = transfer2[("cnn_pre_ft", 4)]["balanced_accuracy"]["moyenne"]
    l2 = transfer2[("lda", 4)]["balanced_accuracy"]["moyenne"]
    c3 = transfer3[("cnn_pre_ft", 4)]["balanced_accuracy"]["moyenne"]
    l3 = transfer3[("lda", 4)]["balanced_accuracy"]["moyenne"]
    best2, best3 = max(c2, l2), max(c3, l3)
    rows.append(("EX-08", "Nouveau porteur : ≤ 10 min de calibration pour atteindre EX-03",
                 "≥ 50 % avec 4 répétitions (~9 min)", "R7", "Leave-one-subject-out (`src/transfer.py`)",
                 f"CNN pré-entraîné {c2:.1f} %, LDA {l2:.1f} %", f"CNN pré-entraîné {c3:.1f} %, LDA {l3:.1f} %",
                 f"DB2 {status(best2 >= 50)} · DB3 {status(best3 >= 50)}"))

    n_tests, n_ok = len(tests), sum(tests.values())
    lines = [
        "# 3. Matrice de traçabilité exigences → vérification",
        "",
        f"*Générée automatiquement par `python src/verification_report.py` le {date.today():%d/%m/%Y} "
        f"à partir des résultats enregistrés et de la suite de tests ({n_ok}/{n_tests} tests réussis). "
        "Ne pas modifier à la main.*",
        "",
        "Les critères d'acceptation sont **fixés par l'auteur à titre d'exercice** (ils ne proviennent "
        "pas d'une norme). DB2 = 5 sujets valides, DB3 = 11 sujets amputés.",
        "",
        "| ID | Exigence | Critère | Risques | Vérification | Valides (DB2) | Amputés (DB3) | Statut |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    lines += [
        "",
        "## Lecture",
        "",
        "- Les exigences **de sécurité du logiciel** (délai, état sûr, défaut capteur, "
        "embarquabilité, quantification) sont vérifiées.",
        "- Les exigences **de performance clinique** (EX-02, EX-03, EX-08) sont atteintes sur "
        "sujets valides mais **pas chez les amputés** : en l'état, le logiciel ne serait pas "
        "acceptable pour la population visée. C'est le principal écart ouvert (voir l'analyse "
        "des risques, R1, R2 et R7).",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)} : {n_ok}/{n_tests} tests réussis")
    for r in rows:
        print(f"  {r[0]} {r[-1]}")


if __name__ == "__main__":
    main()
