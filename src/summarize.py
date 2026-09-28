"""Tableau récapitulatif LDA / CNN sur tous les sujets évalués en flux continu.

Usage : python src/summarize.py [--db db3]
"""

import argparse
import json

import numpy as np

from train import DB, models_dir

METRICS = [
    ("balanced_accuracy", "Acc. équilibrée", 100, "%"),
    ("rappel_gestes", "Gestes reconnus", 100, "%"),
    ("faux_gestes_au_repos", "Faux gestes au repos", 100, "%"),
    ("declenchements_intempestifs_par_min", "Déclench. intempestifs/min", 1, ""),
    ("delai_ms", "Délai (ms)", 1, ""),
]


def main(db=DB):
    out_dir = models_dir(db)
    files = sorted(out_dir.glob("stream_s*.json"), key=lambda p: int(p.stem[8:]))
    runs = [json.loads(p.read_text()) for p in files]
    subjects = [r["subject"] for r in runs]
    print(f"Sujets : {subjects}  (test en flux continu, répétitions 2 et 5, avec post-traitement)\n")

    header = f"{'':28s}" + "".join(f"{'S' + str(s):>8s}" for s in subjects) + f"{'moyenne ± écart-type':>24s}"
    summary = {}
    for model in ("lda", "cnn"):
        print(f"{model.upper()}\n{header}")
        summary[model] = {}
        for key, name, scale, unit in METRICS:
            vals = np.array([r[model]["post_traite"][key] * scale for r in runs])
            summary[model][key] = {"moyenne": float(vals.mean()), "ecart_type": float(vals.std())}
            print(f"{name:28s}" + "".join(f"{v:8.1f}" for v in vals)
                  + f"{vals.mean():>14.1f} ± {vals.std():.1f} {unit}")
        print()

    diff = np.array([
        r["cnn"]["post_traite"]["balanced_accuracy"] - r["lda"]["post_traite"]["balanced_accuracy"]
        for r in runs
    ]) * 100
    # Rapport non défini si le CNN ne déclenche jamais (ex. sujet sans signal exploitable)
    ratio = np.array([
        r["lda"]["post_traite"]["declenchements_intempestifs_par_min"]
        / r["cnn"]["post_traite"]["declenchements_intempestifs_par_min"]
        if r["cnn"]["post_traite"]["declenchements_intempestifs_par_min"] > 0 else np.nan
        for r in runs
    ])
    print(f"CNN - LDA, acc. équilibrée : {', '.join(f'{d:+.1f}' for d in diff)} points "
          f"(CNN meilleur sur {(diff > 0).sum()}/{len(diff)} sujets)")
    print(f"Déclenchements intempestifs, LDA / CNN : {', '.join('—' if np.isnan(x) else f'x{x:.1f}' for x in ratio)}")

    summary["subjects"] = subjects
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DB)
    main(parser.parse_args().db)
