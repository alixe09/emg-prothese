"""Évaluation en flux continu, avec seuil de confiance et vote majoritaire.

1. Réglage de (seuil, k) sur la répétition 6, avec des modèles entraînés sur
   les répétitions 1, 3, 4 uniquement (le test n'intervient jamais dans le choix).
2. Application des réglages retenus aux modèles finaux (répétitions 1, 3, 4, 6),
   évalués en flux continu sur les répétitions 2 et 5.

Prérequis : `python src/train.py --subject N` (modèle final + nombre d'époques).
Usage : python src/evaluate_stream.py --subject 1
"""

import argparse
import json

import numpy as np
import tensorflow as tf

from data_loading import load_subject
from features import hudgins_features
from postprocessing import decision_delay_ms, stream_metrics, tune, majority_vote, apply_threshold
from preprocessing import TEST_REPS, TRAIN_REPS, filter_emg, make_stream_windows, make_windows
from train import MODELS_DIR, VAL_REP, fit_cnn, train_lda

TAUS = np.round(np.arange(0.0, 0.96, 0.05), 2)
KS = [1, 3]  # au-delà, le délai total (fenêtre + vote + calcul) dépasse ~300 ms


def main(subject):
    with open(MODELS_DIR / f"results_s{subject}.json") as f:
        results = json.load(f)

    emg, labels, reps = load_subject(subject)
    emg = filter_emg(emg)
    X, y, rep = make_windows(emg, labels, reps)
    Xs, ys, reps_s, block = make_stream_windows(emg, labels, reps)
    n_classes = int(y.max()) + 1
    del emg

    fit = np.isin(rep, [r for r in TRAIN_REPS if r != VAL_REP])
    train = np.isin(rep, TRAIN_REPS)
    val_s, test_s = reps_s == VAL_REP, np.isin(reps_s, TEST_REPS)

    # --- Modèles « de réglage » (sans la répétition 6) et leurs probabilités sur le flux de validation
    lda_fit = train_lda(X[fit], y[fit])
    cnn_fit, _ = fit_cnn(X[fit], y[fit], n_classes, results["cnn"]["epochs"])
    probs_val = {
        "lda": lda_fit.predict_proba(hudgins_features(Xs[val_s])),
        "cnn": cnn_fit.predict(Xs[val_s], verbose=0),
    }

    # --- Modèles finaux, probabilités sur le flux de test
    lda = train_lda(X[train], y[train])
    cnn = tf.keras.models.load_model(MODELS_DIR / f"cnn_s{subject}.keras")
    probs_test = {
        "lda": lda.predict_proba(hudgins_features(Xs[test_s])),
        "cnn": cnn.predict(Xs[test_s], verbose=0),
    }

    out = {"subject": subject}
    for name in ("lda", "cnn"):
        best, grid = tune(probs_val[name], ys[val_s], block[val_s], TAUS, KS)
        raw = stream_metrics(ys[test_s], probs_test[name].argmax(axis=1), block[test_s])
        pred = majority_vote(apply_threshold(probs_test[name], best["tau"]), best["k"], block[test_s])
        post = stream_metrics(ys[test_s], pred, block[test_s])
        compute = results[name]["latence_ms"]
        out[name] = {
            "reglage": {"tau": best["tau"], "k": best["k"]},
            "brut": {**raw, "delai_ms": decision_delay_ms(1, compute)},
            "post_traite": {**post, "delai_ms": decision_delay_ms(best["k"], compute)},
            "grille_validation": grid,
        }
        print(f"\n{name.upper()} — réglage retenu sur validation : seuil={best['tau']}, k={best['k']}")
        for kind in ("brut", "post_traite"):
            m = out[name][kind]
            print(f"  {kind:12s} acc. équilibrée {m['balanced_accuracy']:.3f} | "
                  f"rappel gestes {m['rappel_gestes']:.3f} | "
                  f"faux gestes au repos {m['faux_gestes_au_repos']:.3f} | "
                  f"déclench. intempestifs/min {m['declenchements_intempestifs_par_min']:.1f} | "
                  f"délai {m['delai_ms']:.0f} ms")

    with open(MODELS_DIR / f"stream_s{subject}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    main(parser.parse_args().subject)
