"""Choix du poids de la classe repos, sur la validation uniquement.

Entraîne le CNN sur les répétitions 1, 3, 4 avec plusieurs poids du repos et
l'évalue en flux continu sur la répétition 6 (sans toucher au test).
Usage : python src/tune_rest_weight.py --subject 1
"""

import argparse
import json

import numpy as np

from data_loading import load_subject
from postprocessing import stream_metrics, tune
from preprocessing import TRAIN_REPS, filter_emg, make_stream_windows, make_windows
from train import EPOCHS_MAX, MODELS_DIR, VAL_REP, fit_cnn

# None = équilibrage strict ; 1 = le repos pèse comme un geste ; 3 = davantage
REST_FACTORS = [None, 1, 3, 9]
TAUS = np.round(np.arange(0.0, 0.96, 0.05), 2)
KS = [1, 3]  # au-delà, le délai total dépasse ~300 ms


def main(subject):
    emg, labels, reps = load_subject(subject)
    emg = filter_emg(emg)
    X, y, rep = make_windows(emg, labels, reps)
    Xs, ys, reps_s, block = make_stream_windows(emg, labels, reps)
    del emg
    n_classes = int(y.max()) + 1
    fit = np.isin(rep, [r for r in TRAIN_REPS if r != VAL_REP])
    val = rep == VAL_REP
    val_s = reps_s == VAL_REP

    out = []
    for rf in REST_FACTORS:
        model, history = fit_cnn(X[fit], y[fit], n_classes, EPOCHS_MAX,
                                 X[val], y[val], rest_factor=rf)
        probs = model.predict(Xs[val_s], verbose=0)
        raw = stream_metrics(ys[val_s], probs.argmax(axis=1), block[val_s])
        best, _ = tune(probs, ys[val_s], block[val_s], TAUS, KS)
        epochs = int(np.argmin(history.history["val_loss"])) + 1
        out.append({"rest_factor": rf, "epochs": epochs, "brut": raw, "post_traite": best})
        print(f"\nrest_factor={rf} (époques={epochs})")
        for kind, m in (("brut", raw), ("post-traité", best)):
            extra = f" seuil={m['tau']} k={m['k']}" if "tau" in m else ""
            print(f"  {kind:12s} acc. équilibrée {m['balanced_accuracy']:.3f} | "
                  f"faux gestes au repos {m['faux_gestes_au_repos']:.3f} | "
                  f"déclench./min {m['declenchements_intempestifs_par_min']:.1f}{extra}")

    with open(MODELS_DIR / f"tune_rest_s{subject}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    main(parser.parse_args().subject)
