"""Modèle embarqué int8 : taille et non-régression des performances (EX-04, EX-05, R6)."""

import numpy as np

from conftest import APP
from postprocessing import majority_vote, stream_metrics
from signal_quality import guard

# Valeurs de référence mesurées à la publication (extrait de démo, répétition 2) ;
# une modification du code ou du modèle qui les dégrade fait échouer le test.
REFERENCE = {"db2_s1": 0.59, "db3_s8": 0.47}
TOLERANCE = 0.03


def test_EX04_modele_tient_dans_un_microcontroleur():
    for path in (APP / "model").glob("*_int8.tflite"):
        assert path.stat().st_size <= 256 * 1024, path.name


def decode(e):
    labels, emg = e["labels"], e["emg"]
    block = np.r_[0, np.cumsum((labels[1:] == 0) & (labels[:-1] > 0))]
    ends, blocks = [], []
    for b in np.unique(block):
        idx = np.flatnonzero(block == b)
        ends.append(np.arange(idx[0] + 399, idx[-1] + 1, 100))
        blocks.append(np.full(len(ends[-1]), b))
    ends, blocks = np.concatenate(ends), np.concatenate(blocks)
    windows = [emg[i - 399 : i + 1] for i in ends]
    pred = np.array([e["model"].predict_one(w).argmax() for w in windows])
    pred, _ = guard(pred, windows, e["std"])
    return labels[ends], majority_vote(pred, 3, blocks), blocks


def test_R6_non_regression_du_modele_int8(embedded):
    for key, e in embedded.items():
        y, pred, blocks = decode(e)
        acc = stream_metrics(y, pred, blocks)["balanced_accuracy"]
        assert acc >= REFERENCE[key] - TOLERANCE, (key, acc)
