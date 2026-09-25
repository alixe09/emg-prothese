"""Filtrage et découpage en fenêtres glissantes.

Une prothèse décide en continu à partir des dernières millisecondes de signal :
on reproduit ce fonctionnement avec des fenêtres de 200 ms avancées de 50 ms.
Le délai de décision reste ainsi sous le seuil d'environ 300 ms au-delà duquel
le porteur perçoit un retard.
"""

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt

from data_loading import FS

WINDOW_MS = 200
STEP_MS = 50
TRAIN_REPS = (1, 3, 4, 6)
TEST_REPS = (2, 5)


def filter_emg(emg, fs=FS):
    """Passe-bande 20–450 Hz (bande utile de l'EMG) + coupe-bande 50 Hz (secteur)."""
    sos = butter(4, [20, 450], btype="bandpass", fs=fs, output="sos")
    emg = sosfiltfilt(sos, emg, axis=0)
    b, a = iirnotch(50, Q=30, fs=fs)
    return filtfilt(b, a, emg, axis=0).astype(np.float32)


def make_windows(emg, labels, repetitions, fs=FS, window_ms=WINDOW_MS, step_ms=STEP_MS):
    """Découpe le signal en fenêtres.

    Une fenêtre n'est gardée que si toute sa durée porte le même label et la même
    répétition, pour éviter les transitions ambiguës entre deux gestes.
    Renvoie X (n_fenêtres, taille_fenêtre, n_canaux), y et rep.
    """
    size = int(fs * window_ms / 1000)
    step = int(fs * step_ms / 1000)
    starts = np.arange(0, len(emg) - size + 1, step)

    X, y, rep = [], [], []
    for s in starts:
        lab = labels[s : s + size]
        r = repetitions[s : s + size]
        if lab[0] == lab[-1] and (lab == lab[0]).all() and (r == r[0]).all():
            X.append(emg[s : s + size])
            y.append(lab[0])
            rep.append(r[0])
    return np.stack(X), np.array(y), np.array(rep)


def make_stream_windows(emg, labels, repetitions, fs=FS, window_ms=WINDOW_MS, step_ms=STEP_MS):
    """Toutes les fenêtres, dans l'ordre, comme les verrait la prothèse en continu.

    Contrairement à `make_windows`, les transitions entre gestes sont gardées.
    Le label d'une fenêtre est celui de son dernier échantillon (l'intention
    actuelle de l'utilisateur). `block` identifie les séquences contiguës d'une
    même répétition (repos + geste), pour que les post-traitements temporels ne
    mélangent pas deux moments éloignés de l'enregistrement.
    """
    size = int(fs * window_ms / 1000)
    step = int(fs * step_ms / 1000)
    ends = np.arange(size - 1, len(emg), step)
    X = np.stack([emg[e - size + 1 : e + 1] for e in ends])
    rep = repetitions[ends]
    block = np.r_[0, np.cumsum(rep[1:] != rep[:-1])]
    return X, labels[ends], rep, block


def split_by_repetition(X, y, rep, train_reps=TRAIN_REPS, test_reps=TEST_REPS):
    """Split standard Ninapro : on teste sur des répétitions jamais vues."""
    train = np.isin(rep, train_reps)
    test = np.isin(rep, test_reps)
    return X[train], y[train], X[test], y[test]
