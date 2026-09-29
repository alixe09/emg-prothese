"""Prétraitement, fenêtrage et features : le signal arrive correct au modèle."""

import numpy as np

from data_loading import FS
from features import hudgins_features
from preprocessing import filter_emg, make_stream_windows, make_windows


def amplitude_at(x, f):
    spectrum = np.abs(np.fft.rfft(x)) / len(x) * 2
    freqs = np.fft.rfftfreq(len(x), 1 / FS)
    return spectrum[np.argmin(np.abs(freqs - f))]


def test_filtre_supprime_continu_et_secteur_garde_bande_utile():
    t = np.arange(4 * FS) / FS
    x = 1.0 + np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)
    y = filter_emg(np.tile(x[:, None], (1, 2)).astype(np.float32))[FS:-FS, 0]
    assert abs(y.mean()) < 0.01                       # composante continue supprimée
    assert amplitude_at(y, 50) < 0.05                 # secteur 50 Hz atténué > 26 dB
    assert 0.8 < amplitude_at(y, 120) < 1.1           # bande utile conservée


def test_fenetres_d_entrainement_sans_transition():
    labels = np.r_[np.zeros(1000), np.ones(1000), np.zeros(1000)].astype(int)
    reps = np.ones(3000, dtype=int)
    index = np.tile(np.arange(3000.0)[:, None], (1, 2))  # le signal vaut son indice
    X, y, _ = make_windows(index, labels, reps)
    assert X.shape[1:] == (400, 2)
    # Aucune fenêtre ne chevauche un changement de geste
    for window, lab in zip(X, y):
        start = int(window[0, 0])
        assert (labels[start : start + 400] == lab).all()


def test_flux_continu_label_de_la_fin_de_fenetre_et_blocs():
    labels = np.r_[np.zeros(600), np.full(600, 3)].astype(int)
    reps = np.r_[np.full(1200, 2)]
    reps[900:] = 5
    X, y, rep, block = make_stream_windows(np.zeros((1200, 1)), labels, reps)
    ends = np.arange(399, 1200, 100)
    assert (y == labels[ends]).all()                  # intention actuelle = dernier échantillon
    assert (np.diff(block)[rep[1:] != rep[:-1]] == 1).all()


def test_features_hudgins_sur_signal_connu():
    t = np.arange(400) / FS
    sine = np.sin(2 * np.pi * 50 * t)[None, :, None]  # 10 périodes sur 200 ms
    mav, rms, wl, zc, ssc = hudgins_features(sine)[0]
    assert abs(mav - 2 / np.pi) < 0.01
    assert abs(rms - 1 / np.sqrt(2)) < 0.01
    assert 19 <= zc <= 20 and 19 <= ssc <= 20
