"""Exigences de sécurité de la chaîne de décision (voir docs/dispositif-medical)."""

import numpy as np

from postprocessing import apply_threshold, decision_delay_ms, majority_vote, stream_metrics
from signal_quality import AMPLITUDE_MAX_RATIO, check_window, guard


# --- EX-01 : délai de décision ---------------------------------------------------

def test_EX01_delai_total_sous_300_ms(embedded):
    for key, e in embedded.items():
        compute = e["embarque"]["latence_pc_ms"]
        assert decision_delay_ms(3, compute) <= 300, key


def test_EX01_calcul_int8_rapide(embedded):
    import time

    m = embedded["db2_s1"]["model"]
    x = embedded["db2_s1"]["emg"][:400]
    m.predict_one(x)
    t = time.perf_counter()
    for _ in range(100):
        m.predict_one(x)
    assert (time.perf_counter() - t) / 100 * 1000 < 10


# --- EX-06 : état sûr par défaut ------------------------------------------------------

def test_EX06_confiance_insuffisante_donne_repos():
    probs = np.array([[0.1, 0.5, 0.4], [0.05, 0.9, 0.05], [0.8, 0.1, 0.1]])
    assert apply_threshold(probs, 0.6).tolist() == [0, 1, 0]


def test_EX06_signal_nul_donne_repos(embedded):
    for key, e in embedded.items():
        assert e["model"].predict_one(np.zeros((400, 12), np.float32)).argmax() == 0, key


def test_EX06_vote_causal_et_limite_au_bloc():
    pred = np.array([0, 5, 5, 5, 0, 0, 7, 7])
    block = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    out = majority_vote(pred, 3, block)
    # Modifier le futur ne change pas les décisions passées
    later = pred.copy()
    later[6:] = 2
    assert (majority_vote(later, 3, block)[:6] == out[:6]).all()
    # Le vote ne mélange pas deux blocs : 1re décision du bloc 1 = sa propre valeur
    assert out[5] == 0


def test_comptage_des_declenchements_intempestifs():
    y_true = np.zeros(40, dtype=int)
    y_pred = np.zeros(40, dtype=int)
    y_pred[[5, 6, 20]] = 4  # deux déclenchements distincts pendant le repos
    m = stream_metrics(y_true, y_pred, np.zeros(40, dtype=int))
    assert m["faux_gestes_au_repos"] == 3 / 40
    assert abs(m["declenchements_intempestifs_par_min"] - 2 / (40 * 0.05 / 60)) < 1e-9


# --- EX-07 : signal aberrant -> repos (risque R4) ----------------------------------

def faulty_windows(e):
    rest_start = np.flatnonzero(e["labels"] == 0)[500]
    rest = e["emg"][rest_start : rest_start + 400]
    saturated = rest.copy()
    saturated[:, 0] = 5e-3
    noisy = rest + np.random.default_rng(0).normal(0, 1e-3, rest.shape).astype(np.float32)
    return rest, {"électrode saturée": saturated, "bruit fort": noisy}


def test_EX07_defaut_capteur_detecte_et_force_le_repos(embedded):
    for key, e in embedded.items():
        rest, faults = faulty_windows(e)
        assert check_window(rest, e["std"])["fiable"], key
        for name, x in faults.items():
            assert not check_window(x, e["std"])["fiable"], (key, name)
            pred = e["model"].predict_one(x).argmax()
            safe, blocked = guard([pred], [x], e["std"])
            assert blocked[0] and safe[0] == 0, (key, name)


def test_EX07_peu_de_fausses_alarmes_sur_signal_reel(embedded):
    for key, e in embedded.items():
        ends = np.arange(399, len(e["emg"]), 100)
        windows = [e["emg"][i - 399 : i + 1] for i in ends]
        _, blocked = guard(np.ones(len(windows), int), windows, e["std"])
        assert blocked.mean() < 0.001, key


def test_electrode_absente_ignoree_electrode_muette_signalee():
    std = np.array([1e-5, 0.0, 1e-5])
    x = np.zeros((400, 3), np.float32)
    x[:, 0] = np.random.default_rng(0).normal(0, 1e-5, 400)
    r = check_window(x, std)
    assert r["fiable"] and r["muets"] == [2] and 1 not in r["satures"] + r["muets"]
    assert AMPLITUDE_MAX_RATIO == 20.0
