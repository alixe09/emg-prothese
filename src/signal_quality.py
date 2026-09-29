"""Contrôle de qualité du signal avant décision (mesure de réduction du risque R4).

Constat (analyse de risques) : face à un signal aberrant — électrode saturée,
bruit fort d'un câble ou d'une interférence — le CNN déclenche un geste avec une
confiance élevée (jusqu'à 99 %) : le seuil de confiance ne protège pas.

Règle : si l'amplitude (RMS) d'une électrode sur la fenêtre dépasse
AMPLITUDE_MAX_RATIO fois son écart-type mesuré à la calibration, la fenêtre est
jugée non fiable et la prothèse reste au repos (état sûr). Seuil réglé sur les
données réelles des 16 sujets (tests, 5 valides + 11 amputés) : le maximum
observé en usage normal est de 32 fois, le 99,99e centile de 17 fois ; à 20,
0,007 % des fenêtres normales sont bloquées (fausse alarme sans danger : repos).

Une électrode quasi muette (décollée) est seulement signalée : le modèle y
répond déjà par le repos (vérifié), mais le porteur doit être alerté.
"""

import numpy as np

AMPLITUDE_MAX_RATIO = 20.0
FLAT_RATIO = 0.02
ABSENT_STD = 1e-7  # électrode absente dès la calibration (canal nul, amputés S6/S7)


def check_window(x, calib_std):
    """x : (taille_fenêtre, n_canaux) ; calib_std : (n_canaux,).

    Renvoie un dict : `fiable` (bool), `satures` et `muets` (indices d'électrodes).
    """
    present = calib_std > ABSENT_STD
    ratio = np.zeros_like(calib_std)
    rms = np.sqrt(np.mean(np.square(x, dtype=np.float64), axis=0))
    ratio[present] = rms[present] / calib_std[present]
    satures = np.flatnonzero(present & (ratio > AMPLITUDE_MAX_RATIO))
    muets = np.flatnonzero(present & (ratio < FLAT_RATIO))
    return {"fiable": satures.size == 0, "satures": satures.tolist(), "muets": muets.tolist()}


def guard(pred, windows, calib_std, rest_label=0):
    """Force le repos sur les fenêtres jugées non fiables.

    Renvoie (décisions corrigées, masque des fenêtres bloquées).
    """
    blocked = np.array([not check_window(x, calib_std)["fiable"] for x in windows], dtype=bool)
    out = np.asarray(pred).copy()
    out[blocked] = rest_label
    return out, blocked
