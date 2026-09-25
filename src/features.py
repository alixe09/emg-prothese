"""Features temporelles de Hudgins (1993), la base des prothèses myoélectriques.

Calculées par canal sur chaque fenêtre, elles sont très peu coûteuses : c'est ce
qui les rend utilisables sur le microcontrôleur d'une prothèse.
"""

import numpy as np

FEATURE_NAMES = ["MAV", "RMS", "WL", "ZC", "SSC"]


def hudgins_features(X, threshold=0.0):
    """X : (n_fenêtres, taille_fenêtre, n_canaux) -> (n_fenêtres, 5 * n_canaux).

    - MAV : amplitude moyenne absolue (niveau d'activation du muscle)
    - RMS : puissance du signal
    - WL  : longueur de la courbe (complexité / fréquence)
    - ZC  : passages par zéro (proxy de la fréquence)
    - SSC : changements de pente
    `threshold` ignore les petites oscillations dues au bruit pour ZC et SSC.
    """
    diff = np.diff(X, axis=1)

    mav = np.mean(np.abs(X), axis=1)
    rms = np.sqrt(np.mean(X**2, axis=1))
    wl = np.sum(np.abs(diff), axis=1)
    zc = np.sum(
        (X[:, :-1] * X[:, 1:] < 0) & (np.abs(diff) >= threshold), axis=1
    )
    ssc = np.sum(
        (diff[:, :-1] * diff[:, 1:] < 0)
        & ((np.abs(diff[:, :-1]) >= threshold) | (np.abs(diff[:, 1:]) >= threshold)),
        axis=1,
    )
    # Ordre : toutes les features du canal 1, puis du canal 2, etc.
    return np.stack([mav, rms, wl, zc, ssc], axis=2).reshape(len(X), -1)
