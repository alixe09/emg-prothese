"""Chargement des fichiers Ninapro (.mat) et conversion en .npz compact.

Chaque fichier `S<n>_E<k>_A1.mat` contient notamment :
- `emg`          : signal EMG, shape (n_échantillons, 12), 2 kHz
- `restimulus`   : label du mouvement, ré-aligné a posteriori sur l'EMG
                   (plus fiable que `stimulus`, qui suit l'écran de consigne)
- `rerepetition` : numéro de répétition (1 à 6), ré-aligné de la même façon
Le label 0 correspond au repos.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

FS = 2000  # fréquence d'échantillonnage Ninapro DB2 / DB3 (Hz)
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_mat(path):
    """Charge un fichier Ninapro et renvoie (emg, labels, repetitions)."""
    mat = loadmat(path, variable_names=["emg", "restimulus", "rerepetition"])
    emg = mat["emg"].astype(np.float32)
    labels = mat["restimulus"].ravel().astype(np.int16)
    repetitions = mat["rerepetition"].ravel().astype(np.int8)
    # Dans certains fichiers (ex. S1_E3), les labels ont un échantillon de moins
    n = min(len(emg), len(labels), len(repetitions))
    return emg[:n], labels[:n], repetitions[:n]


def load_subject(subject, db="db2", exercises=(1,), raw_dir=RAW_DIR):
    """Concatène les exercices demandés d'un sujet.

    Dans DB2, les labels sont déjà uniques d'un exercice à l'autre :
    E1 = mouvements 1–17, E2 = 18–40, E3 = 41–49.
    """
    emgs, labels, reps = [], [], []
    for ex in exercises:
        matches = list((raw_dir / db).rglob(f"S{subject}_E{ex}_A1.mat"))
        if not matches:
            raise FileNotFoundError(
                f"S{subject}_E{ex}_A1.mat introuvable dans {raw_dir / db}"
            )
        emg, lab, rep = load_mat(matches[0])
        emgs.append(emg)
        labels.append(lab)
        reps.append(rep)
    return np.concatenate(emgs), np.concatenate(labels), np.concatenate(reps)


def save_npz(subject, db="db2", exercises=(1,), out_dir=PROCESSED_DIR):
    """Sauvegarde un sujet en .npz compressé (float32) pour libérer le disque."""
    emg, labels, reps = load_subject(subject, db, exercises)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{db}_s{subject}.npz"
    np.savez_compressed(out, emg=emg, labels=labels, repetitions=reps)
    return out
