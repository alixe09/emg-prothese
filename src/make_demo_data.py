"""Prépare les fichiers légers embarqués par la démo Streamlit.

- Un extrait réel : exercice 1, répétition 2 (jamais vue à l'entraînement) du
  sujet choisi, EMG filtré en µV (float16), avec labels et répétitions.
- Le modèle TFLite int8 du même sujet et ses paramètres de normalisation.
- Les mesures d'export (taille, MAC, latence) pour l'affichage.

Usage : python src/make_demo_data.py --subject 1 [--db db3]
"""

import argparse
import json
import shutil

import numpy as np

from data_loading import ROOT, load_subject
from preprocessing import filter_emg
from train import DB, models_dir

APP_DIR = ROOT / "app"
DEMO_REP = 2


def main(subject, db=DB):
    name = f"{db}_s{subject}"
    emg, labels, reps = load_subject(subject, db=db)
    emg = filter_emg(emg)
    keep = reps == DEMO_REP
    out_dir = APP_DIR / "demo_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{name}_rep{DEMO_REP}.npz",
        emg_uv=(emg[keep] * 1e6).astype(np.float16),
        labels=labels[keep].astype(np.int8),
    )

    model_dir = APP_DIR / "model"
    model_dir.mkdir(exist_ok=True)
    tflite = models_dir(db) / "tflite"
    shutil.copy(tflite / f"cnn_s{subject}_int8.tflite", model_dir / f"cnn_{name}_int8.tflite")
    shutil.copy(tflite / f"cnn_s{subject}_norm.json", model_dir / f"cnn_{name}_norm.json")

    results = json.loads((models_dir(db) / "tflite_results.json").read_text())[str(subject)]
    (model_dir / f"cnn_{name}_embarque.json").write_text(json.dumps({
        "macs_par_decision": results["macs_par_decision"],
        "taille_ko": results["int8"]["taille_ko"],
        "taille_float32_ko": results["float32"]["taille_ko"],
        "latence_pc_ms": results["int8"]["latence_pc_ms"],
    }, indent=2))
    size = (out_dir / f"{name}_rep{DEMO_REP}.npz").stat().st_size / 1e6
    print(f"Extrait : {keep.sum() / 2000:.0f} s, {size:.1f} Mo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--db", default=DB)
    args = parser.parse_args()
    main(args.subject, args.db)
