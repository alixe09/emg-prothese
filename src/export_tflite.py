"""Export TensorFlow Lite des CNN et mesure du coût embarqué.

Trois variantes par sujet :
- float32 : conversion directe (référence)
- dynamic : poids quantifiés en int8, calculs en float (réduction de taille)
- int8    : poids ET activations en int8 (quantification complète, calibrée sur
            des fenêtres d'entraînement) — le format visé pour un microcontrôleur
            (TFLite Micro, CMSIS-NN…).
            La normalisation par canal est sortie du réseau et faite en float
            avant l'entrée (12 opérations par échantillon), avec écrêtage à
            ±NORM_CLIP écarts-types : quantifier directement le signal brut (en
            volts, très dynamique) écrase les petites amplitudes à zéro et le
            modèle s'effondre (accuracy au niveau du hasard, vérifié).

Pour chaque variante : taille du fichier, accuracy équilibrée (fenêtres de test et
flux continu avec vote sur 3), latence d'inférence mesurée ici sur le PC.
La latence sur microcontrôleur n'est pas mesurée (pas de carte) : on donne le
nombre d'opérations (MAC) par décision, qui permet de l'estimer pour une cible.

Usage : python src/export_tflite.py --subjects 1 2 3 4 5 [--db db3]
"""

import argparse
import json
import tempfile
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from data_loading import load_subject
from inference import NORM_CLIP, TFLiteModel, normalize
from postprocessing import majority_vote, stream_metrics
from preprocessing import TEST_REPS, TRAIN_REPS, filter_emg, make_stream_windows, make_windows
from train import DB, models_dir

VARIANTS = ("float32", "dynamic", "int8")
N_REPRESENTATIVE = 500
VOTE_K = 3


def split_normalization(model):
    """Renvoie (mean, std) de la couche Normalization et le réseau qui la suit
    (sans la normalisation ni le bruit d'augmentation, inactif en inférence)."""
    norm = next(l for l in model.layers if isinstance(l, layers.Normalization))
    mean = np.asarray(norm.mean).reshape(-1)
    # Plancher : électrodes absentes (canal nul) sur certains amputés
    std = np.maximum(np.sqrt(np.asarray(norm.variance).reshape(-1)), 1e-7)
    inputs = layers.Input(shape=model.input_shape[1:])
    x = inputs
    for layer in model.layers:
        if isinstance(layer, (layers.InputLayer, layers.Normalization, layers.GaussianNoise)):
            continue
        x = layer(x)
    return mean.astype(np.float32), std.astype(np.float32), tf.keras.Model(inputs, x)


def count_macs(model):
    """Multiplications-accumulations par fenêtre (Conv1D et Dense)."""
    macs = 0
    for layer in model.layers:
        if isinstance(layer, layers.Conv1D):
            length, out_ch = layer.output.shape[1], layer.output.shape[2]
            k, in_ch = layer.kernel.shape[0], layer.kernel.shape[1]
            macs += length * out_ch * k * in_ch
        elif isinstance(layer, layers.Dense):
            macs += int(np.prod(layer.kernel.shape))
    return int(macs)


def convert(model, variant, representative):
    with tempfile.TemporaryDirectory() as d:
        model.export(d, verbose=False)
        conv = tf.lite.TFLiteConverter.from_saved_model(d)
        if variant in ("dynamic", "int8"):
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
        if variant == "int8":
            def rep_data():
                for x in representative:
                    yield [x[None].astype(np.float32)]
            conv.representative_dataset = rep_data
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # Entrée/sortie int8 : le microcontrôleur reçoit directement des entiers
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
        return conv.convert()


def latency_ms(model, x, n=300):
    model.predict_one(x)
    t = time.perf_counter()
    for _ in range(n):
        model.predict_one(x)
    return (time.perf_counter() - t) / n * 1000


def main(subjects, db=DB):
    out_dir = models_dir(db)
    tflite_dir = out_dir / "tflite"
    tflite_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for s in subjects:
        print(f"\n=== Sujet {s} ===")
        emg, labels, reps = load_subject(s, db=db)
        emg = filter_emg(emg)
        X, y, rep = make_windows(emg, labels, reps)
        Xs, ys, reps_s, block = make_stream_windows(emg, labels, reps)
        del emg
        test, st = np.isin(rep, TEST_REPS), np.isin(reps_s, TEST_REPS)
        X_test, y_test = X[test], y[test]
        Xs, ys, block = Xs[st], ys[st], block[st]
        rng = np.random.default_rng(0)
        train_idx = np.flatnonzero(np.isin(rep, TRAIN_REPS))
        representative = X[rng.choice(train_idx, N_REPRESENTATIVE, replace=False)]
        del X

        keras_model = tf.keras.models.load_model(out_dir / f"cnn_s{s}.keras")
        res = {"macs_par_decision": count_macs(keras_model)}
        ref = keras_model.predict(X_test, verbose=0).argmax(axis=1)
        mean, std, core = split_normalization(keras_model)
        (tflite_dir / f"cnn_s{s}_norm.json").write_text(json.dumps(
            {"mean": mean.tolist(), "std": std.tolist(), "clip": NORM_CLIP}))

        for variant in VARIANTS:
            if variant == "int8":
                content = convert(core, variant, normalize(representative, mean, std))
                m = TFLiteModel(content, norm=(mean, std))
            else:
                content = convert(keras_model, variant, representative)
                m = TFLiteModel(content)
            path = tflite_dir / f"cnn_s{s}_{variant}.tflite"
            path.write_bytes(content)
            pred = m.predict(X_test).argmax(axis=1)
            stream = stream_metrics(ys, majority_vote(m.predict(Xs).argmax(axis=1), VOTE_K, block), block)
            classes = np.unique(y_test)
            res[variant] = {
                "taille_ko": len(content) / 1024,
                "acc_equilibree_fenetres": float(np.mean([(pred[y_test == c] == c).mean() for c in classes])),
                "accord_avec_keras": float((pred == ref).mean()),
                "flux": stream,
                "latence_pc_ms": latency_ms(m, X_test[0]),
                "plus_gros_tenseur_ko": m.largest_tensor_bytes() / 1024,
            }
            r = res[variant]
            print(f"  {variant:8s} {r['taille_ko']:6.1f} Ko | acc. équil. {r['acc_equilibree_fenetres']:.3f} "
                  f"| flux {stream['balanced_accuracy']:.3f} | accord Keras {r['accord_avec_keras']:.3f} "
                  f"| {r['latence_pc_ms']:.2f} ms")
        results[s] = res
        tf.keras.backend.clear_session()

    with open(out_dir / "tflite_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--db", default=DB)
    args = parser.parse_args()
    main(args.subjects, args.db)
