"""Entraînement et comparaison baseline LDA / CNN 1D sur un sujet Ninapro.

Protocole commun aux deux modèles :
- répétitions 1, 3, 4, 6 pour l'entraînement, 2 et 5 pour le test ;
- pour le CNN, la répétition 6 sert d'abord de validation (choix du nombre
  d'époques), puis le modèle est ré-entraîné sur les 4 répétitions, pour être
  comparé à la LDA à données d'entraînement égales.

Usage : python src/train.py --subject 1
"""

import argparse
import json
import time

import numpy as np
import tensorflow as tf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models

from data_loading import ROOT, load_subject
from features import hudgins_features
from preprocessing import TEST_REPS, TRAIN_REPS, filter_emg, make_windows

MODELS_DIR = ROOT / "models"
VAL_REP = 6
EPOCHS_MAX = 40
BATCH_SIZE = 64
REST_FACTOR = 1  # choisi sur validation (src/tune_rest_weight.py) ; voir class_weights()


def build_dataset(subject, exercises=(1,)):
    emg, labels, reps = load_subject(subject, exercises=exercises)
    X, y, rep = make_windows(filter_emg(emg), labels, reps)
    return X, y, rep


def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        # Rappel des gestes hors repos : la prothèse réagit-elle quand on bouge ?
        "balanced_accuracy_gestes": float(
            balanced_accuracy_score(y_true[y_true > 0], y_pred[y_true > 0])
        ),
    }


def latency_ms(predict_one, n=200):
    predict_one()  # échauffement
    t = time.perf_counter()
    for _ in range(n):
        predict_one()
    return (time.perf_counter() - t) / n * 1000


# --- Baseline : features de Hudgins + LDA ------------------------------------

def train_lda(X_train, y_train):
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    return clf.fit(hudgins_features(X_train), y_train)


# --- CNN 1D --------------------------------------------------------------------

def build_cnn(n_samples, n_channels, n_classes, mean, std):
    """CNN 1D compact (~50 k paramètres) : pensé pour tenir sur un microcontrôleur.

    La normalisation par canal est intégrée au modèle : il prend directement le
    signal filtré en entrée, comme il le ferait dans la prothèse.
    """
    inputs = layers.Input(shape=(n_samples, n_channels))
    x = layers.Normalization(mean=mean, variance=std**2)(inputs)
    x = layers.GaussianNoise(0.1)(x)  # augmentation : robustesse au bruit capteur

    for filters, kernel in [(32, 15), (64, 9), (64, 5)]:
        x = layers.Conv1D(filters, kernel, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(4 if filters == 32 else 2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def class_weights(y_train, rest_factor=REST_FACTOR):
    """Poids « balanced » pour les gestes ; le repos reçoit `rest_factor` fois le
    poids moyen d'un geste (au lieu d'être fortement sous-pondéré).

    Avec un équilibrage strict, le repos (~50 % des fenêtres) pèse ~9 fois moins
    qu'un geste : le modèle déclenche alors trop facilement des gestes au repos,
    le défaut le plus gênant pour un porteur de prothèse.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    if rest_factor is not None:
        weights[classes == 0] = rest_factor * weights[classes != 0].mean()
    return dict(zip(classes, weights))


def fit_cnn(X_train, y_train, n_classes, epochs, X_val=None, y_val=None,
            rest_factor=REST_FACTOR):
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1))
    tf.keras.utils.set_random_seed(42)
    model = build_cnn(X_train.shape[1], X_train.shape[2], n_classes, mean, std)

    cbs = [callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3)]
    if X_val is not None:
        cbs.append(callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ))
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if X_val is not None else None,
        epochs=epochs, batch_size=BATCH_SIZE,
        class_weight=class_weights(y_train, rest_factor),
        callbacks=cbs, verbose=2,
    )
    return model, history


def main(subject):
    X, y, rep = build_dataset(subject)
    n_classes = int(y.max()) + 1
    train, test = np.isin(rep, TRAIN_REPS), np.isin(rep, TEST_REPS)
    print(f"Sujet {subject} : {train.sum()} fenêtres d'entraînement, "
          f"{test.sum()} de test, {n_classes} classes")

    results = {"subject": subject}

    lda = train_lda(X[train], y[train])
    results["lda"] = evaluate(y[test], lda.predict(hudgins_features(X[test])))
    one = X[test][:1]
    results["lda"]["latence_ms"] = latency_ms(lambda: lda.predict(hudgins_features(one)))
    print("LDA :", results["lda"])

    # 1) Choix du nombre d'époques sur la répétition 6
    fit_reps = [r for r in TRAIN_REPS if r != VAL_REP]
    fit, val = np.isin(rep, fit_reps), rep == VAL_REP
    _, history = fit_cnn(X[fit], y[fit], n_classes, EPOCHS_MAX, X[val], y[val])
    best_epochs = int(np.argmin(history.history["val_loss"])) + 1
    print(f"Meilleure époque sur la validation : {best_epochs}")

    # 2) Ré-entraînement sur les 4 répétitions, évaluation sur 2 et 5
    cnn, _ = fit_cnn(X[train], y[train], n_classes, best_epochs)
    y_pred = cnn.predict(X[test], verbose=0).argmax(axis=1)
    results["cnn"] = evaluate(y[test], y_pred)
    results["cnn"]["epochs"] = best_epochs
    results["cnn"]["params"] = int(cnn.count_params())
    one_tf = tf.constant(one)
    results["cnn"]["latence_ms"] = latency_ms(lambda: cnn(one_tf, training=False))
    print("CNN :", results["cnn"])

    MODELS_DIR.mkdir(exist_ok=True)
    cnn.save(MODELS_DIR / f"cnn_s{subject}.keras")
    results["confusion_cnn"] = confusion_matrix(y[test], y_pred).tolist()
    results["history_val"] = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(MODELS_DIR / f"results_s{subject}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    main(parser.parse_args().subject)
