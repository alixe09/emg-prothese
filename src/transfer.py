"""Nouvel utilisateur : pré-entraînement multi-sujets + recalibration rapide.

Protocole « leave-one-subject-out » : chaque sujet joue à tour de rôle le
nouveau porteur de prothèse (sujet cible).
- Pré-entraînement du CNN sur les autres sujets (répétitions 1–5, validation
  sur leur répétition 6).
- Calibration sur le sujet cible avec un budget de 1, 2 ou 4 répétitions
  (~8 s de geste par mouvement et par répétition), prises parmi 1, 3, 4, 6.
- Test en flux continu sur les répétitions 2 et 5 du sujet cible.

Méthodes comparées, à budget de calibration égal :
- lda            : LDA entraînée sur la calibration seule (approche classique)
- cnn_seul       : CNN entraîné sur la calibration seule
- cnn_pre_0      : CNN pré-entraîné, sans aucune calibration (« zéro-shot »)
- cnn_pre_ft     : CNN pré-entraîné puis ajusté (fine-tuning) sur la calibration,
                   avec normalisation ré-estimée sur le sujet cible

Avec peu de calibration, il n'y a pas de répétition de validation pour régler un
seuil : toutes les méthodes utilisent le même post-traitement fixe (vote sur 3
décisions, sans seuil), soit ~250–275 ms de délai total.

Usage :
    python src/transfer.py --subjects 1 2 3 4 5 [--db db3]
    python src/transfer.py --summary [--db db3]
"""

import argparse
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers

from data_loading import FS, load_subject
from features import hudgins_features
from postprocessing import majority_vote, stream_metrics
from preprocessing import STEP_MS, TEST_REPS, WINDOW_MS, filter_emg, make_stream_windows, make_windows
from train import BATCH_SIZE, DB, build_cnn, class_weights, fit_cnn, models_dir, train_lda

BUDGETS = {1: (1,), 2: (1, 3), 4: (1, 3, 4, 6)}
PRETRAIN_REPS = (1, 2, 3, 4, 5)
PRETRAIN_VAL_REP = 6
PRETRAIN_EPOCHS_MAX = 30
SCRATCH_EPOCHS = 10   # ~ nombre d'époques retenu sur validation en intra-sujet
FT_EPOCHS = 10
FT_LR = 3e-4
VOTE_K = 3


def window_starts(labels, reps, size, step):
    """Débuts des fenêtres gardées par `make_windows` (label et répétition
    constants sur toute la fenêtre), sans construire les fenêtres."""
    starts = np.arange(0, len(labels) - size + 1, step)
    seg_l = np.r_[0, np.cumsum(labels[1:] != labels[:-1])]
    seg_r = np.r_[0, np.cumsum(reps[1:] != reps[:-1])]
    ok = (seg_l[starts] == seg_l[starts + size - 1]) & (seg_r[starts] == seg_r[starts + size - 1])
    return starts[ok]


class WindowBatches(tf.keras.utils.PyDataset):
    """Lots de fenêtres découpées à la volée dans les signaux filtrés.

    Évite de matérialiser toutes les fenêtres du pool de pré-entraînement
    (~3 Go pour 10 sujets), trop pour une machine à 6 Go de RAM.
    """

    def __init__(self, signals, index, labels, size, weights=None, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.signals, self.index, self.labels = signals, index, labels
        self.size, self.weights, self.shuffle = size, weights, shuffle
        self.order = np.arange(len(index))
        self.rng = np.random.default_rng(42)
        if shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return int(np.ceil(len(self.index) / BATCH_SIZE))

    def __getitem__(self, i):
        idx = self.order[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        X = np.stack([self.signals[a][s : s + self.size] for a, s in self.index[idx]])
        y = self.labels[idx]
        return (X, y, self.weights[idx]) if self.weights is not None else (X, y)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def pretrain(emg_cache, others, n_classes, step_ms=STEP_MS, epochs_max=PRETRAIN_EPOCHS_MAX):
    size, step = int(FS * WINDOW_MS / 1000), int(FS * step_ms / 1000)
    signals, parts = [], {"tr": [], "va": []}
    for a, s in enumerate(others):
        emg, labels, reps = emg_cache[s]
        signals.append(emg)
        st = window_starts(labels, reps, size, step)
        rep = reps[st]
        for key, mask in (("tr", np.isin(rep, PRETRAIN_REPS)), ("va", rep == PRETRAIN_VAL_REP)):
            parts[key].append((np.full(mask.sum(), a), st[mask], labels[st[mask]]))
    index, y, index_v, yv = [], [], [], []
    for a, st, lab in parts["tr"]:
        index.append(np.c_[a, st]); y.append(lab)
    for a, st, lab in parts["va"]:
        index_v.append(np.c_[a, st]); yv.append(lab)
    index, y = np.concatenate(index), np.concatenate(y).astype(np.int64)
    index_v, yv = np.concatenate(index_v), np.concatenate(yv).astype(np.int64)
    print(f"  pré-entraînement : {len(index)} fenêtres, validation : {len(index_v)} "
          f"(pas de {step_ms} ms)")

    # Normalisation par canal estimée sur les échantillons du pool d'entraînement
    pool = np.concatenate([emg_cache[s][0][np.isin(emg_cache[s][2], PRETRAIN_REPS)] for s in others])
    mean, std = pool.mean(axis=0), pool.std(axis=0)
    del pool

    cw = class_weights(y)
    weights = np.array([cw[c] for c in y], dtype=np.float32)
    train_seq = WindowBatches(signals, index, y, size, weights)
    val_seq = WindowBatches(signals, index_v, yv, size, shuffle=False)

    tf.keras.utils.set_random_seed(42)
    model = build_cnn(size, signals[0].shape[1], n_classes, mean, std)
    history = model.fit(
        train_seq, validation_data=val_seq, epochs=epochs_max, verbose=2,
        callbacks=[
            callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3),
            callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ],
    )
    return model, int(np.argmin(history.history["val_loss"])) + 1


def fine_tune(pretrained, X, y, n_classes):
    """Copie du modèle pré-entraîné avec une normalisation ré-estimée sur le
    sujet cible (les amplitudes EMG varient beaucoup d'une personne à l'autre),
    puis ajustement de tous les poids à faible learning rate."""
    tf.keras.utils.set_random_seed(42)
    model = build_cnn(X.shape[1], X.shape[2], n_classes, X.mean(axis=(0, 1)), X.std(axis=(0, 1)))
    for new, old in zip(model.layers, pretrained.layers):
        if not isinstance(new, layers.Normalization):
            new.set_weights(old.get_weights())
    model.compile(
        optimizer=tf.keras.optimizers.Adam(FT_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        X, y, epochs=FT_EPOCHS, batch_size=BATCH_SIZE,
        class_weight=class_weights(y), verbose=0,
        callbacks=[callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3)],
    )
    return model


def evaluate(probs, y_stream, block):
    pred = majority_vote(probs.argmax(axis=1), VOTE_K, block)
    return stream_metrics(y_stream, pred, block)


def run_target(target, subjects, emg_cache, pretrain_kwargs=None):
    X, y, rep = make_windows(*emg_cache[target])
    n_classes = int(max(emg_cache[s][1].max() for s in subjects)) + 1
    Xs, ys, reps_s, block = make_stream_windows(*emg_cache[target])
    test = np.isin(reps_s, TEST_REPS)
    Xs, ys, block = Xs[test], ys[test], block[test]

    pre, pre_epochs = pretrain(emg_cache, [s for s in subjects if s != target], n_classes,
                               **(pretrain_kwargs or {}))
    out = {"target": target, "pretrain_epochs": pre_epochs, "zero_shot": None, "budgets": {}}
    out["zero_shot"] = evaluate(pre.predict(Xs, verbose=0), ys, block)
    print(f"  zéro-shot : acc. équilibrée {out['zero_shot']['balanced_accuracy']:.3f}")

    for budget, reps in BUDGETS.items():
        cal = np.isin(rep, reps)
        res = {}
        lda = train_lda(X[cal], y[cal])
        res["lda"] = evaluate(lda.predict_proba(hudgins_features(Xs)), ys, block)
        scratch, _ = fit_cnn(X[cal], y[cal], n_classes, SCRATCH_EPOCHS)
        res["cnn_seul"] = evaluate(scratch.predict(Xs, verbose=0), ys, block)
        ft = fine_tune(pre, X[cal], y[cal], n_classes)
        res["cnn_pre_ft"] = evaluate(ft.predict(Xs, verbose=0), ys, block)
        out["budgets"][str(budget)] = res
        print(f"  {budget} rép. : " + " | ".join(
            f"{m} {r['balanced_accuracy']:.3f}" for m, r in res.items()))
    return out


def summary(db=DB):
    out_dir = models_dir(db)
    files = sorted(out_dir.glob("transfer_s*.json"), key=lambda p: int(p.stem[len("transfer_s"):]))
    runs = [json.loads(p.read_text()) for p in files]
    targets = [r["target"] for r in runs]
    print(f"Sujets cibles : {targets} — test en flux continu (rép. 2 et 5), vote sur {VOTE_K}\n")
    rows = [("cnn_pre_0", "0", lambda r: r["zero_shot"])]
    for b in BUDGETS:
        for m in ("lda", "cnn_seul", "cnn_pre_ft"):
            rows.append((m, str(b), lambda r, b=b, m=m: r["budgets"][str(b)][m]))
    keys = [("balanced_accuracy", "acc. équil."), ("faux_gestes_au_repos", "faux gestes repos"),
            ("declenchements_intempestifs_par_min", "déclench./min")]
    print(f"{'méthode':12s}{'calib.':>8s}" + "".join(f"{n:>22s}" for _, n in keys))
    table = []
    for m, b, get in rows:
        cells, entry = [], {"methode": m, "repetitions_calibration": int(b)}
        for k, _ in keys:
            v = np.array([get(r)[k] for r in runs]) * (100 if k != keys[2][0] else 1)
            cells.append(f"{v.mean():>14.1f} ± {v.std():4.1f}")
            entry[k] = {"moyenne": float(v.mean()), "ecart_type": float(v.std()),
                        "par_sujet": v.round(2).tolist()}
        table.append(entry)
        print(f"{m:12s}{b:>8s}" + "".join(f"{c:>22s}" for c in cells))
    (out_dir / "transfer_summary.json").write_text(json.dumps(table, indent=2))


def main(subjects, db=DB, all_subjects=(1, 2, 3, 4, 5), pretrain_kwargs=None):
    """`all_subjects` : sujets disponibles pour le pré-entraînement ;
    `subjects` : ceux qui jouent tour à tour le nouveau porteur."""
    all_subjects = list(all_subjects)
    out_dir = models_dir(db)
    print("Chargement et filtrage des sujets (exercice 1)…")
    emg_cache = {}
    for s in all_subjects:
        emg, labels, reps = load_subject(s, db=db)
        emg_cache[s] = (filter_emg(emg), labels, reps)
    for target in subjects:
        path = out_dir / f"transfer_s{target}.json"
        if path.exists():
            print(f"Sujet cible {target} : déjà fait, ignoré")
            continue
        print(f"\n=== Sujet cible {target} ===")
        path.write_text(json.dumps(run_target(target, all_subjects, emg_cache, pretrain_kwargs), indent=2))
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--all-subjects", type=int, nargs="+", default=None,
                        help="sujets du pool (par défaut : ceux de --subjects)")
    parser.add_argument("--db", default=DB)
    parser.add_argument("--pretrain-step-ms", type=int, default=STEP_MS,
                        help="pas des fenêtres de pré-entraînement (100 = moitié moins de fenêtres)")
    parser.add_argument("--pretrain-epochs", type=int, default=PRETRAIN_EPOCHS_MAX)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        summary(args.db)
    else:
        main(args.subjects, args.db, args.all_subjects or args.subjects,
             {"step_ms": args.pretrain_step_ms, "epochs_max": args.pretrain_epochs})
