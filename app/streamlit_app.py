import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.append(str(APP_DIR.parent / "src"))

from gestures import GESTURES_E1  # noqa: E402
from inference import TFLiteModel  # noqa: E402
from postprocessing import majority_vote, stream_metrics  # noqa: E402

# Sujets proposés : un valide et un amputé (chiffres : README, évaluation en flux continu)
SUBJECTS = {
    "db2_s1": {
        "label": "Sujet valide — Ninapro DB2, sujet 1",
        "desc": "Main intacte : le sujet exécute réellement chaque geste.",
        "context": "Précision de ce sujet sur ses 2 répétitions de test : 64 % · moyenne des 5 sujets valides : 59 %.",
    },
    "db3_s8": {
        "label": "Sujet amputé — Ninapro DB3, sujet 8",
        "desc": "Amputé de la main droite (50 % de l'avant-bras restant), porteur d'une "
        "prothèse myoélectrique depuis 4 ans. Il **imagine** chaque geste avec la main "
        "amputée : la main affichée est ce que ferait la prothèse.",
        "context": "Précision de ce sujet sur ses 2 répétitions de test : 46 %, dans la moitié haute des amputés · "
        "moyenne des 11 amputés : 32 % (de 6 % à 54 % selon le moignon et l'expérience).",
    },
}
FS = 2000
WINDOW = 400   # 200 ms
STEP = 100     # une décision toutes les 50 ms
VOTE_K = 3
DISPLAY_DECIM = 8  # 2000 Hz -> 250 Hz pour l'affichage

st.set_page_config(page_title="Contrôle myoélectrique d'une prothèse", page_icon="🦾", layout="wide")

st.title("🦾 Contrôle myoélectrique d'une prothèse de main")
st.caption(
    "Décodage du geste voulu à partir de 12 électrodes EMG de l'avant-bras — "
    "CNN 1D quantifié en int8 (TensorFlow Lite, 58 Ko), pensé pour tourner sur "
    "le microcontrôleur d'une prothèse."
)
st.warning(
    "Outil pédagogique réalisé dans un cadre de projet personnel / recherche de "
    "stage. Ce n'est **pas** un dispositif médical.",
    icon="⚠️",
)


@st.cache_resource
def load_model(key):
    norm = json.loads((APP_DIR / "model" / f"cnn_{key}_norm.json").read_text())
    return TFLiteModel(
        path=str(APP_DIR / "model" / f"cnn_{key}_int8.tflite"),
        norm=(np.array(norm["mean"], np.float32), np.array(norm["std"], np.float32)),
    )


@st.cache_data
def load_recording(key):
    d = np.load(APP_DIR / "demo_data" / f"{key}_rep2.npz")
    labels = d["labels"].astype(int)
    # L'extrait concatène, pour chaque geste, [repos + geste] de la répétition 2 :
    # un nouveau bloc commence à chaque retour geste -> repos.
    block = np.r_[0, np.cumsum((labels[1:] == 0) & (labels[:-1] > 0))]
    return d["emg_uv"].astype(np.float32), labels, block


@st.cache_data
def decode(key):
    """Rejoue l'enregistrement comme la prothèse : une fenêtre de 200 ms toutes les
    50 ms, inférence du modèle int8, vote sur les 3 dernières décisions."""
    emg_uv, labels, block = load_recording(key)
    model = load_model(key)
    emg_v = emg_uv * 1e-6
    ends, blocks = [], []
    for b in np.unique(block):
        idx = np.flatnonzero(block == b)
        e = np.arange(idx[0] + WINDOW - 1, idx[-1] + 1, STEP)
        ends.append(e)
        blocks.append(np.full(len(e), b))
    ends, blocks = np.concatenate(ends), np.concatenate(blocks)
    probs = np.stack([model.predict_one(emg_v[e - WINDOW + 1 : e + 1]) for e in ends])
    pred = majority_vote(probs.argmax(axis=1), VOTE_K, blocks)
    conf = probs[np.arange(len(pred)), pred]
    # Activité de chaque électrode sur la fenêtre (RMS), pour l'animation
    rms = np.stack([np.sqrt(np.mean(emg_uv[e - WINDOW + 1 : e + 1] ** 2, axis=0)) for e in ends])
    return ends, blocks, labels[ends], pred, conf, rms


key = st.radio(
    "Porteur", list(SUBJECTS), format_func=lambda k: SUBJECTS[k]["label"], horizontal=True,
)
st.markdown(SUBJECTS[key]["desc"])
st.caption(SUBJECTS[key]["context"])

emg_uv, labels, block = load_recording(key)
ends, blocks, y_true, y_pred, conf, rms = decode(key)
embarque = json.loads((APP_DIR / "model" / f"cnn_{key}_embarque.json").read_text())

# --- Sélection -----------------------------------------------------------------
left, right = st.columns([2, 1])
with left:
    options = ["Séquence complète (17 gestes)"] + [
        f"{g} — {GESTURES_E1[g]}" for g in range(1, 18)
    ]
    choice = st.selectbox(
        "Geste à rejouer",
        options,
        help="Enregistrement réel du porteur choisi (répétition 2, jamais vue par "
        "son modèle pendant l'entraînement) : quelques secondes de repos, puis le geste.",
    )
with right:
    speed = st.segmented_control("Vitesse", ["0.5×", "1×", "2×"], default="1×")

if choice.startswith("Séquence"):
    sel_blocks = np.unique(block)
else:
    g = int(choice.split(" — ")[0])
    sel_blocks = np.unique(block[labels == g])

s_mask = np.isin(block, sel_blocks)
d_mask = np.isin(blocks, sel_blocks)
s_idx = np.flatnonzero(s_mask)

# --- Métriques de la sélection ---------------------------------------------------
m = stream_metrics(y_true[d_mask], y_pred[d_mask], blocks[d_mask])
c1, c2, c3, c4 = st.columns(4)
if choice.startswith("Séquence"):
    c1.metric("Accuracy équilibrée (cet extrait)", f"{m['balanced_accuracy']:.0%}")
gest = y_true[d_mask] > 0
c2.metric("Fenêtres de geste bien décodées", f"{(y_pred[d_mask][gest] == y_true[d_mask][gest]).mean():.0%}")
c3.metric("Faux gestes pendant le repos", f"{m['faux_gestes_au_repos']:.1%}")
c4.metric("Délai de décision", f"{200 + (VOTE_K - 1) / 2 * 50 + embarque['latence_pc_ms']:.0f} ms",
          help="Fenêtre de 200 ms + vote sur 3 décisions (~50 ms) + calcul du modèle int8.")

# --- Données envoyées au lecteur ------------------------------------------------
emg_sel = emg_uv[s_idx]
n_disp = len(emg_sel) // DISPLAY_DECIM
blk = emg_sel[: n_disp * DISPLAY_DECIM].reshape(n_disp, DISPLAY_DECIM, 12)
# On garde l'échantillon d'amplitude maximale de chaque paquet : les pics restent visibles
pick = np.abs(blk).argmax(axis=1)
disp = np.take_along_axis(blk, pick[:, None, :], axis=1)[:, 0, :]
scale = np.percentile(np.abs(emg_uv), 99.9, axis=0)
disp = np.clip(disp / scale, -1, 1)

# Temps (s) de chaque décision dans la sélection : position de la fin de fenêtre
pos_in_sel = np.searchsorted(s_idx, ends[d_mask])
rms_sel = rms[d_mask]
rms_norm = np.clip(rms_sel / np.percentile(rms, 99, axis=0), 0, 1)

payload = {
    "fsDisp": FS / DISPLAY_DECIM,
    "duration": len(s_idx) / FS,
    "emg": [np.round(disp[:, c] * 100).astype(int).tolist() for c in range(12)],
    "dec": {
        "t": np.round(pos_in_sel / FS, 3).tolist(),
        "truth": y_true[d_mask].tolist(),
        "pred": y_pred[d_mask].tolist(),
        "conf": np.round(conf[d_mask], 3).tolist(),
        "rms": np.round(rms_norm, 2).tolist(),
    },
    "names": {int(k): v for k, v in GESTURES_E1.items()},
    "speed": float((speed or "1×").rstrip("×")),
    "computeMs": round(embarque["latence_pc_ms"], 2),
}

html = (APP_DIR / "player.html").read_text(encoding="utf-8")
st.iframe(html.replace("__PAYLOAD__", json.dumps(payload)), height=640)

# --- Embarqué -------------------------------------------------------------------
st.subheader("Contraintes d'un objet connecté embarqué")
e1, e2, e3 = st.columns(3)
e1.metric("Taille du modèle int8", f"{embarque['taille_ko']:.0f} Ko",
          help=f"Contre {embarque['taille_float32_ko']:.0f} Ko en float32.")
e2.metric("Opérations par décision", f"{embarque['macs_par_decision'] / 1e6:.1f} M MAC")
e3.metric("Calcul par décision (PC)", f"{embarque['latence_pc_ms']:.2f} ms",
          help="Mesuré sur PC avec l'interpréteur TFLite ; non mesuré sur microcontrôleur.")
st.caption(
    "La normalisation par électrode est faite avant le réseau (12 opérations par "
    "échantillon) : quantifier directement le signal brut en int8 écrase les petites "
    "amplitudes et fait chuter le modèle au niveau du hasard. Chaque porteur a son "
    "propre modèle, entraîné sur ses répétitions 1, 3, 4 et 6. Détails et résultats "
    "sur 5 sujets valides et 11 amputés dans le README du projet."
)
