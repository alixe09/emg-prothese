"""Post-traitements de décision, comme dans les prothèses myoélectriques réelles.

Deux règles, appliquées en temps réel (elles n'utilisent que le passé) :
1. **Seuil de confiance** : un geste n'est déclenché que si le modèle est assez
   sûr de lui ; sinon la prothèse reste au repos. Un mouvement non voulu est bien
   plus gênant (voire dangereux) qu'un léger retard.
2. **Vote majoritaire** sur les k dernières décisions : lisse les décisions
   isolées. Coût : environ (k - 1) / 2 pas de retard supplémentaire.
"""

import numpy as np

from preprocessing import STEP_MS, WINDOW_MS


def apply_threshold(probs, tau):
    pred = probs.argmax(axis=1)
    pred[(pred != 0) & (probs.max(axis=1) < tau)] = 0
    return pred


def majority_vote(pred, k, block):
    """Vote causal sur les k dernières décisions d'un même bloc continu."""
    if k <= 1:
        return pred.copy()
    out = np.empty_like(pred)
    n_classes = pred.max() + 1
    for b in np.unique(block):
        idx = np.flatnonzero(block == b)
        p = pred[idx]
        for i in range(len(p)):
            counts = np.bincount(p[max(0, i - k + 1) : i + 1], minlength=n_classes)
            # En cas d'égalité, on garde la décision la plus récente
            out[idx[i]] = p[i] if counts[p[i]] == counts.max() else counts.argmax()
    return out


def added_delay_ms(k, step_ms=STEP_MS):
    return (k - 1) / 2 * step_ms


def decision_delay_ms(k, compute_ms, window_ms=WINDOW_MS):
    """Délai total approximatif : fenêtre + vote + calcul."""
    return window_ms + added_delay_ms(k) + compute_ms


def stream_metrics(y_true, y_pred, block, step_ms=STEP_MS):
    rest = y_true == 0
    gestes = ~rest
    classes = np.unique(y_true)
    recalls = [(y_pred[y_true == c] == c).mean() for c in classes]

    # Déclenchements intempestifs : passages repos -> geste pendant un vrai repos
    onsets = 0
    for b in np.unique(block):
        idx = np.flatnonzero(block == b)
        t, p = y_true[idx], y_pred[idx]
        start = (p[1:] != 0) & (p[:-1] == 0) & (t[1:] == 0)
        onsets += int(start.sum()) + int(p[0] != 0 and t[0] == 0)
    rest_minutes = rest.sum() * step_ms / 1000 / 60

    return {
        "balanced_accuracy": float(np.mean(recalls)),
        "rappel_gestes": float((y_pred[gestes] == y_true[gestes]).mean()),
        "faux_gestes_au_repos": float((y_pred[rest] != 0).mean()),
        "declenchements_intempestifs_par_min": float(onsets / rest_minutes),
    }


def tune(probs, y, block, taus, ks, max_false_rest=0.05):
    """Choisit (tau, k) maximisant l'accuracy équilibrée sous contrainte de
    faux gestes au repos <= max_false_rest ; à défaut, minimise ces faux gestes."""
    grid = []
    for tau in taus:
        base = apply_threshold(probs, tau)
        for k in ks:
            m = stream_metrics(y, majority_vote(base, k, block), block)
            grid.append({"tau": float(tau), "k": int(k), **m})
    ok = [g for g in grid if g["faux_gestes_au_repos"] <= max_false_rest]
    best = (max(ok, key=lambda g: g["balanced_accuracy"]) if ok
            else min(grid, key=lambda g: g["faux_gestes_au_repos"]))
    return best, grid
