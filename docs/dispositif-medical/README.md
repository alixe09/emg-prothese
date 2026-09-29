# Dossier « dispositif médical » (exercice)

> ⚠️ Exercice pédagogique réalisé dans le cadre d'un projet personnel. Ce projet
> n'est **pas** un dispositif médical ; ce dossier n'a aucune valeur réglementaire.
> Il applique, de façon simplifiée, la démarche d'un fabricant : définir l'usage
> prévu, analyser les risques, fixer des exigences et prouver qu'elles sont
> vérifiées.

| Document | Contenu |
|---|---|
| [1. Usage prévu et statut réglementaire](01-usage-prevu.md) | Pour qui, dans quel contexte, hors périmètre ; MDR, IEC 62304 (classe B proposée) ; ce qui manquerait pour un vrai dossier |
| [2. Analyse des risques](02-analyse-risques.md) | 9 risques (démarche ISO 14971) : gravité, fréquence mesurée, mesures de réduction, risque résiduel |
| [3. Matrice de traçabilité](03-matrice-tracabilite.md) | 8 exigences → méthode de vérification → résultat valides / amputés → statut. **Générée automatiquement** |

## En bref

- **Maîtrisé et vérifié par tests** : délai ≤ 300 ms, état sûr par défaut, modèle
  embarquable (58 Ko), quantification int8 sans perte notable, et un défaut de
  sécurité découvert pendant l'analyse puis corrigé : face à une électrode saturée
  ou à un bruit fort, le réseau déclenchait un geste avec 99 % de confiance. Un
  contrôle de qualité du signal ([`src/signal_quality.py`](../../src/signal_quality.py))
  force désormais le repos.
- **Non conforme chez les amputés** : mouvements non voulus, précision et
  calibration. Le dossier le dit explicitement : c'est l'écart principal.

## Régénérer

```bash
python -m pytest tests            # 15 tests : traitement du signal, sécurité, modèle embarqué
python src/verification_report.py # relance les tests et régénère la matrice
```
