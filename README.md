# Contrôle myoélectrique d'une prothèse de main (EMG)

Décoder le geste voulu (pince, poing fermé, index tendu…) à partir des signaux
électriques des muscles de l'avant-bras (EMG de surface), comme le fait une
prothèse de main myoélectrique — avec les contraintes d'un **objet connecté de
santé embarqué** : décision en temps réel, modèle assez léger pour tourner sur
un microcontrôleur, adaptation rapide à un nouveau porteur.

Troisième volet d'un portfolio en IA appliquée à la santé, après
[stroke-risk-predictor](../test/stroke-risk-predictor) (données tabulaires, ML
classique) et [radio_thoracique](../radio_thoracique) (images, deep learning) :
ici, **signal physiologique temporel**, avec la même démarche (pipeline complet,
interprétabilité, démo Streamlit).

⚠️ **Disclaimer** : projet pédagogique / recherche de stage. Ce n'est **pas** un
dispositif médical.

## Dataset

[Ninapro](https://ninapro.hevs.ch/) — base publique de référence pour le contrôle
myoélectrique :

- **DB2** : 40 sujets valides, 12 électrodes EMG (Delsys Trigno, 2 kHz),
  ~50 mouvements répétés 6 fois.
- **DB3** : 11 sujets **amputés transradiaux**, même protocole et même matériel.

Fichiers `S<n>_E<k>_A1.mat` à placer dans `data/raw/db2/` (et `data/raw/db3/`).
Pour économiser le disque : supprimer les `.zip` après extraction, puis convertir
en `.npz` compact (`src/data_loading.py`).

Protocole d'évaluation standard de la littérature Ninapro : répétitions
1, 3, 4, 6 pour l'entraînement, 2 et 5 pour le test.

## Exploration

Détails dans [`notebooks/01_exploration.ipynb`](notebooks/01_exploration.ipynb).
Chaque geste sollicite une combinaison d'électrodes différente — c'est ce motif
que le modèle apprend à reconnaître :

![Carte d'activation musculaire par geste](docs/carte_activation.png)

## Résultats (5 sujets DB2, exercice 1 : 17 gestes + repos)

Évaluation **en flux continu** (une décision toutes les 50 ms, transitions entre
gestes comprises), sur les répétitions 2 et 5 jamais vues à l'entraînement.
Métriques orientées porteur : accuracy équilibrée entre gestes, **faux gestes au
repos** (la prothèse bouge sans que l'utilisateur le veuille) et délai total de
décision (fenêtre 200 ms + vote + calcul, budget ~300 ms).

### Nouvel utilisateur : combien de calibration faut-il ?

Protocole *leave-one-subject-out* : chaque sujet joue le nouveau porteur, le CNN
est pré-entraîné sur les 4 autres puis ajusté avec 1, 2 ou 4 répétitions du
nouveau porteur (1 répétition ≈ 2 min 20 d'enregistrement). Post-traitement
identique pour toutes les méthodes (vote sur 3 décisions, ~250–275 ms de délai).
Moyenne ± écart-type sur les 5 sujets :

| Calibration | Méthode | Acc. équilibrée | Faux gestes au repos |
|---|---|---|---|
| aucune | CNN pré-entraîné | 17.7 ± 3.8 % | 8.6 % |
| 1 répétition | **LDA** (Hudgins) | **52.6 ± 5.6 %** | 11.4 % |
| | CNN seul | 39.8 ± 3.9 % | 4.3 % |
| | CNN pré-entraîné + ajusté | 45.5 ± 4.3 % | 7.5 % |
| 2 répétitions | LDA | 59.0 ± 4.2 % | 7.9 % |
| | CNN seul | 57.1 ± 4.4 % | 9.9 % |
| | CNN pré-entraîné + ajusté | 56.8 ± 5.7 % | **4.6 %** |
| 4 répétitions | LDA | 61.8 ± 4.3 % | 7.1 % |
| | CNN seul | 60.5 ± 8.6 % | 4.7 % |
| | **CNN pré-entraîné + ajusté** | **63.4 ± 6.3 %** | **3.8 %** |

Lecture :
- **Sans calibration, rien ne marche** (17.7 %) : les signaux EMG diffèrent trop
  d'une personne à l'autre (placement des électrodes, morphologie). Une phase de
  calibration est indispensable.
- **Calibration très courte (1 répétition) : la LDA est la plus robuste**, au prix
  de nombreux faux gestes au repos.
- **Avec 4 répétitions, le CNN pré-entraîné sur d'autres sujets donne la
  meilleure précision et divise presque par 2 les faux gestes au repos** par
  rapport à la LDA. Le pré-entraînement améliore le CNN à 1 et 4 répétitions
  (+5.7 et +2.9 points), pas à 2.
- Le gain en précision reste modeste et n'apparaît que sur 3 sujets sur 5 : avec
  5 sujets, il n'est pas statistiquement établi. Le sujet 4, le plus difficile
  pour toutes les méthodes, est aussi le seul sujet gaucher (protocole réalisé main
  droite) — hypothèse non vérifiée.

Détails : `models/transfer_summary.json` (valeurs par sujet), code dans
[`src/transfer.py`](src/transfer.py). Comparaison intra-sujet avec seuil de
confiance réglé sur validation : [`src/evaluate_stream.py`](src/evaluate_stream.py),
résumé par `python src/summarize.py`.

### Embarqué : export TensorFlow Lite

Export des CNN intra-sujet ([`src/export_tflite.py`](src/export_tflite.py)),
moyenne sur 5 sujets, flux continu avec vote sur 3 :

| Format | Taille | Acc. équilibrée | Accord avec Keras | Calcul / décision (PC) |
|---|---|---|---|---|
| float32 | 188 Ko | 60.8 % | 100 % | 0.18 ms |
| poids int8 | 56 Ko | 60.9 % | ≥ 99.6 % | 0.50 ms |
| **int8 complet** | **58 Ko** | 59.8 % | ≥ 97.9 % | 0.24 ms |

- **5.2 M multiplications-accumulations par décision**, soit ~100 M/s au rythme
  d'une décision toutes les 50 ms. La latence sur microcontrôleur n'a pas été
  mesurée (pas de carte) : c'est la prochaine vérification à faire sur cible.
- Piège rencontré : quantifier en int8 le signal brut (en volts, très dynamique)
  écrase les petites amplitudes à zéro et le modèle tombe au niveau du hasard
  (5.6 %). Solution : normalisation par électrode faite avant le réseau
  (12 opérations par échantillon) + écrêtage à ±8 écarts-types.

### Limites

- 5 sujets valides seulement, un seul entraînement par configuration (pas de
  répétition sur plusieurs graines aléatoires).
- Réglages (seuil, nombre d'époques) choisis sur une seule répétition de
  validation : peu fiable sur certains sujets.
- Pas encore de sujets amputés (DB3) : la question clé pour une vraie prothèse.

## Démo

Application Streamlit qui rejoue un enregistrement réel (sujet 1, répétition 2,
jamais vue à l'entraînement) comme le ferait la prothèse : une fenêtre de 200 ms
toutes les 50 ms, décodée par le modèle **int8** de 58 Ko. On y voit le signal
des 12 électrodes défiler, une main qui prend la pose du geste décodé, les
électrodes qui s'allument selon l'activité musculaire, et une frise des
décisions (gestes corrects, mauvais gestes, mouvements non voulus au repos).

```bash
streamlit run app/streamlit_app.py
```

Les fichiers de la démo (extrait de 148 s, modèle int8) se régénèrent avec
`python src/make_demo_data.py --subject 1`.

## Démarche prévue

1. **Exploration** — visualisation des signaux par électrode et par geste.
2. **Prétraitement** — filtrage passe-bande 20–450 Hz + coupe-bande 50 Hz,
   fenêtres glissantes de 200 ms (pas de 50 ms).
3. **Baseline « prothèse réelle »** — features temporelles de Hudgins
   (MAV, RMS, WL, ZC, SSC) + LDA : l'approche historique des prothèses
   commerciales.
4. **Deep learning** — CNN 1D sur les fenêtres brutes, comparé à la baseline.
5. **Contraintes objet connecté** ✅
   - latence de décision (fenêtre + inférence) < 300 ms ;
   - export TensorFlow Lite quantifié int8, taille mémoire compatible
     microcontrôleur ;
   - nouvel utilisateur : modèle pré-entraîné sur d'autres sujets, recalibré
     avec quelques secondes de données.
6. **Valides vs amputés** — écart de performance DB2 / DB3.
7. **Interprétabilité** — contribution de chaque électrode (muscle) par geste.
8. **Démo** ✅ — signal EMG rejoué en flux + main animée exécutant le geste prédit.

## Structure

```
data/raw/        données Ninapro (non versionnées)
data/processed/  fenêtres / features compactes (.npz)
notebooks/       exploration
src/             chargement, prétraitement, features, entraînement
models/          modèles entraînés
app/             démo Streamlit
docs/            figures
```

## Installation

Le projet réutilise l'environnement de `radio_thoracique` (TensorFlow,
scikit-learn, SciPy, Streamlit déjà installés) pour économiser l'espace disque :

```bash
../radio_thoracique/venv/Scripts/activate
```

Sinon : `python -m venv venv && pip install -r requirements.txt`.

## Référence

Atzori M. et al. (2014). *Electromyography data for non-invasive
naturally-controlled robotic hand prostheses*. Scientific Data 1, 140053.
