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

## Démarche prévue

1. **Exploration** — visualisation des signaux par électrode et par geste.
2. **Prétraitement** — filtrage passe-bande 20–450 Hz + coupe-bande 50 Hz,
   fenêtres glissantes de 200 ms (pas de 50 ms).
3. **Baseline « prothèse réelle »** — features temporelles de Hudgins
   (MAV, RMS, WL, ZC, SSC) + LDA : l'approche historique des prothèses
   commerciales.
4. **Deep learning** — CNN 1D sur les fenêtres brutes, comparé à la baseline.
5. **Contraintes objet connecté**
   - latence de décision (fenêtre + inférence) < 300 ms ;
   - export TensorFlow Lite quantifié int8, taille mémoire compatible
     microcontrôleur ;
   - nouvel utilisateur : modèle pré-entraîné sur d'autres sujets, recalibré
     avec quelques secondes de données.
6. **Valides vs amputés** — écart de performance DB2 / DB3.
7. **Interprétabilité** — contribution de chaque électrode (muscle) par geste.
8. **Démo** — signal EMG rejoué en flux + main animée exécutant le geste prédit.

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
