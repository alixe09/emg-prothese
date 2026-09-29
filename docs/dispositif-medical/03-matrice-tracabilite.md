# 3. Matrice de traçabilité exigences → vérification

*Générée automatiquement par `python src/verification_report.py` le 29/09/2026 à partir des résultats enregistrés et de la suite de tests (15/15 tests réussis). Ne pas modifier à la main.*

Les critères d'acceptation sont **fixés par l'auteur à titre d'exercice** (ils ne proviennent pas d'une norme). DB2 = 5 sujets valides, DB3 = 11 sujets amputés.

| ID | Exigence | Critère | Risques | Vérification | Valides (DB2) | Amputés (DB3) | Statut |
|---|---|---|---|---|---|---|---|
| EX-01 | Délai total de décision (fenêtre + vote + calcul) | ≤ 300 ms | R3 | Mesure sur flux continu ; tests `test_EX01_delai_total_sous_300_ms`, `test_EX01_calcul_int8_rapide` | max 277 ms | max 273 ms | ✅ conforme |
| EX-02 | Faux gestes pendant le repos (CNN) | moyenne ≤ 5 %, chaque sujet ≤ 10 % | R1 | Flux continu, répétitions de test 2 et 5 (`src/evaluate_stream.py`) | moy. 3.0 %, max 4.9 % | moy. 7.2 %, max 23.6 % | DB2 ✅ conforme · DB3 ❌ non conforme |
| EX-03 | Précision équilibrée, 17 gestes + repos (CNN) | moyenne ≥ 50 % | R2 | Flux continu, répétitions de test 2 et 5 | moy. 59.3 % (min 47.5 %) | moy. 32.3 % (min 5.8 %) | DB2 ✅ conforme · DB3 ❌ non conforme |
| EX-04 | Taille du modèle embarqué (int8) | ≤ 256 Ko | — | Test `test_EX04_modele_tient_dans_un_microcontroleur` | 58 Ko | 58 Ko | ✅ conforme |
| EX-05 | Perte de précision due à la quantification int8 | moyenne ≤ 2 points, chaque sujet ≤ 5 points | R6 | `src/export_tflite.py` (5 sujets DB2) ; test de non-régression `test_R6_non_regression_du_modele_int8` | moy. 1.0 pt, max 3.2 pt | sujet 8 : pas de perte | ✅ conforme |
| EX-06 | État sûr : sans signal ou sans décision fiable, la prothèse reste au repos | décision = repos | R1, R4 | Tests `test_EX06_confiance_insuffisante_donne_repos`, `test_EX06_signal_nul_donne_repos`, `test_EX06_vote_causal_et_limite_au_bloc` | vérifié par test | vérifié par test | ✅ conforme |
| EX-07 | Électrode saturée ou bruit fort : pas de geste déclenché | décision = repos, fausses alarmes < 0,1 % | R4 | Tests `test_EX07_defaut_capteur_detecte_et_force_le_repos`, `test_EX07_peu_de_fausses_alarmes_sur_signal_reel` (`src/signal_quality.py`) | vérifié par test | vérifié par test | ✅ conforme |
| EX-08 | Nouveau porteur : ≤ 10 min de calibration pour atteindre EX-03 | ≥ 50 % avec 4 répétitions (~9 min) | R7 | Leave-one-subject-out (`src/transfer.py`) | CNN pré-entraîné 63.4 %, LDA 61.8 % | CNN pré-entraîné 36.3 %, LDA 41.4 % | DB2 ✅ conforme · DB3 ❌ non conforme |

## Lecture

- Les exigences **de sécurité du logiciel** (délai, état sûr, défaut capteur, embarquabilité, quantification) sont vérifiées.
- Les exigences **de performance clinique** (EX-02, EX-03, EX-08) sont atteintes sur sujets valides mais **pas chez les amputés** : en l'état, le logiciel ne serait pas acceptable pour la population visée. C'est le principal écart ouvert (voir l'analyse des risques, R1, R2 et R7).
