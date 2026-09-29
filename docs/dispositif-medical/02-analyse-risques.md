# 2. Analyse des risques (démarche inspirée d'ISO 14971)

> Exercice pédagogique, limité au **logiciel de décision**. Les probabilités sont
> estimées à partir des mesures du projet (données Ninapro de laboratoire), pas
> d'un usage réel.

## Échelles

**Gravité** du dommage :
1 = négligeable (gêne passagère) · 2 = mineure (geste à refaire, frustration) ·
3 = sérieuse (blessure légère : pincement, objet chaud lâché) · 4 = critique
(blessure grave — non identifiée pour ce logiciel).

**Fréquence** estimée pendant l'usage :
fréquente (plus d'une fois par minute) · occasionnelle (plus d'une fois par heure) ·
rare (moins d'une fois par heure) · non estimée.

**Acceptabilité** : ❌ inacceptable (gravité ≥ 3 et fréquente) · ⚠️ à réduire autant
que possible · ✅ acceptable.

## Tableau des risques

| ID | Danger → situation dangereuse → dommage | Gravité | Avant mesures | Mesures de réduction (vérification) | Après mesures | Risque résiduel |
|---|---|---|---|---|---|---|
| **R1** | Repos décodé comme un geste → la main se ferme ou s'ouvre sans le vouloir → objet lâché ou serré, pincement | 3 | Fréquente : CNN brut, sujet valide 1 : 26 % des fenêtres de repos, 75 déclenchements/min | M1 poids du repos à l'entraînement ; M2 vote sur 3 décisions ; M3 seuil de confiance réglé sur validation ; état sûr par défaut (EX-06) | Valides : 3,0 % des fenêtres, 9,8 déclench./min. **Amputés : 7,2 % (jusqu'à 23,6 %), 16,6 déclench./min** | ⚠️ valides (encore fréquent) · ❌ amputés |
| **R2** | Mauvais geste décodé pendant un geste voulu → prise inadaptée → objet lâché | 2–3 | Fréquente | M2 vote ; EX-03 précision ; calibration par porteur | Valides : 59 % de précision équilibrée. **Amputés : 32 %** | ⚠️ valides · ❌ amputés |
| **R3** | Décision trop lente → la main réagit en retard → frustration, abandon de la prothèse | 2 | — | Fenêtre de 200 ms, vote limité à 3 décisions, modèle int8 rapide (EX-01) | ≤ 277 ms mesurés | ✅ |
| **R4** | Électrode saturée, bruit fort (câble, interférence) → **le CNN déclenche un geste avec jusqu'à 99 % de confiance** (constaté) → mouvement non voulu | 3 | Chaque fois que le défaut survient ; le seuil de confiance ne protège pas | M4 contrôle qualité du signal (`src/signal_quality.py`) : amplitude > 20 × calibration → repos ; électrode muette signalée (EX-07) | Défauts testés bloqués ; 0,007 % de fausses alarmes sur signaux réels (sans danger : repos) | ⚠️ défauts non testés : dérive lente, électrode déplacée |
| **R5** | Dérive du signal au fil de la journée (fatigue, transpiration, emboîture remise) → baisse progressive de la précision | 2–3 | Non estimée (enregistrements d'une seule séance) | Recalibration rapide (R7) ; à tester sur plusieurs séances | Non vérifié | ⚠️ ouvert |
| **R6** | Portage embarqué incorrect (quantification int8) → précision effondrée | 3 | **Constaté : 5,6 % (hasard)** avec quantification du signal brut | Normalisation hors du réseau + écrêtage ; accord int8/float ≥ 97,9 % ; test de non-régression (EX-05) | Perte moyenne 1,0 point | ✅ |
| **R7** | Porteur non calibré ou moignon inexploitable → décisions quasi aléatoires | 3 | Zéro-shot : 9,9 % chez les amputés ; sujet sans avant-bras : 5,8 % même calibré | Calibration obligatoire (usage prévu) ; **proposé** : refuser la mise en service si la précision de calibration est sous un seuil | Amputés calibrés (~9 min) : 36–41 % | ❌ amputés |
| **R8** | Signal absent (panne capteur, batterie, liaison) → décision sur des données manquantes | 3 | — | Signal nul → repos, vérifié par test (EX-06) ; surveillance batterie/liaison hors périmètre logiciel | Vérifié pour un signal nul | ✅ logiciel · matériel hors périmètre |
| **R9** | Données EMG = données de santé → atteinte à la vie privée | — | — | Données publiques pseudonymisées ; la démo n'enregistre rien | — | ✅ |

## Conclusion bénéfice / risque (exercice)

- Les défauts **techniques** du logiciel (délai, défaut capteur, portage embarqué,
  absence de signal) sont maîtrisés et vérifiés par des tests automatisés.
- Les risques liés à la **performance de reconnaissance** (R1, R2, R7) restent
  **inacceptables chez les amputés**, la population visée. En l'état, le logiciel ne
  pourrait pas être mis sur le marché.
- Pistes de réduction identifiées, non réalisées :
  - exiger plusieurs décisions consécutives identiques avant de lancer un mouvement
    (réduit R1 au prix d'un peu de délai) ;
  - réduire le nombre de gestes à ceux que le porteur contrôle bien (moins de classes,
    comme les prothèses commerciales) ;
  - critère d'acceptation à la calibration (R7) ;
  - évaluation sur plusieurs séances et avec retour visuel d'une prothèse réelle (R5).
