# 1. Usage prévu et statut réglementaire

> Exercice pédagogique : ce projet n'est **pas** un dispositif médical et n'a fait
> l'objet d'aucune évaluation clinique ni réglementaire. Ce document décrit ce que
> serait l'usage prévu **si** le logiciel était intégré à une prothèse, afin de
> raisonner comme on le ferait pour un vrai dispositif.

## Destination (intended purpose)

Logiciel embarqué de **reconnaissance de gestes** destiné à piloter une **prothèse de
main myoélectrique** : à partir de l'EMG de surface de l'avant-bras (12 électrodes),
il décide toutes les 50 ms quel geste le porteur veut réaliser (17 gestes de la main
et du poignet, ou repos) et transmet cette consigne aux moteurs de la prothèse.

| Élément | Description |
|---|---|
| **Population de patients** | Adultes amputés transradiaux (sous le coude), avec un moignon d'avant-bras permettant la pose des électrodes |
| **Utilisateurs** | Le porteur au quotidien ; l'orthoprothésiste pour la pose des électrodes et la calibration |
| **Environnement** | Vie quotidienne (domicile, travail), après appareillage et calibration en cabinet |
| **Principe** | Filtrage 20–450 Hz → fenêtre de 200 ms → contrôle qualité du signal → CNN 1D int8 (58 Ko) → vote sur 3 décisions → consigne au moteur |
| **Calibration** | Obligatoire pour chaque porteur : ~9 min d'enregistrement guidé (4 répétitions des gestes) |

## Hors périmètre (contre-indications / limites d'usage)

- **Utilisation sans calibration du porteur** : précision au niveau du hasard (9,9 %
  en moyenne chez les amputés), comportement imprévisible — voir risque R7.
- **Moignon sans avant-bras exploitable** (ex. sujet DB3 n°7, 0 % d'avant-bras
  restant) : signaux au niveau du bruit, aucun contrôle possible.
- Enfants, amputations plus proximales (au-dessus du coude), autres prothèses.
- Toute utilisation diagnostique : le logiciel ne mesure ni n'évalue l'état de santé.
- Commande de charges lourdes ou de tâches à risque (conduite, outils coupants) :
  les mouvements non voulus ne sont pas éliminés (risque R1).

## Statut réglementaire (s'il s'agissait d'un vrai produit)

- **Règlement (UE) 2017/745 (MDR)** : une prothèse de main est un dispositif médical ;
  ce logiciel en serait un composant (logiciel qui pilote un dispositif). La classe du
  dispositif relève des règles de l'annexe VIII et n'est pas déterminée ici.
- **IEC 62304 (cycle de vie du logiciel médical)** : classe de sécurité **B proposée**,
  car une défaillance peut causer une blessure non grave (objet lâché ou serré,
  pincement : risques R1, R2, R4), sans décès ni blessure grave identifiés. À confirmer
  par l'analyse des risques du dispositif complet (force de serrage des moteurs).
- **ISO 14971 (gestion des risques)** : démarche appliquée de façon simplifiée dans
  [l'analyse des risques](02-analyse-risques.md).

## Ce qui manquerait pour un vrai dossier

| Domaine | Référence | État dans ce projet |
|---|---|---|
| Système de management de la qualité | ISO 13485 | absent |
| Évaluation clinique (porteurs réels, en conditions d'usage) | MDR annexe XIV | absent : données de laboratoire, gestes imaginés, sans prothèse réelle |
| Aptitude à l'utilisation | IEC 62366-1 | absent (calibration, alertes au porteur) |
| Sécurité électrique du matériel | IEC 60601-1 | hors périmètre (pas de matériel) |
| Exigences des prothèses externes | ISO 22523 | non consulté |
| Cybersécurité (si liaison Bluetooth) | MDCG 2019-16 | hors périmètre |
| Processus logiciel complet (exigences, architecture, gestion de configuration) | IEC 62304 | partiel : exigences, tests et traçabilité ([matrice](03-matrice-tracabilite.md)) |
| Surveillance après commercialisation | MDR | sans objet |
