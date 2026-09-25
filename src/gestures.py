"""Noms des mouvements et des électrodes Ninapro DB2 (Atzori et al., 2014)."""

# Exercice 1 (« exercise B ») : mouvements isométriques des doigts et du poignet
GESTURES_E1 = {
    0: "Repos",
    1: "Pouce levé",
    2: "Index + majeur tendus",
    3: "Annulaire + auriculaire fléchis",
    4: "Pouce vers auriculaire",
    5: "Doigts écartés",
    6: "Poing fermé",
    7: "Index pointé",
    8: "Doigts serrés tendus",
    9: "Supination (axe majeur)",
    10: "Pronation (axe majeur)",
    11: "Supination (axe auriculaire)",
    12: "Pronation (axe auriculaire)",
    13: "Flexion du poignet",
    14: "Extension du poignet",
    15: "Inclinaison radiale",
    16: "Inclinaison ulnaire",
    17: "Extension poignet, main fermée",
}

# Électrodes 1–8 : réparties régulièrement autour de l'avant-bras ;
# 9–12 : placées sur des muscles précis.
ELECTRODES = [f"E{i} (anneau)" for i in range(1, 9)] + [
    "E9 fléchisseur des doigts",
    "E10 extenseur des doigts",
    "E11 biceps",
    "E12 triceps",
]


def gesture_name(label):
    return GESTURES_E1.get(int(label), f"Mouvement {int(label)}")
