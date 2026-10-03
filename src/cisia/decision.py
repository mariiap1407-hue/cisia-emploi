"""Recalibration du risque de la classe 2 et règle de décision (notebook, sections 6.3 à 6.6)."""

import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

METHODES = ["Aucune correction", "Logistique", "Isotonique"]

RECOMMANDATIONS = {0: "Accompagnement léger", 1: "Accompagnement standard", 2: "Accompagnement renforcé"}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ajuster_recalibrage(methode, p, y_classe_2):
    """Apprend la correction de la probabilité de la classe 2 à partir de probabilités hors pli.

    y_classe_2 est une cible BINAIRE : 1 si l'usager est en classe 2, 0 sinon,
    soit (y == 2).astype(int). Passer la cible à 3 classes serait une erreur silencieuse :
    la régression logistique apprendrait 3 classes et la correction renverrait le risque de la classe 1.
    """
    if methode not in METHODES:
        raise ValueError(f"Méthode de recalibration inconnue : {methode}")
    y_classe_2 = np.asarray(y_classe_2)
    if not set(np.unique(y_classe_2)) <= {0, 1}:
        raise ValueError("La cible de la recalibration doit être binaire (1 = classe 2, 0 = autre classe).")
    if len(np.unique(y_classe_2)) < 2:
        raise ValueError("La cible de la recalibration doit contenir des usagers de classe 2 et des autres.")
    if methode == "Aucune correction":
        return None
    if methode == "Logistique":                     # régression logistique sur les log-odds
        return LogisticRegression().fit(logit(p).reshape(-1, 1), y_classe_2)
    return IsotonicRegression(out_of_bounds="clip").fit(p, y_classe_2)


def appliquer_recalibrage(methode, modele, p):
    """Transforme la probabilité brute de la classe 2 en probabilité recalibrée."""
    if methode == "Aucune correction":
        return p
    if methode == "Logistique":
        return modele.predict_proba(logit(p).reshape(-1, 1))[:, 1]
    if methode == "Isotonique":
        return modele.predict(p)
    raise ValueError(f"Méthode de recalibration inconnue : {methode}")


def appliquer_regle(probas, risque, seuil_classe_2=None, seuil_garde_fou=None):
    """Règle de décision.

    - probas : probabilités brutes des 3 classes (pour choisir entre les classes 0 et 1) ;
    - risque : risque recalibré de la classe 2 ;
    - seuil_classe_2 : classe 2 si le risque atteint ce seuil, sinon la plus probable entre 0 et 1.
      Absent : règle par défaut (classe la plus probable) ;
    - seuil_garde_fou : la classe 0 n'est jamais prédite si le risque atteint ce seuil
      (orientation en classe 1).
    """
    def absent(s):
        return s is None or pd.isna(s)

    if absent(seuil_classe_2):
        y_pred = probas.argmax(axis=1)
    else:
        y_pred = np.where(risque >= seuil_classe_2, 2, probas[:, :2].argmax(axis=1))
    if not absent(seuil_garde_fou):
        y_pred = np.where((y_pred == 0) & (risque >= seuil_garde_fou), 1, y_pred)
    return y_pred


def presenter(probas_usager, risque_usager, seuil_classe_2, seuil_garde_fou):
    """Recommandation, niveau d'alerte et risque de longue durée pour un usager (section 6.6).

    Dans le notebook, les seuils étaient des variables globales ; ici, ils sont passés en paramètres.
    """
    classe = appliquer_regle(probas_usager.reshape(1, -1), np.array([risque_usager]),
                             seuil_classe_2, seuil_garde_fou)[0]
    return {"Recommandation": RECOMMANDATIONS[classe],
            "Niveau d'alerte": niveau_alerte(classe, risque_usager, seuil_garde_fou),
            "Risque de longue durée": f"{risque_usager:.0%}"}


def niveau_alerte(classe, risque_usager, seuil_garde_fou):
    """Élevé si la règle oriente en classe 2 ; Modéré si le risque atteint le garde-fou ; Faible sinon."""
    if classe == 2:
        return "Élevé"
    if seuil_garde_fou is not None and risque_usager >= seuil_garde_fou:
        return "Modéré"
    return "Faible"


def charger_regle(chemin):
    """Lit la règle de décision et vérifie qu'elle est utilisable.

    Champs obligatoires : seuil_classe_2 et seuil_garde_fou_classe_0 (entre 0 et 1, ou null pour « absent »),
    methode_recalibration (une des METHODES).
    """
    with open(chemin, encoding="utf-8") as fichier:
        regle = json.load(fichier)
    for champ in ["seuil_classe_2", "seuil_garde_fou_classe_0", "methode_recalibration"]:
        if champ not in regle:
            raise ValueError(f"Règle de décision incomplète : champ « {champ} » absent ({chemin})")
    if regle["methode_recalibration"] not in METHODES:
        raise ValueError(f"Méthode de recalibration inconnue : {regle['methode_recalibration']}")
    for champ in ["seuil_classe_2", "seuil_garde_fou_classe_0"]:
        seuil = regle[champ]
        if seuil is not None and not (isinstance(seuil, (int, float)) and 0 <= seuil <= 1):
            raise ValueError(f"Seuil invalide pour « {champ} » : {seuil} (attendu : entre 0 et 1, ou null)")
    return regle
