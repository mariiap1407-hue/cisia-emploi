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


def ajuster_recalibrage(methode, p, y):
    """Apprend la correction de la probabilité de la classe 2 à partir de probabilités hors pli."""
    if methode == "Aucune correction":
        return None
    if methode == "Logistique":                     # régression logistique sur les log-odds
        return LogisticRegression().fit(logit(p).reshape(-1, 1), y)
    if methode == "Isotonique":
        return IsotonicRegression(out_of_bounds="clip").fit(p, y)
    raise ValueError(f"Méthode de recalibration inconnue : {methode}")


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
    if classe == 2:
        niveau = "Élevé"
    elif seuil_garde_fou is not None and risque_usager >= seuil_garde_fou:
        niveau = "Modéré"
    else:
        niveau = "Faible"
    return {"Recommandation": RECOMMANDATIONS[classe], "Niveau d'alerte": niveau,
            "Risque de longue durée": f"{risque_usager:.0%}"}


def charger_regle(chemin):
    """Lit la règle de décision (seuils et méthode de recalibration) depuis le fichier de configuration."""
    with open(chemin, encoding="utf-8") as fichier:
        regle = json.load(fichier)
    if regle["methode_recalibration"] not in METHODES:
        raise ValueError(f"Méthode de recalibration inconnue : {regle['methode_recalibration']}")
    return regle
