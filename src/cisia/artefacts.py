"""Sauvegarde et chargement du modèle, de la correction du risque et de la règle de décision.

Les trois éléments sont toujours enregistrés ensemble, dans le même dossier : l'API ne peut pas
charger un modèle avec la correction ou la règle d'un autre entraînement.
"""

import json
from pathlib import Path

import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from cisia.decision import charger_regle

FICHIER_MODELE = "modele.joblib"
FICHIER_CORRECTION = "recalibration.joblib"
FICHIER_REGLE = "regle_decision.json"
FICHIER_INFOS = "infos_entrainement.json"


def sauvegarder(dossier, modele, correction, regle, infos):
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    joblib.dump(modele, dossier / FICHIER_MODELE)
    joblib.dump(correction, dossier / FICHIER_CORRECTION)
    for nom, contenu in [(FICHIER_REGLE, regle), (FICHIER_INFOS, infos)]:
        with open(dossier / nom, "w", encoding="utf-8") as fichier:
            json.dump(contenu, fichier, ensure_ascii=False, indent=2)


def charger(dossier):
    dossier = Path(dossier)
    regle = charger_regle(dossier / FICHIER_REGLE)   # même vérification que pour config/
    with open(dossier / FICHIER_INFOS, encoding="utf-8") as fichier:
        infos = json.load(fichier)
    modele = joblib.load(dossier / FICHIER_MODELE)
    correction = joblib.load(dossier / FICHIER_CORRECTION)

    # Vérifications : ordre des classes et cohérence entre la méthode déclarée et la correction chargée
    if list(modele.classes_) != [0, 1, 2]:
        raise ValueError(f"Classes du modèle inattendues : {list(modele.classes_)} (attendu : [0, 1, 2])")
    attendu = {"Aucune correction": type(None), "Logistique": LogisticRegression,
               "Isotonique": IsotonicRegression}
    if not isinstance(correction, attendu[regle["methode_recalibration"]]):
        raise ValueError(f"La correction chargée ({type(correction).__name__}) ne correspond pas "
                         f"à la méthode déclarée ({regle['methode_recalibration']})")
    return {"modele": modele, "correction": correction, "regle": regle, "infos": infos}
