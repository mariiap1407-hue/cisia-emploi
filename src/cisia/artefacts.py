"""Sauvegarde et chargement des composants d'un entraînement : modèle, correction du risque, règle.

Les composants sont enregistrés ensemble, avec l'empreinte (SHA-256) de chaque fichier.
Au chargement, les empreintes sont vérifiées : un composant remplacé par celui d'un autre
entraînement est détecté.
"""

import hashlib
import json
from pathlib import Path

import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from cisia.decision import charger_regle

FICHIERS = {
    "modele": "modele.joblib",
    "correction": "recalibration.joblib",
    "regle": "regle_decision.json",
    "infos": "infos_entrainement.json",
}


def empreinte(chemin):
    """Empreinte SHA-256 d'un fichier : elle change dès qu'un seul octet change."""
    return hashlib.sha256(Path(chemin).read_bytes()).hexdigest()


def sauvegarder(dossier, modele, correction, regle, infos):
    """Enregistre les composants dans `dossier` et renvoie le chemin de chaque fichier."""
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    chemins = {nom: dossier / fichier for nom, fichier in FICHIERS.items()}
    joblib.dump(modele, chemins["modele"])
    joblib.dump(correction, chemins["correction"])
    with open(chemins["regle"], "w", encoding="utf-8") as fichier:
        json.dump(regle, fichier, ensure_ascii=False, indent=2)
    empreintes = {nom: empreinte(chemins[nom]) for nom in ["modele", "correction", "regle"]}
    infos = {**infos, "empreintes": empreintes}
    with open(chemins["infos"], "w", encoding="utf-8") as fichier:
        json.dump(infos, fichier, ensure_ascii=False, indent=2)
    return chemins


def charger(dossier):
    """Charge les composants enregistrés dans `dossier`."""
    return charger_fichiers({nom: Path(dossier) / fichier for nom, fichier in FICHIERS.items()})


def charger_fichiers(chemins):
    """Charge les composants à partir du chemin de chaque fichier, et vérifie qu'ils vont ensemble."""
    with open(chemins["infos"], encoding="utf-8") as fichier:
        infos = json.load(fichier)
    for nom in ["modele", "correction", "regle"]:
        if empreinte(chemins[nom]) != infos["empreintes"][nom]:
            raise ValueError(f"Le fichier « {nom} » ne correspond pas à l'entraînement enregistré "
                             f"(empreinte différente) : composants mélangés ou modifiés.")

    regle = charger_regle(chemins["regle"])   # même vérification que pour config/
    modele = joblib.load(chemins["modele"])
    correction = joblib.load(chemins["correction"])

    # Vérifications : ordre des classes et cohérence entre la méthode déclarée et la correction chargée
    if list(modele.classes_) != [0, 1, 2]:
        raise ValueError(f"Classes du modèle inattendues : {list(modele.classes_)} (attendu : [0, 1, 2])")
    attendu = {"Aucune correction": type(None), "Logistique": LogisticRegression,
               "Isotonique": IsotonicRegression}
    if not isinstance(correction, attendu[regle["methode_recalibration"]]):
        raise ValueError(f"La correction chargée ({type(correction).__name__}) ne correspond pas "
                         f"à la méthode déclarée ({regle['methode_recalibration']})")
    return {"modele": modele, "correction": correction, "regle": regle, "infos": infos}
