"""Quality gate : test de non-régression sur un jeu de référence (même principe que le projet M5).

Le test charge un modèle (ou en entraîne un de contrôle), calcule ses indicateurs sur un jeu de
référence et ÉCHOUE si un seuil de la section 6.6 n'est pas respecté :
erreurs critiques <= 10 %, rappel de la classe 2 >= 60 %, F1 macro >= 0,62.

Variables d'environnement (facultatives) :
- CISIA_MODELE : dossier des composants d'un modèle (ex. models/candidats/<run>), à tester tel quel ;
  sans elle, un modèle de contrôle est entraîné sur les données factices ;
- CISIA_REFERENCE : jeu de référence (par défaut data/factice/reference.csv).
  Avec data/factice/reference_derivee.csv, le test doit échouer : c'est la démonstration
  que le quality gate détecte une dégradation.
"""

import os
from pathlib import Path

import pytest

from cisia import artefacts
from cisia.decision import charger_regle
from cisia.entrainement import entrainer
from cisia.evaluation import SEUILS_QUALITE, mesurer, verifier_seuils_qualite
from cisia.inference import predire_usagers
from cisia.modele_mlflow import charger_modele_en_service
from cisia.preparation import CIBLE, charger_donnees, decouper, nettoyer, separer_x_y

RACINE = Path(__file__).resolve().parents[1]
REFERENCE = Path(os.getenv("CISIA_REFERENCE", RACINE / "data" / "factice" / "reference.csv"))
VRAIES_DONNEES = RACINE / "data" / "raw" / "dataset_trajectoire_emploi.csv"
PRODUCTION = RACINE / "models" / "production"


@pytest.fixture(scope="module")
def composants(donnees_factices):
    """Le modèle à tester : celui indiqué par CISIA_MODELE, sinon un modèle de contrôle (données factices)."""
    if os.getenv("CISIA_MODELE"):
        return artefacts.charger(os.getenv("CISIA_MODELE"))
    entrainement, _ = decouper(charger_donnees(donnees_factices / "entrainement.csv"))
    X_train, y_train = separer_x_y(nettoyer(entrainement))
    regle = charger_regle(RACINE / "config" / "regle_decision.json")
    modele, correction = entrainer(X_train, y_train, regle)
    return {"modele": modele, "correction": correction, "regle": regle}


def indicateurs(composants, chemin_reference):
    """Prédictions de la chaîne complète sur le jeu de référence, puis indicateurs."""
    reference = charger_donnees(chemin_reference)
    sortie = predire_usagers(composants, reference.drop(columns=[CIBLE]))
    return mesurer(reference[CIBLE], sortie["classe"])


def afficher(titre, resultats):
    print(f"\n{titre}")
    for indicateur, (sens, seuil) in SEUILS_QUALITE.items():
        print(f"  {indicateur:<25} {resultats[indicateur]:.3f}   (seuil {sens} : {seuil})")


def test_quality_gate_sur_le_jeu_de_reference(composants):
    resultats = indicateurs(composants, REFERENCE)
    afficher(f"Jeu de référence : {REFERENCE.name}", resultats)
    echecs = verifier_seuils_qualite(resultats)
    assert not echecs, "Quality gate non respecté : " + " ; ".join(echecs)


def test_la_degradation_est_detectee(composants, donnees_factices):
    # Sur le jeu dérivé, le lien entre la synthèse et la classe a changé : le gate doit refuser le modèle
    resultats = indicateurs(composants, donnees_factices / "reference_derivee.csv")
    afficher("Jeu dérivé (dégradation simulée)", resultats)
    assert verifier_seuils_qualite(resultats), "Le quality gate n'a pas détecté la dégradation"


@pytest.mark.skipif(not (VRAIES_DONNEES.exists() and (PRODUCTION / "actuelle.json").exists()),
                    reason="vraies données ou modèle en production absents (CI)")
def test_modele_en_production_sur_les_vraies_donnees():
    """Poste local : le modèle EN SERVICE, chargé comme par l'API, respecte le gate sur le jeu de test."""
    _, test = decouper(charger_donnees(VRAIES_DONNEES))
    sortie = charger_modele_en_service(PRODUCTION).predict(test.drop(columns=[CIBLE]))
    resultats = mesurer(test[CIBLE], sortie["classe"])
    afficher("Modèle en production, jeu de test réel", resultats)
    assert not verifier_seuils_qualite(resultats)
    assert resultats["Nb erreurs critiques"] == 4   # chiffre du notebook (section 6.6)
