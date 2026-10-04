"""Quality gate : test de non-régression sur un jeu de référence (même principe que le projet M5).

Le test calcule les indicateurs de la chaîne complète sur un jeu de référence et ÉCHOUE si un seuil
de la section 6.6 n'est pas respecté : erreurs critiques <= 10 %, rappel de la classe 2 >= 60 %,
F1 macro >= 0,62. Ces seuils d'acceptation sont plus larges que l'objectif d'optimisation de
l'étape 6 (6,5 %) : l'un sert à choisir une règle, l'autre à refuser un modèle devenu inacceptable.

Variables d'environnement (facultatives) :
- CISIA_MODELE : dossier des composants d'un modèle à accepter (ex. models/candidats/<run>) ;
  elle exige CISIA_REFERENCE, le jeu de référence adapté à ce modèle (même nature de données) ;
- sans CISIA_MODELE : un modèle de contrôle est entraîné sur les données factices et évalué sur
  data/factice/reference.csv (ou sur CISIA_REFERENCE si elle est donnée).
  Avec CISIA_REFERENCE=data/factice/reference_derivee.csv, le test d'acceptation doit échouer :
  c'est la démonstration que le gate de performance rejette une dégradation simulée.

Les scores sur données factices vérifient le comportement du logiciel ; ce ne sont pas des
estimations de la performance réelle.
"""

import json
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


def entrainer_modele_de_controle(donnees_factices):
    entrainement, _ = decouper(charger_donnees(donnees_factices / "entrainement.csv"))
    X_train, y_train = separer_x_y(nettoyer(entrainement))
    regle = charger_regle(RACINE / "config" / "regle_decision.json")
    modele, correction = entrainer(X_train, y_train, regle)
    return {"modele": modele, "correction": correction, "regle": regle}


@pytest.fixture(scope="module")
def modele_de_controle(donnees_factices):
    """Modèle entraîné sur les données factices : sert à la démonstration de dégradation."""
    return entrainer_modele_de_controle(donnees_factices)


@pytest.fixture(scope="module")
def modele_a_accepter(request):
    """Le modèle soumis au test d'acceptation : CISIA_MODELE s'il est donné, sinon le modèle de contrôle.

    Le modèle de contrôle n'est entraîné que s'il est nécessaire (pas quand on teste un candidat fourni).
    """
    if not os.getenv("CISIA_MODELE"):
        return request.getfixturevalue("modele_de_controle")
    if not os.getenv("CISIA_REFERENCE"):
        pytest.fail("CISIA_MODELE exige CISIA_REFERENCE : un modèle s'évalue sur un jeu de référence "
                    "de même nature que ses données d'entraînement.")
    return artefacts.charger(os.getenv("CISIA_MODELE"))


def indicateurs(composants, chemin_reference):
    """Prédictions de la chaîne complète sur le jeu de référence, puis indicateurs."""
    reference = charger_donnees(chemin_reference)
    sortie = predire_usagers(composants, reference.drop(columns=[CIBLE]))
    return mesurer(reference[CIBLE], sortie["classe"])


def afficher(titre, resultats):
    print(f"\n{titre}")
    for indicateur, (sens, seuil) in SEUILS_QUALITE.items():
        print(f"  {indicateur:<25} {resultats[indicateur]:.3f}   (seuil {sens} : {seuil})")


def test_quality_gate_sur_le_jeu_de_reference(modele_a_accepter):
    resultats = indicateurs(modele_a_accepter, REFERENCE)
    afficher(f"Jeu de référence : {REFERENCE.name}", resultats)
    echecs = verifier_seuils_qualite(resultats)
    if os.getenv("CISIA_RAPPORT_GATE"):   # CI : rapport conservé avec le modèle testé (traçabilité)
        rapport = {"modele": os.getenv("CISIA_MODELE", "modele de controle"), "reference": str(REFERENCE),
                   "reference_sha256": artefacts.empreinte(REFERENCE),
                   "resultats": {k: round(float(v), 4) for k, v in resultats.items()},
                   "seuils": {k: list(v) for k, v in SEUILS_QUALITE.items()},
                   "quality_gate_ok": not echecs, "echecs": echecs}
        chemin = Path(os.getenv("CISIA_RAPPORT_GATE"))
        chemin.parent.mkdir(parents=True, exist_ok=True)
        chemin.write_text(json.dumps(rapport, ensure_ascii=False, indent=2), encoding="utf-8")
    assert not echecs, "Quality gate non respecté : " + " ; ".join(echecs)


def test_la_degradation_est_detectee(modele_de_controle, donnees_factices):
    # Jeu dérivé : le lien entre le profil, la synthèse et la classe s'inverse pour les classes 0 et 2.
    # Le gate de performance doit rejeter le modèle de contrôle sur ce jeu (dégradation simulée).
    resultats = indicateurs(modele_de_controle, donnees_factices / "reference_derivee.csv")
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
    # Critères d'acceptation seulement : la reproduction exacte du notebook (4 erreurs critiques,
    # matrice de confusion) est vérifiée dans test_coherence_notebook.py
    echecs = verifier_seuils_qualite(resultats)
    assert not echecs, "Quality gate non respecté : " + " ; ".join(echecs)
