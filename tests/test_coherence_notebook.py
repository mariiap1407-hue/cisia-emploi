"""Le code de src/cisia/ retrouve les chiffres du notebook sur les VRAIES données.

Ce test ne tourne que sur le poste local, là où se trouve le vrai fichier.
Dans la CI (GitHub Actions), le fichier n'existe pas : le test est ignoré (« skipped »).
Chiffres attendus : sorties du notebook CISIA_emploi8 (sections 5.6, 6.1 et 6.6).
(Les tests sur la règle retenue entraînent le modèle complet : compter une à deux minutes.)
"""

from pathlib import Path

import pytest
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_predict

from cisia import artefacts
from cisia.decision import charger_regle
from cisia.entrainement import entrainer, evaluer, predire
from cisia.evaluation import CV, mesurer
from cisia.modele import construire_pipeline
from cisia.preparation import charger_donnees, decouper, nettoyer, separer_x_y

VRAIES_DONNEES = Path(__file__).resolve().parents[1] / "data" / "raw" / "dataset_trajectoire_emploi.csv"

pytestmark = pytest.mark.skipif(not VRAIES_DONNEES.exists(), reason="vraies données absentes (CI)")

# Le notebook affiche les chiffres arrondis à 3 décimales
PRECISION = 0.0005


@pytest.fixture(scope="module")
def donnees():
    entrainement, test = decouper(charger_donnees(VRAIES_DONNEES))
    X_train, y_train = separer_x_y(nettoyer(entrainement))
    X_test, y_test = separer_x_y(nettoyer(test))
    return X_train, y_train, X_test, y_test


def test_decoupage_identique(donnees):
    X_train, _, X_test, y_test = donnees
    assert len(X_train) == 2000
    assert len(X_test) == 500
    assert (y_test == 2).sum() == 90


def test_validation_croisee_regle_par_defaut(donnees):
    """Section 5.6 / 6.1 : probabilités hors pli, classe la plus probable."""
    X_train, y_train, _, _ = donnees
    probas = cross_val_predict(clone(construire_pipeline()), X_train, y_train, cv=CV, method="predict_proba")
    resultats = mesurer(y_train, probas.argmax(axis=1))
    assert resultats["F1 macro"] == pytest.approx(0.681, abs=PRECISION)
    assert resultats["Rappel classe 2"] == pytest.approx(0.707, abs=PRECISION)
    assert resultats["Précision classe 2"] == pytest.approx(0.544, abs=PRECISION)
    assert resultats["Nb erreurs critiques"] == 47
    assert resultats["Nb fausses alertes classe 2"] == 215


def test_jeu_de_test_regle_par_defaut(donnees):
    """Section 6.6 : modèle entraîné sur tout le jeu d'entraînement, classe la plus probable."""
    X_train, y_train, X_test, y_test = donnees
    probas = construire_pipeline().fit(X_train, y_train).predict_proba(X_test)
    y_pred = probas.argmax(axis=1)
    resultats = mesurer(y_test, y_pred)
    assert accuracy_score(y_test, y_pred) == pytest.approx(0.706, abs=PRECISION)
    assert resultats["F1 macro"] == pytest.approx(0.688, abs=PRECISION)
    assert resultats["Rappel classe 2"] == pytest.approx(0.644, abs=PRECISION)
    assert resultats["Nb erreurs critiques"] == 9
    assert resultats["Nb fausses alertes classe 2"] == 47
    assert roc_auc_score(y_test == 2, probas[:, 2]) == pytest.approx(0.866, abs=PRECISION)


def test_jeu_de_test_regle_retenue(donnees, tmp_path):
    """Section 6.6 : toute la chaîne de décision, des données brutes à la classe retenue.

    Données brutes -> nettoyage -> probabilités du modèle -> recalibration -> règle (seuils 0,25 et 0,08).
    Puis sauvegarde et rechargement : les prédictions doivent être identiques.
    """
    X_train, y_train, X_test, y_test = donnees
    regle = charger_regle(VRAIES_DONNEES.parents[2] / "config" / "regle_decision.json")
    modele, correction = entrainer(X_train, y_train, regle)
    probas, risque, classes = predire(modele, correction, regle, X_test)
    resultats = evaluer(y_test, probas, risque, classes)

    # Matrice de confusion du notebook (section 6.6, règle retenue)
    assert confusion_matrix(y_test, classes).tolist() == [[113, 65, 9], [20, 169, 34], [4, 27, 59]]
    assert resultats["Accuracy"] == pytest.approx(0.682, abs=PRECISION)
    assert resultats["F1 macro"] == pytest.approx(0.670, abs=PRECISION)
    assert resultats["Rappel classe 2"] == pytest.approx(0.656, abs=PRECISION)
    assert resultats["Précision classe 2"] == pytest.approx(0.578, abs=PRECISION)
    assert resultats["Nb erreurs critiques"] == 4
    assert resultats["Nb fausses alertes classe 2"] == 43
    assert resultats["Brier classe 2 (brut)"] == pytest.approx(0.102, abs=PRECISION)
    assert resultats["Brier classe 2 (recalibré)"] == pytest.approx(0.097, abs=PRECISION)

    # Sauvegarde puis rechargement : mêmes classes et même risque
    artefacts.sauvegarder(tmp_path, modele, correction, regle, {"run_id": "test"})
    charge = artefacts.charger(tmp_path)
    _, risque_recharge, classes_rechargees = predire(charge["modele"], charge["correction"], charge["regle"],
                                                     X_test)
    assert (classes_rechargees == classes).all()
    assert abs(risque_recharge - risque).max() < 1e-12
