"""Le code de src/cisia/ retrouve les chiffres du notebook sur les VRAIES données.

Ce test ne tourne que sur le poste local, là où se trouve le vrai fichier.
Dans la CI (GitHub Actions), le fichier n'existe pas : le test est ignoré (« skipped »).
Chiffres attendus : sorties du notebook CISIA_emploi8 (sections 5.6, 6.1 et 6.6).
"""

from pathlib import Path

import pytest
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_predict

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
