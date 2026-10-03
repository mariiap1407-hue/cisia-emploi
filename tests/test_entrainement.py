"""Entraînement complet, sauvegarde et quality gate, sur des données factices."""

from pathlib import Path

import numpy as np
import pytest

from cisia import artefacts
from cisia.decision import charger_regle
from cisia.entrainement import entrainer, evaluer, predire
from cisia.evaluation import taux_erreurs_critiques, verifier_seuils_qualite
from cisia.inference import predire_usagers
from cisia.preparation import charger_donnees, decouper, nettoyer, separer_x_y

REGLE = charger_regle(Path(__file__).resolve().parents[1] / "config" / "regle_decision.json")


def preparer(chemin):
    entrainement, test = decouper(charger_donnees(chemin))
    return (*separer_x_y(nettoyer(entrainement)), *separer_x_y(nettoyer(test)))


def test_entrainer_predire_evaluer(donnees_factices):
    X_train, y_train, X_test, y_test = preparer(donnees_factices / "entrainement.csv")
    modele, correction = entrainer(X_train, y_train, REGLE)
    probas, risque, classes = predire(modele, correction, REGLE, X_test)

    assert probas.shape == (len(X_test), 3)
    assert ((risque >= 0) & (risque <= 1)).all()
    assert set(np.unique(classes)) <= {0, 1, 2}
    resultats = evaluer(y_test, probas, risque, classes)
    assert 0 <= resultats["F1 macro"] <= 1


def test_sauvegarde_puis_chargement(donnees_factices, tmp_path):
    X_train, y_train, X_test, _ = preparer(donnees_factices / "entrainement.csv")
    modele, correction = entrainer(X_train, y_train, REGLE)
    artefacts.sauvegarder(tmp_path, modele, correction, REGLE, {"run_id": "test"})

    charge = artefacts.charger(tmp_path)
    assert charge["regle"] == REGLE
    assert charge["infos"]["run_id"] == "test"
    assert np.allclose(charge["modele"].predict_proba(X_test), modele.predict_proba(X_test))


def test_quality_gate():
    bon = {"Erreurs critiques (2→0)": 0.044, "Rappel classe 2": 0.656, "F1 macro": 0.670}
    assert verifier_seuils_qualite(bon) == []
    mauvais = {"Erreurs critiques (2→0)": 0.286, "Rappel classe 2": 0.464, "F1 macro": 0.617}
    assert len(verifier_seuils_qualite(mauvais)) == 3


def test_quality_gate_refuse_un_indicateur_non_calculable():
    # Sans usager de classe 2, le taux d'erreurs critiques n'est pas défini : ce n'est pas un succès
    assert np.isnan(taux_erreurs_critiques([0, 1, 1], [0, 0, 1]))
    resultats = {"Erreurs critiques (2→0)": np.nan, "Rappel classe 2": 0.7, "F1 macro": 0.7}
    assert len(verifier_seuils_qualite(resultats)) == 1


def test_composants_melanges_detectes(donnees_factices, tmp_path):
    # Deux entraînements différents : on remplace la correction du premier par celle du second
    X_train, y_train, _, _ = preparer(donnees_factices / "entrainement.csv")
    modele, correction = entrainer(X_train, y_train, REGLE)
    _, autre_correction = entrainer(X_train.iloc[:600], y_train.iloc[:600], REGLE)
    artefacts.sauvegarder(tmp_path / "a", modele, correction, REGLE, {})
    artefacts.sauvegarder(tmp_path / "b", modele, autre_correction, REGLE, {})
    autre_fichier = (tmp_path / "b" / "recalibration.joblib").read_bytes()
    (tmp_path / "a" / "recalibration.joblib").write_bytes(autre_fichier)
    with pytest.raises(ValueError):
        artefacts.charger(tmp_path / "a")


def test_predire_usagers_depuis_donnees_brutes(donnees_factices, tmp_path):
    X_train, y_train, _, _ = preparer(donnees_factices / "entrainement.csv")
    modele, correction = entrainer(X_train, y_train, REGLE)
    composants = {"modele": modele, "correction": correction, "regle": REGLE}
    reference = charger_donnees(donnees_factices / "reference.csv")
    bruts = reference.drop(columns=["classe_retour_emploi"]).head(20)
    sortie = predire_usagers(composants, bruts)
    assert list(sortie.columns) == ["classe", "recommandation", "niveau_alerte", "risque_longue_duree"]
    assert len(sortie) == 20
    assert sortie.loc[sortie["classe"] == 2, "niveau_alerte"].eq("Élevé").all()
