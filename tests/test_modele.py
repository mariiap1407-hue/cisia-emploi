"""Le pipeline du modèle s'entraîne et prédit sur des données factices."""

import numpy as np

from cisia.modele import construire_pipeline
from cisia.preparation import charger_donnees, decouper, nettoyer, separer_x_y


def test_pipeline_entraine_et_predit(donnees_factices):
    entrainement, test = decouper(charger_donnees(donnees_factices / "entrainement.csv"))
    X_train, y_train = separer_x_y(nettoyer(entrainement))
    X_test, _ = separer_x_y(nettoyer(test))

    modele = construire_pipeline().fit(X_train, y_train)
    probas = modele.predict_proba(X_test)

    assert probas.shape == (len(X_test), 3)
    assert np.allclose(probas.sum(axis=1), 1)


def test_reglages_retenus_appliques():
    reglages = construire_pipeline().named_steps["modele"].get_params()
    assert (reglages["num_leaves"], reglages["learning_rate"], reglages["n_estimators"]) == (7, 0.1, 100)
    assert reglages["class_weight"] == "balanced"
