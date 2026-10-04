"""Explicabilité (contributions SHAP calculées par LightGBM), sur des données factices."""

import json

import numpy as np
import pytest

from cisia.decision import appliquer_regle
from cisia.explication import contributions, decision_de_la_regle, expliquer, importance_globale
from cisia.modele import construire_pipeline
from cisia.preparation import charger_donnees, decouper, nettoyer, separer_x_y


@pytest.fixture(scope="module")
def modele_et_test(donnees_factices):
    entrainement, test = decouper(charger_donnees(donnees_factices / "entrainement.csv"))
    X_train, y_train = separer_x_y(nettoyer(entrainement))
    X_test, _ = separer_x_y(nettoyer(test))
    return construire_pipeline().fit(X_train, y_train), X_test.head(40)


def test_additivite_shap(modele_et_test):
    # Valeur de base + somme des contributions = score brut du modèle, pour chaque usager et chaque classe
    modele, X = modele_et_test
    Xt, contribs, _ = contributions(modele, X)
    score_brut = modele.named_steps["modele"].booster_.predict(Xt, raw_score=True)
    assert np.allclose(contribs.sum(axis=2), score_brut, atol=1e-6)


def test_explication_de_la_classe_retenue(modele_et_test):
    modele, X = modele_et_test
    classes = modele.predict(X)
    explications = expliquer(modele, X, classes)
    Xt, _, _ = contributions(modele, X)
    score_brut = modele.named_steps["modele"].booster_.predict(Xt, raw_score=True)

    json.dumps(explications)                                   # sérialisable : journal et API
    for i, explication in enumerate(explications):
        classe = int(classes[i])
        assert explication["classe_expliquee"] == classe
        noms = [f["facteur"].split(" (non renseigné")[0] for f in explication["facteurs"]]
        assert sorted(noms) == sorted(["Âge", "Ancienneté dans le dernier poste", "Niveau de diplôme",
                                       "Métier visé (domaine)", "Synthèse de l'entretien"])
        # Regroupement sans perte : base + facteurs = score brut de la classe (à l'arrondi près)
        total = explication["valeur_de_base"] + sum(f["contribution"] for f in explication["facteurs"])
        assert abs(total - score_brut[i, classe]) < 1e-3
        # Facteurs triés par effet décroissant ; mots tirés de la synthèse
        effets = [abs(f["contribution"]) for f in explication["facteurs"]]
        assert effets == sorted(effets, reverse=True)
        synthese = X.iloc[i]["synthese_entretien"].lower()
        assert all(m["mot"] in synthese for m in explication["mots"])


def test_information_manquante_signalee(modele_et_test):
    modele, X = modele_et_test
    X = X.head(1).copy()
    X["age"] = np.nan
    explication = expliquer(modele, X, [2])[0]
    assert "Âge (non renseigné)" in [f["facteur"] for f in explication["facteurs"]]


def test_importance_globale(modele_et_test):
    modele, X = modele_et_test
    importance = importance_globale(modele, X)
    assert len(importance) == 5 and all(v >= 0 for v in importance.values())
    assert list(importance.values()) == sorted(importance.values(), reverse=True)


def test_decision_de_la_regle_conforme_a_la_regle():
    # Le motif affiché correspond exactement à ce que fait appliquer_regle(), y compris le garde-fou
    regle = {"seuil_classe_2": 0.25, "seuil_garde_fou_classe_0": 0.08}
    generateur = np.random.default_rng(0)
    probas, risques = generateur.dirichlet([1, 1, 1], 2000), generateur.uniform(0, 0.5, 2000)
    classes = appliquer_regle(probas, risques, 0.25, 0.08)
    sans_garde_fou = appliquer_regle(probas, risques, 0.25, None)
    for p, r, c, c0 in zip(probas, risques, classes, sans_garde_fou):
        motif = decision_de_la_regle(p, r, int(c), regle)["motif"]
        assert (motif == "garde_fou") == (c == 1 and c0 == 0)
        assert (motif == "seuil_classe_2") == (c == 2)
    # Cas relevé par la relecture : le modèle penche pour la classe 0, le garde-fou relève en classe 1
    exemple = decision_de_la_regle([0.5421, 0.2915, 0.1664], 0.1016, 1, regle)
    assert exemple["motif"] == "garde_fou" and exemple["classe_la_plus_probable"] == 0
