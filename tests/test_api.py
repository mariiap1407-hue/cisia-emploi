"""L'API de bout en bout, avec un modèle de contrôle (données factices) et une base temporaires."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import creer_application
from cisia import artefacts
from cisia.decision import charger_regle
from cisia.entrainement import entrainer
from cisia.modele_mlflow import publier_en_production
from cisia.preparation import charger_donnees, decouper, nettoyer, separer_x_y

RACINE = Path(__file__).resolve().parents[1]

USAGER = {"age": 35, "niveau_diplome": "Bac", "anciennete_poste_ans": 4.5,
          "code_rome_vise": "M1607", "synthese_entretien": "Projet clair, mobilité possible."}


@pytest.fixture(scope="module")
def production(donnees_factices, tmp_path_factory):
    """Un modèle de contrôle mis en production dans un dossier temporaire."""
    dossier = tmp_path_factory.mktemp("api")
    entrainement, _ = decouper(charger_donnees(donnees_factices / "entrainement.csv"))
    X_train, y_train = separer_x_y(nettoyer(entrainement))
    regle = charger_regle(RACINE / "config" / "regle_decision.json")
    modele, correction = entrainer(X_train, y_train, regle)
    chemins = artefacts.sauvegarder(dossier / "candidat", modele, correction, regle, {})
    publier_en_production(chemins, dossier / "production", "controle", {"run_id": "controle"})
    return dossier / "production"


@pytest.fixture
def client(production, tmp_path):
    with TestClient(creer_application(production, tmp_path / "journal.db")) as client:
        yield client


def test_sante(client):
    reponse = client.get("/health")
    assert reponse.status_code == 200
    assert reponse.json()["modele_charge"] is True


def test_prediction(client):
    reponse = client.post("/predict", json={"usager": USAGER, "id_session": "S-1"})
    assert reponse.status_code == 200
    resultat = reponse.json()
    assert resultat["classe"] in (0, 1, 2)
    assert 0 <= resultat["risque_longue_duree"] <= 1
    assert resultat["niveau_alerte"] in ("Faible", "Modéré", "Élevé")
    assert (resultat["classe"] == 2) == (resultat["niveau_alerte"] == "Élevé")


def test_prediction_avec_informations_manquantes(client):
    usager = {"code_rome_vise": "K2204", "synthese_entretien": "Recherche en cours."}
    assert client.post("/predict", json={"usager": usager}).status_code == 200


def test_entree_invalide_refusee_et_journalisee(client):
    reponse = client.post("/predict", json={"usager": {**USAGER, "age": -5}})
    assert reponse.status_code == 422
    assert reponse.json()["detail"][0]["loc"][-1] == "age"   # le champ en cause est indiqué
    assert client.get("/history").json()[0]["statut"] == "invalide"


def test_historique_et_feedback(client):
    reponse = client.post("/predict", json={"usager": USAGER, "id_session": "S-2"})
    id_prediction = reponse.json()["id_prediction"]
    dernier = client.get("/history?limite=1").json()[0]
    assert dernier["id_prediction"] == id_prediction
    assert dernier["entrees"]["code_rome_vise"] == "M1607" and dernier["duree_ms"] > 0

    def envoyer(identifiant, classe):
        corps = {"id_prediction": identifiant, "classe_reelle": classe}
        return client.post("/feedback", json=corps).status_code

    assert envoyer(id_prediction, 2) == 201
    assert envoyer("inconnu", 2) == 404       # feedback sur une prédiction qui n'existe pas
    assert envoyer(id_prediction, 5) == 422   # classe impossible


def test_sans_modele_en_production(tmp_path):
    with TestClient(creer_application(tmp_path / "vide", tmp_path / "journal.db")) as client:
        assert client.get("/health").json()["modele_charge"] is False
        assert client.post("/predict", json={"usager": USAGER}).status_code == 503
