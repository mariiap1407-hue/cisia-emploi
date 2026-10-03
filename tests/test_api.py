"""L'API de bout en bout, avec un modèle de contrôle (données factices), une base et une clé temporaires."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import creer_application
from cisia import artefacts
from cisia.decision import charger_regle
from cisia.entrainement import entrainer
from cisia.modele_mlflow import charger_modele_en_service, publier_en_production
from cisia.preparation import CIBLE, charger_donnees, decouper, nettoyer, separer_x_y

RACINE = Path(__file__).resolve().parents[1]
CLE = "cle-de-test"
ENTETE = {"X-API-Key": CLE}
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
    with TestClient(creer_application(production, tmp_path / "journal.db", cle_api=CLE)) as client:
        yield client


def predire(client, usager=USAGER, id_session=None):
    return client.post("/predict", json={"usager": usager, "id_session": id_session}, headers=ENTETE)


def test_sante_publique(client):
    reponse = client.get("/health")   # sans clé : sert aux sondes de disponibilité (Docker)
    assert reponse.status_code == 200
    assert reponse.json()["modele_charge"] is True


def test_cle_obligatoire(client):
    assert client.post("/predict", json={"usager": USAGER}).status_code == 401
    assert client.post("/predict", json={"usager": USAGER}, headers={"X-API-Key": "faux"}).status_code == 401
    assert client.get("/history").status_code == 401


def test_prediction(client):
    reponse = predire(client, id_session="S-1")
    assert reponse.status_code == 200
    resultat = reponse.json()
    assert resultat["classe"] in (0, 1, 2)
    assert 0 <= resultat["risque_longue_duree"] <= 1
    assert resultat["niveau_alerte"] in ("Faible", "Modéré", "Élevé")
    assert (resultat["classe"] == 2) == (resultat["niveau_alerte"] == "Élevé")


def test_prediction_avec_informations_manquantes(client):
    usager = {"code_rome_vise": "K2204", "synthese_entretien": "Recherche en cours."}
    assert predire(client, usager).status_code == 200


def test_entree_invalide_refusee_et_journalisee_sans_valeur(client):
    reponse = predire(client, {**USAGER, "age": -5}, id_session="S-3")
    assert reponse.status_code == 422
    assert reponse.json()["detail"][0]["champ"][-1] == "age"   # le champ en cause est indiqué
    dernier = client.get("/history", headers=ENTETE).json()[0]
    assert dernier["statut"] == "invalide" and dernier["id_session"] == "S-3"
    assert dernier["entrees"] is None   # aucune valeur d'une requête refusée n'est conservée


def test_nombre_infini_refuse_et_historique_intact(client):
    # 1e999 est lu comme l'infini : il doit être refusé (422) sans casser l'historique
    texte = json.dumps({"usager": USAGER}).replace('"age": 35', '"age": 1e999')
    reponse = client.post("/predict", content=texte, headers={**ENTETE, "Content-Type": "application/json"})
    assert reponse.status_code == 422
    assert client.get("/history", headers=ENTETE).status_code == 200


def test_donnee_non_demandee_jamais_conservee(client):
    reponse = predire(client, {**USAGER, "nationalite_hors_ue": 1})
    assert reponse.status_code == 422
    historique = json.dumps(client.get("/history", headers=ENTETE).json())
    assert "nationalite_hors_ue\": 1" not in historique and "\"input\"" not in historique


def test_historique_et_feedback(client):
    id_prediction = predire(client, id_session="S-2").json()["id_prediction"]
    dernier = client.get("/history?limite=1", headers=ENTETE).json()[0]
    assert dernier["id_prediction"] == id_prediction
    assert dernier["entrees"]["code_rome_vise"] == "M1607" and dernier["duree_ms"] > 0

    def envoyer(identifiant, classe):
        corps = {"id_prediction": identifiant, "classe_reelle": classe}
        return client.post("/feedback", json=corps, headers=ENTETE)

    premier = envoyer(id_prediction, 0)
    assert premier.status_code == 201 and premier.json()["remplace_un_feedback_precedent"] is False
    # Correction du conseiller : le nouveau feedback remplace le précédent
    assert envoyer(id_prediction, 2).json()["remplace_un_feedback_precedent"] is True
    assert envoyer("inconnu", 2).status_code == 404   # prédiction qui n'existe pas
    assert envoyer(id_prediction, 5).status_code == 422   # classe impossible


def test_sans_modele_en_production(tmp_path):
    with TestClient(creer_application(tmp_path / "vide", tmp_path / "journal.db", cle_api=CLE)) as client:
        assert client.get("/health").status_code == 503   # pas prête : aucun modèle chargé
        assert predire(client).status_code == 503


def pd_manquant(valeur):
    """Valeur manquante dans le fichier (None ou NaN) : envoyée vide à l'API."""
    return valeur is None or (isinstance(valeur, float) and valeur != valeur)


VRAIES_DONNEES = RACINE / "data" / "raw" / "dataset_trajectoire_emploi.csv"
PRODUCTION = RACINE / "models" / "production"


@pytest.mark.skipif(not (VRAIES_DONNEES.exists() and (PRODUCTION / "actuelle.json").exists()),
                    reason="vraies données ou modèle en production absents (CI)")
def test_api_identique_au_modele_en_production(tmp_path):
    """Poste local : pour 50 usagers réels du jeu de test, l'API donne exactement la prédiction du modèle."""
    _, test = decouper(charger_donnees(VRAIES_DONNEES))
    test = test[test["synthese_entretien"].notna()].head(50)   # l'API exige une synthèse
    attendu = charger_modele_en_service(PRODUCTION).predict(test.drop(columns=[CIBLE]))
    champs = ["age", "niveau_diplome", "anciennete_poste_ans", "code_rome_vise", "synthese_entretien"]
    with TestClient(creer_application(PRODUCTION, tmp_path / "journal.db", cle_api=CLE)) as client:
        for (_, usager), (_, prevu) in zip(test[champs].iterrows(), attendu.iterrows()):
            corps = {champ: (None if pd_manquant(valeur) else valeur) for champ, valeur in usager.items()}
            resultat = predire(client, corps).json()
            assert resultat["classe"] == int(prevu["classe"])
            assert resultat["risque_longue_duree"] == round(float(prevu["risque_longue_duree"]), 4)
