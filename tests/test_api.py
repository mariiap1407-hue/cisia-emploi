"""L'API de bout en bout, avec un modèle de contrôle (données factices), une base et une clé temporaires."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.journal import Journal
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
    # Explication SHAP de la classe retenue, conservée aussi dans l'historique
    explication = resultat["explication"]
    assert explication["classe_expliquee"] == resultat["classe"]
    assert len(explication["facteurs"]) == 5
    assert explication["regle"]["motif"] in ("seuil_classe_2", "garde_fou", "plus_probable_0_1")
    assert explication["regle"]["risque_recalibre"] == resultat["risque_longue_duree"]
    sorties = client.get("/history", headers=ENTETE).json()[0]["sorties"]
    assert sorties["explication"] == explication


def test_prediction_avec_informations_manquantes(client):
    usager = {"code_rome_vise": "K2204", "synthese_entretien": "Recherche en cours."}
    reponse = predire(client, usager)
    assert reponse.status_code == 200
    facteurs = [f["facteur"] for f in reponse.json()["explication"]["facteurs"]]
    assert "Âge (non renseigné)" in facteurs and "Niveau de diplôme (non renseigné)" in facteurs


def test_cle_avec_accents_refusee_proprement(client):
    # Une clé reçue avec des caractères non ASCII est une clé fausse : 401, pas une erreur 500
    entete = {"X-API-Key": "clé-fausse".encode("latin-1")}
    assert client.get("/history", headers=entete).status_code == 401


def test_sante_sans_cle_configuree(production, tmp_path):
    # Modèle chargé mais aucune clé : /predict refuserait tout, donc le service n'est pas prêt
    with TestClient(creer_application(production, tmp_path / "journal.db", cle_api="")) as client:
        reponse = client.get("/health")
        assert reponse.status_code == 503
        assert reponse.json()["modele_charge"] is True and reponse.json()["cle_configuree"] is False


def test_entree_invalide_refusee_et_journalisee_sans_valeur(client):
    reponse = predire(client, {**USAGER, "age": -5}, id_session="S-3")
    assert reponse.status_code == 422
    # Champ en cause sans le préfixe technique « body », message en français, sans la valeur reçue
    assert reponse.json()["detail"] == [{"champ": ["usager", "age"], "type": "greater_than_equal",
                                         "message": "Doit être supérieur ou égal à 16."}]
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


def test_identifiant_usager_pseudonyme(client):
    corps = {"usager": USAGER, "id_usager": "DE-0248"}
    id_prediction = client.post("/predict", json=corps, headers=ENTETE).json()["id_prediction"]
    dernier = client.get("/history?limite=1", headers=ENTETE).json()[0]
    assert dernier["id_prediction"] == id_prediction and dernier["id_usager"] == "DE-0248"
    # Un nom (espace, accents) n'est pas un identifiant pseudonyme : refusé
    refus = client.post("/predict", json={"usager": USAGER, "id_usager": "Jean Dupont"}, headers=ENTETE)
    assert refus.status_code == 422 and refus.json()["detail"][0]["champ"] == ["id_usager"]


def test_avis_du_conseiller_distinct_du_feedback(client):
    reponse = predire(client)
    id_prediction, classe = reponse.json()["id_prediction"], reponse.json()["classe"]

    def envoyer(corps):
        return client.post("/avis", json={"id_prediction": id_prediction, **corps}, headers=ENTETE)

    confirme = envoyer({"avis": "confirme"})
    assert confirme.status_code == 201 and confirme.json()["classe_proposee"] == classe
    assert envoyer({"avis": "corrige", "classe_proposee": 1}).status_code == 422   # motif manquant
    corrige = envoyer({"avis": "corrige", "classe_proposee": 1, "motif": "Autre", "precisions": "Formation"})
    assert corrige.status_code == 201 and corrige.json()["remplace_un_avis_precedent"] is True
    inconnue = client.post("/avis", json={"id_prediction": "inconnue", "avis": "confirme"}, headers=ENTETE)
    assert inconnue.status_code == 404

    element = client.get("/history?limite=1", headers=ENTETE).json()[0]
    avis_enregistre = element["avis_conseiller"]
    assert avis_enregistre["avis"] == "corrige" and avis_enregistre["classe_proposee"] == 1
    assert avis_enregistre["motif"] == "Autre" and avis_enregistre["precisions"] == "Formation"   # relisible
    assert element["situation_observee"] is None   # l'avis n'est PAS une situation observée
    client.post("/feedback", json={"id_prediction": id_prediction, "classe_reelle": 2}, headers=ENTETE)
    element = client.get("/history?limite=1", headers=ENTETE).json()[0]
    assert element["situation_observee"]["classe_reelle"] == 2


# --- Réentraînement (/retrain) ---------------------------------------------------------------

@pytest.fixture
def production_copiee(production, tmp_path):
    """Copie de la production de contrôle : un test peut la modifier sans gêner les autres.

    Le dossier ne s'appelle pas « production » : on vérifie que le script publie bien dans le
    dossier configuré pour l'API (CISIA_PRODUCTION), quel que soit son nom."""
    copie = tmp_path / "models" / "en_service"
    shutil.copytree(production, copie)
    return copie


def chemins_du_candidat_de_controle(production):
    return {nom: production.parent / "candidat" / fichier for nom, fichier in artefacts.FICHIERS.items()}


def faux_lanceur(production, quality_gate_ok, recu, comparaison=None):
    """Remplace scripts/entrainer.py : écrit le candidat (et le met en service si le gate passe et,
    s'il y a une comparaison au modèle en production, si le candidat est meilleur)."""
    def lanceur(chemin_feedbacks, donnees, dossier_modeles, dossier_production, promouvoir):
        recu["feedbacks"] = charger_donnees(chemin_feedbacks)
        recu["dossier_production"] = dossier_production
        candidat = dossier_modeles / "candidats" / "run-test"
        candidat.mkdir(parents=True)
        infos = {"run_id": "run-test", "quality_gate_ok": quality_gate_ok,
                 "echecs_quality_gate": [] if quality_gate_ok else ["F1 macro = 0.300 (seuil min : 0.62)"],
                 "resultats_test": {"F1 macro": 0.7 if quality_gate_ok else 0.3},
                 "comparaison_production": comparaison}
        (candidat / "infos_entrainement.json").write_text(json.dumps(infos), encoding="utf-8")
        if quality_gate_ok and promouvoir and (comparaison is None or comparaison["meilleur"]):
            publier_en_production(chemins_du_candidat_de_controle(production), dossier_production,
                                  "run-test", {"run_id": "run-test", "version_registre": "2"})
        return (0 if quality_gate_ok else 1), "Run MLflow : run-test"
    return lanceur


def application_reentrainement(production, production_copiee, tmp_path, donnees_factices, lanceur):
    return creer_application(production_copiee, tmp_path / "journal.db", cle_api=CLE, lanceur=lanceur,
                             donnees_entrainement=donnees_factices / "entrainement.csv")


def prediction_avec_feedback(client, classe=2):
    id_prediction = predire(client).json()["id_prediction"]
    client.post("/feedback", json={"id_prediction": id_prediction, "classe_reelle": classe}, headers=ENTETE)
    return id_prediction


def test_reentrainement_sans_feedback_refuse(production, production_copiee, tmp_path, donnees_factices):
    application = application_reentrainement(production, production_copiee, tmp_path, donnees_factices,
                                             faux_lanceur(production, True, {}))
    with TestClient(application) as client:
        assert client.post("/retrain", headers=ENTETE).status_code == 409
        assert client.post("/retrain").status_code == 401


def test_reentrainement_refuse_par_le_quality_gate(production, production_copiee, tmp_path, donnees_factices):
    recu = {}
    application = application_reentrainement(production, production_copiee, tmp_path, donnees_factices,
                                             faux_lanceur(production, False, recu))
    with TestClient(application) as client:
        id_prediction = prediction_avec_feedback(client, classe=2)
        reponse = client.post("/retrain", headers=ENTETE)
        assert reponse.status_code == 200
        resultat = reponse.json()
        assert resultat["statut"] == "refuse_quality_gate" and resultat["echecs_quality_gate"]
        assert resultat["version_modele"] == resultat["version_modele_avant"] == "controle"
        assert client.get("/health").json()["version_modele"] == "controle"   # production inchangée

    # Le script a reçu le feedback au format du fichier d'origine, avec la classe OBSERVÉE
    feedbacks = recu["feedbacks"]
    assert len(feedbacks) == 1 and feedbacks.loc[0, "usager_id"] == f"FEEDBACK_{id_prediction}"
    assert feedbacks.loc[0, CIBLE] == 2 and feedbacks.loc[0, "code_rome_vise"] == "M1607"
    assert feedbacks["nationalite_hors_ue"].isna().all()   # jamais collectée
    assert recu["dossier_production"] == production_copiee   # le dossier de l'API, quel que soit son nom


def test_candidat_pas_meilleur_que_la_production_non_promu(production, production_copiee, tmp_path,
                                                          donnees_factices):
    """Champion / challenger : quality gate respecté, mais candidat moins bon que le modèle en place."""
    comparaison = {"meilleur": False, "motif": "Candidat moins bon que le modèle en production — "
                   "Nb erreurs critiques : 5 contre 4.", "criteres": []}
    application = application_reentrainement(production, production_copiee, tmp_path, donnees_factices,
                                             faux_lanceur(production, True, {}, comparaison))
    with TestClient(application) as client:
        prediction_avec_feedback(client)
        resultat = client.post("/retrain", headers=ENTETE).json()
        assert resultat["statut"] == "non_promu_pas_meilleur"
        assert resultat["comparaison_production"]["meilleur"] is False
        assert resultat["version_modele"] == "controle"                      # le modèle en place reste
    trace, = Journal(tmp_path / "journal.db").reentrainements()
    assert trace["statut"] == "non_promu_pas_meilleur"


def test_reentrainement_mis_en_production(production, production_copiee, tmp_path, donnees_factices):
    application = application_reentrainement(production, production_copiee, tmp_path, donnees_factices,
                                             faux_lanceur(production, True, {}))
    with TestClient(application) as client:
        prediction_avec_feedback(client)
        resultat = client.post("/retrain", json={"promouvoir": True}, headers=ENTETE).json()
        assert resultat["statut"] == "mis_en_production"
        assert (resultat["version_modele_avant"], resultat["version_modele"]) == ("controle", "2")
        # Le nouveau modèle est chargé sans redémarrer l'API
        assert client.get("/health").json()["version_modele"] == "2"
        assert predire(client).json()["version_modele"] == "2"
    traces = Journal(tmp_path / "journal.db").reentrainements()
    assert [t["statut"] for t in traces] == ["mis_en_production"]


def test_echec_du_chargement_retour_a_la_version_precedente(production, production_copiee, tmp_path,
                                                          donnees_factices):
    """Le script publie un modèle impossible à charger : l'API revient à la version précédente."""
    lanceur_normal = faux_lanceur(production, True, {})

    def lanceur_defectueux(chemin_feedbacks, donnees, dossier_modeles, dossier_production, promouvoir):
        resultat = lanceur_normal(chemin_feedbacks, donnees, dossier_modeles, dossier_production, promouvoir)
        shutil.rmtree(dossier_production / "run-test")   # le modèle publié a disparu : chargement impossible
        return resultat

    application = application_reentrainement(production, production_copiee, tmp_path, donnees_factices,
                                             lanceur_defectueux)
    with TestClient(application) as client:
        prediction_avec_feedback(client)
        reponse = client.post("/retrain", headers=ENTETE)
        assert reponse.status_code == 500 and "controle reste en service" in reponse.json()["detail"]
        assert client.get("/health").json()["version_modele"] == "controle"   # toujours servie
        assert predire(client).status_code == 200
    pointeur = json.loads((production_copiee / "actuelle.json").read_text(encoding="utf-8"))
    assert pointeur["run_id"] == "controle"   # le pointeur sur disque est rétabli
    trace, = Journal(tmp_path / "journal.db").reentrainements()
    assert trace["statut"] == "erreur_activation"
    # Retour arrière tracé de façon structurée pour le suivi (B10) ; pas de version du registre ici
    assert trace["resultats"]["retour_arriere"] == {"necessaire": True, "pointeur_retabli": True,
                                                    "alias_retabli": None, "verifie": True}


def test_echec_apres_publication_retour_a_la_version_precedente(production, production_copiee, tmp_path,
                                                               donnees_factices):
    """Le script publie le nouveau modèle PUIS échoue (délai dépassé) : le pointeur est rétabli,
    sinon un simple redémarrage de l'API activerait un modèle dont la mise en service a échoué."""
    lanceur_normal = faux_lanceur(production, True, {})

    def lanceur_qui_expire(chemin_feedbacks, donnees, dossier_modeles, dossier_production, promouvoir):
        lanceur_normal(chemin_feedbacks, donnees, dossier_modeles, dossier_production, promouvoir)
        raise subprocess.TimeoutExpired("entrainer.py", 1800)

    application = application_reentrainement(production, production_copiee, tmp_path, donnees_factices,
                                             lanceur_qui_expire)
    with TestClient(application) as client:
        prediction_avec_feedback(client)
        reponse = client.post("/retrain", headers=ENTETE)
        assert reponse.status_code == 500 and "controle reste en service" in reponse.json()["detail"]
    pointeur = json.loads((production_copiee / "actuelle.json").read_text(encoding="utf-8"))
    assert pointeur["run_id"] == "controle"   # un redémarrage rechargerait bien l'ancienne version
    trace, = Journal(tmp_path / "journal.db").reentrainements()
    assert trace["statut"] == "erreur"
    assert trace["resultats"]["retour_arriere"]["pointeur_retabli"] is True
    assert trace["resultats"]["retour_arriere"]["verifie"] is True


def test_aucun_feedback_utilisable(production, production_copiee, tmp_path, donnees_factices):
    """Tous les feedbacks écartés par le script (profil déjà dans le test) : 409, pas de réentraînement."""
    def lanceur(chemin_feedbacks, donnees, dossier_modeles, dossier_production, promouvoir):
        return 2, "Aucun feedback utilisable : tous ont un profil déjà présent dans le jeu de test."

    application = application_reentrainement(production, production_copiee, tmp_path, donnees_factices,
                                             lanceur)
    with TestClient(application) as client:
        prediction_avec_feedback(client)
        assert client.post("/retrain", headers=ENTETE).status_code == 409


def test_version_coherente_si_le_modele_change_pendant_une_prediction(client):
    """Une prédiction commencée avec l'ancien modèle annonce l'ancienne version, même si le modèle
    est remplacé pendant le calcul (réentraînement concurrent)."""
    application = client.app
    modele, infos = application.state.service

    class ModeleRemplaceEnCours:
        def predict(self, tableau):
            application.state.service = (modele, {**infos, "version_registre": "nouvelle"})
            return modele.predict(tableau)

    application.state.service = (ModeleRemplaceEnCours(), {**infos, "version_registre": "ancienne"})
    reponse = predire(client).json()
    assert reponse["version_modele"] == "ancienne"
    assert client.get("/history?limite=1", headers=ENTETE).json()[0]["version_modele"] == "ancienne"


def test_reentrainement_avec_le_vrai_script(production, production_copiee, tmp_path, donnees_factices,
                                           monkeypatch):
    """Le vrai script, sur données factices, sans mise en production (refusée pour des données factices)."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    application = creer_application(production_copiee, tmp_path / "journal.db", cle_api=CLE,
                                    donnees_entrainement=donnees_factices / "entrainement.csv")
    with TestClient(application) as client:
        prediction_avec_feedback(client)
        reponse = client.post("/retrain", json={"promouvoir": False}, headers=ENTETE)
        assert reponse.status_code == 200, reponse.text
        resultat = reponse.json()
        assert resultat["statut"] in ("candidat_non_promu", "refuse_quality_gate", "non_promu_pas_meilleur")
        assert resultat["n_feedbacks"] == 1 and "F1 macro" in resultat["resultats_test"]
        assert resultat["version_modele"] == "controle"   # pas de mise en production demandée
    infos = json.loads((production_copiee.parent / "candidats" / resultat["run_id"]
                        / "infos_entrainement.json").read_text(encoding="utf-8"))
    assert infos["n_feedbacks"] == 1
    # Champion / challenger : le vrai script a comparé le candidat au modèle en production (même jeu de test)
    criteres = [ligne["critere"] for ligne in infos["comparaison_production"]["criteres"]]
    assert criteres == ["Nb erreurs critiques", "Rappel classe 2", "F1 macro"]


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


def test_rapport_de_suivi(production, tmp_path):
    dossier_suivi = tmp_path / "suivi"
    application = creer_application(production, tmp_path / "journal.db", cle_api=CLE,
                                    dossier_suivi=dossier_suivi)
    with TestClient(application) as client:
        assert client.get("/suivi").status_code == 401
        assert client.get("/suivi", headers=ENTETE).status_code == 404        # pas encore de rapport
        dossier_suivi.mkdir()
        rapport = {"date": "2026-10-05T08:00:00+00:00", "alertes": [], "indicateurs": {},
                   "derive_par_semaine": []}
        (dossier_suivi / "dernier_rapport.json").write_text(json.dumps(rapport), encoding="utf-8")
        assert client.get("/suivi", headers=ENTETE).json() == rapport
