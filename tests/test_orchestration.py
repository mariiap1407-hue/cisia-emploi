"""Le script d'entraînement protège la production (relecture du bloc 3).

Chaque test lance le vrai script sur des données factices, dans un dossier temporaire,
avec une base MLflow temporaire : rien ne touche à models/ ni à mlflow.db du projet.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from cisia import artefacts
from cisia.decision import charger_regle
from cisia.entrainement import entrainer
from cisia.inference import predire_usagers
from cisia.modele_mlflow import charger_modele_en_service, dossier_en_service, publier_en_production
from cisia.preparation import charger_donnees, decouper, nettoyer, separer_x_y

RACINE = Path(__file__).resolve().parents[1]


def lancer(dossier, *options, donnees="data/factice/entrainement.csv"):
    environnement = {**os.environ, "MLFLOW_TRACKING_URI": f"sqlite:///{(dossier / 'mlflow.db').as_posix()}"}
    return subprocess.run(
        [sys.executable, "scripts/entrainer.py", "--donnees", str(donnees),
         "--sans-recherche", "--sortie", str(dossier / "models"), *options],
        cwd=RACINE, env=environnement, capture_output=True, encoding="utf-8",
    )


def test_quality_gate_non_respecte_ne_touche_pas_la_production(donnees_factices, tmp_path):
    # Une « production » existe déjà
    production = tmp_path / "models" / "production"
    production.mkdir(parents=True)
    (production / "marqueur.txt").write_text("version approuvée", encoding="utf-8")

    # Règle volontairement mauvaise : la classe 2 n'est jamais prédite, le rappel tombe à 0
    regle = json.loads((RACINE / "config" / "regle_decision.json").read_text(encoding="utf-8"))
    regle["seuil_classe_2"] = 1.0
    chemin_regle = tmp_path / "regle_mauvaise.json"
    chemin_regle.write_text(json.dumps(regle), encoding="utf-8")

    resultat = lancer(tmp_path, "--regle", str(chemin_regle))

    # Le script doit s'arrêter À CAUSE du quality gate (et non d'une autre erreur), avec le code 1
    assert "Quality gate NON respecté" in resultat.stdout, resultat.stdout + resultat.stderr
    assert resultat.returncode == 1
    assert (production / "marqueur.txt").read_text(encoding="utf-8") == "version approuvée"
    assert len(list((tmp_path / "models" / "candidats").iterdir())) == 1   # candidat gardé pour analyse


def test_donnees_factices_jamais_en_production(donnees_factices, tmp_path):
    resultat = lancer(tmp_path, "--promouvoir")
    assert "Refusé" in resultat.stderr, resultat.stdout + resultat.stderr
    assert resultat.returncode != 0
    assert not (tmp_path / "models" / "production").exists()


def test_modele_mlflow_enregistre_reproduit_la_chaine_complete(donnees_factices, tmp_path):
    resultat = lancer(tmp_path)
    assert resultat.returncode == 0, resultat.stdout + resultat.stderr
    assert not (tmp_path / "models" / "production").exists()   # pas de --promouvoir

    # Le modèle enregistré dans MLflow donne les mêmes décisions que les composants du candidat
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    version = max(mlflow.MlflowClient().search_model_versions("name='cisia-orientation-controle'"),
                  key=lambda v: int(v.version))
    modele_mlflow = mlflow.pyfunc.load_model(f"models:/cisia-orientation-controle/{version.version}")
    composants = artefacts.charger(next((tmp_path / "models" / "candidats").iterdir()))

    reference = charger_donnees(donnees_factices / "reference.csv")
    usagers = reference.drop(columns=["classe_retour_emploi"]).head(50)
    attendu = predire_usagers(composants, usagers)
    obtenu = modele_mlflow.predict(usagers)
    assert (obtenu["classe"].to_numpy() == attendu["classe"].to_numpy()).all()
    ecart = obtenu["risque_longue_duree"].to_numpy() - attendu["risque_longue_duree"].to_numpy()
    assert abs(ecart).max() < 1e-12


def test_mise_en_production_puis_changement_de_version(donnees_factices, tmp_path):
    # Chemin de mise en production (échec sous Windows lors de la première version : dossier renommé)
    entrainement, _ = decouper(charger_donnees(donnees_factices / "entrainement.csv"))
    X_train, y_train = separer_x_y(nettoyer(entrainement))
    regle = charger_regle(RACINE / "config" / "regle_decision.json")
    reference = charger_donnees(donnees_factices / "reference.csv")
    usagers = reference.drop(columns=["classe_retour_emploi"]).head(30)
    production = tmp_path / "production"

    composants = {}
    for identifiant, n in [("version_a", 600), ("version_b", 1000)]:
        modele, correction = entrainer(X_train.head(n), y_train.head(n), regle)
        chemins = artefacts.sauvegarder(tmp_path / "candidats" / identifiant, modele, correction, regle, {})
        publier_en_production(chemins, production, identifiant, {"run_id": identifiant})
        composants[identifiant] = artefacts.charger(tmp_path / "candidats" / identifiant)

    # La dernière version publiée est en service ; la précédente reste disponible
    assert dossier_en_service(production) == production / "version_b"
    assert (production / "version_a" / "MLmodel").exists()
    obtenu = charger_modele_en_service(production).predict(usagers)
    attendu = predire_usagers(composants["version_b"], usagers)
    assert (obtenu["classe"].to_numpy() == attendu["classe"].to_numpy()).all()


def test_fichier_renomme_sans_mot_cle_refuse(donnees_factices, tmp_path):
    # Fichier factice copié sous un nom neutre : il est reconnu à son contenu (identifiants FACTICE_)
    copie = tmp_path / "donnees_entrainement.csv"
    shutil.copy(donnees_factices / "entrainement.csv", copie)
    resultat = lancer(tmp_path, "--promouvoir", donnees=copie)
    assert "Refusé" in resultat.stderr, resultat.stdout + resultat.stderr
    assert not (tmp_path / "models" / "production").exists()


def test_alias_en_echec_ancienne_version_reste_en_service(donnees_factices, tmp_path):
    entrainement, _ = decouper(charger_donnees(donnees_factices / "entrainement.csv"))
    X_train, y_train = separer_x_y(nettoyer(entrainement))
    regle = charger_regle(RACINE / "config" / "regle_decision.json")
    modele, correction = entrainer(X_train, y_train, regle)
    production = tmp_path / "production"
    chemins = artefacts.sauvegarder(tmp_path / "candidats" / "a", modele, correction, regle, {})
    publier_en_production(chemins, production, "version_a", {})

    def alias_en_panne():
        raise RuntimeError("registre MLflow indisponible")

    chemins = artefacts.sauvegarder(tmp_path / "candidats" / "b", modele, correction, regle, {})
    with pytest.raises(RuntimeError):
        publier_en_production(chemins, production, "version_b", {}, publier_alias=alias_en_panne)

    # L'API et le registre continuent de désigner la même version : la précédente
    assert dossier_en_service(production) == production / "version_a"
    assert not list(production.glob("*.tmp"))


def test_feedbacks_ajoutes_a_l_entrainement_jamais_au_test(donnees_factices, tmp_path):
    """Réentraînement (/retrain) : les feedbacks rejoignent l'entraînement ; le jeu de test ne change pas."""
    feedbacks = charger_donnees(donnees_factices / "reference.csv").head(3)
    feedbacks["usager_id"] = [f"FEEDBACK_{i}" for i in range(3)]
    chemin_feedbacks = tmp_path / "feedbacks.csv"
    feedbacks.to_csv(chemin_feedbacks, index=False)

    (tmp_path / "sans").mkdir()
    (tmp_path / "avec").mkdir()
    sans = lancer(tmp_path / "sans")
    avec = lancer(tmp_path / "avec", "--feedbacks", str(chemin_feedbacks))
    assert sans.returncode == avec.returncode == 0, avec.stdout + avec.stderr

    def infos(dossier):
        candidat = next((dossier / "models" / "candidats").iterdir())
        return json.loads((candidat / "infos_entrainement.json").read_text(encoding="utf-8"))

    infos_sans, infos_avec = infos(tmp_path / "sans"), infos(tmp_path / "avec")
    assert infos_avec["n_feedbacks"] == 3 and infos_sans["n_feedbacks"] == 0
    assert infos_avec["n_entrainement"] == infos_sans["n_entrainement"] + 3
    assert infos_avec["n_test"] == infos_sans["n_test"]   # même jeu de test : comparaison à armes égales


def test_feedback_deja_dans_le_jeu_de_test_ecarte(donnees_factices, tmp_path):
    """Un usager du jeu de test renvoyé par predict → feedback n'entre pas dans l'entraînement."""
    _, test = decouper(charger_donnees(donnees_factices / "entrainement.csv"))
    # Même usager que dans le test, écrit autrement : code ROME en minuscules (accepté par l'API),
    # âge entier au lieu de décimal, espaces en trop dans la synthèse
    deguise = test[test["age"].notna()].head(1).copy()
    deguise["code_rome_vise"] = deguise["code_rome_vise"].str.lower()
    deguise["age"] = deguise["age"].astype("Int64")
    deguise["synthese_entretien"] = "  " + deguise["synthese_entretien"].str.replace(" ", "  ") + " "
    feedbacks = pd.concat([deguise, charger_donnees(donnees_factices / "reference.csv").head(1)])
    feedbacks["usager_id"] = ["FEEDBACK_test", "FEEDBACK_nouveau"]
    chemin_feedbacks = tmp_path / "feedbacks.csv"
    feedbacks.to_csv(chemin_feedbacks, index=False)

    resultat = lancer(tmp_path, "--feedbacks", str(chemin_feedbacks))
    assert resultat.returncode == 0, resultat.stdout + resultat.stderr
    candidat = next((tmp_path / "models" / "candidats").iterdir())
    infos = json.loads((candidat / "infos_entrainement.json").read_text(encoding="utf-8"))
    assert (infos["n_feedbacks"], infos["n_feedbacks_ecartes"]) == (1, 1)



def test_aucun_feedback_utilisable_pas_de_reentrainement(donnees_factices, tmp_path):
    _, test = decouper(charger_donnees(donnees_factices / "entrainement.csv"))
    chemin_feedbacks = tmp_path / "feedbacks.csv"
    test.head(2).to_csv(chemin_feedbacks, index=False)   # uniquement des usagers du jeu de test
    resultat = lancer(tmp_path, "--feedbacks", str(chemin_feedbacks))
    assert resultat.returncode == 2 and "Aucun feedback utilisable" in resultat.stdout
    assert not (tmp_path / "models" / "candidats").exists()


def test_modele_de_controle_pour_l_image_de_ci(donnees_factices, tmp_path):
    """CI : le candidat de contrôle est rangé à part (jamais dans « production ») et l'API sait le charger."""
    assert lancer(tmp_path).returncode == 0
    candidat = next((tmp_path / "models" / "candidats").iterdir())
    script = [sys.executable, "scripts/preparer_image_ci.py", str(candidat)]
    resultat = subprocess.run([*script, "--sortie", str(tmp_path / "ci")], cwd=RACINE,
                              capture_output=True, encoding="utf-8")
    assert resultat.returncode == 0, resultat.stdout + resultat.stderr
    modele = charger_modele_en_service(tmp_path / "ci")
    reference = charger_donnees(donnees_factices / "reference.csv")
    usagers = reference.drop(columns=["classe_retour_emploi"]).head(5)
    assert len(modele.predict(usagers)) == 5

    refus = subprocess.run([*script, "--sortie", str(tmp_path / "production")], cwd=RACINE,
                           capture_output=True, encoding="utf-8")
    assert refus.returncode != 0 and not (tmp_path / "production").exists()
