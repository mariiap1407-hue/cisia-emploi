"""Le script d'entraînement protège la production (relecture du bloc 3).

Chaque test lance le vrai script sur des données factices, dans un dossier temporaire,
avec une base MLflow temporaire : rien ne touche à models/ ni à mlflow.db du projet.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow

from cisia import artefacts
from cisia.decision import charger_regle
from cisia.entrainement import entrainer
from cisia.inference import predire_usagers
from cisia.modele_mlflow import charger_modele_en_service, dossier_en_service, publier_en_production
from cisia.preparation import charger_donnees, decouper, nettoyer, separer_x_y

RACINE = Path(__file__).resolve().parents[1]


def lancer(dossier, *options):
    environnement = {**os.environ, "MLFLOW_TRACKING_URI": f"sqlite:///{(dossier / 'mlflow.db').as_posix()}"}
    return subprocess.run(
        [sys.executable, "scripts/entrainer.py", "--donnees", "data/factice/entrainement.csv",
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
