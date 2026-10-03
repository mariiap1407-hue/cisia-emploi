"""Entraînement du modèle, instrumenté avec MLflow.

Reprend la logique du notebook (sections 5.4 à 6.6) avec le code partagé de src/cisia/ :
1. lecture, découpage 80 / 20 et nettoyage des données ;
2. recherche sur grille en deux tours (un run MLflow imbriqué par combinaison), pour la traçabilité ;
3. entraînement avec les réglages retenus dans le notebook, correction du risque, règle de décision ;
4. évaluation sur le jeu de test, matrice de confusion ;
5. sauvegarde dans models/, enregistrement dans le registre MLflow, alias « production »
   seulement si le quality gate passe.

Ce que le script ne refait pas : le choix des réglages, de la méthode de recalibration et des seuils.
Ils ont été décidés dans le notebook (validation croisée, puis validation croisée imbriquée pour la
recalibration) ; le script les applique tels quels (REGLAGES_RETENUS, config/regle_decision.json).
La correction est apprise sur des probabilités HORS PLI, jamais sur les prédictions du modèle
sur ses propres données d'entraînement.

Usage :
    python scripts/entrainer.py                         # vraies données (poste local)
    python scripts/entrainer.py --donnees data/factice/entrainement.csv --sans-recherche   # CI
"""

import argparse
import os
import re
import shutil
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mlflow import MlflowClient
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

from cisia import artefacts
from cisia.decision import charger_regle
from cisia.entrainement import entrainer, evaluer, predire
from cisia.evaluation import CV, METRIQUES, verifier_seuils_qualite
from cisia.modele import REGLAGES_RETENUS, construire_pipeline
from cisia.preparation import charger_donnees, decouper, nettoyer, separer_x_y

RACINE = Path(__file__).resolve().parents[1]
DONNEES_PAR_DEFAUT = RACINE / "data" / "raw" / "dataset_trajectoire_emploi.csv"
CHEMIN_REGLE = RACINE / "config" / "regle_decision.json"
DOSSIER_MODELES = RACINE / "models"
NOM_MODELE = "cisia-orientation"

# Les deux tours de la recherche sur grille (section 5.4)
GRILLES = {
    "tour 1": {"modele__num_leaves": [7, 15, 31], "modele__learning_rate": [0.03, 0.1],
               "modele__n_estimators": [100, 300]},
    "tour 2": {"modele__num_leaves": [3, 5, 7], "modele__learning_rate": [0.03, 0.1],
               "modele__n_estimators": [100, 300]},
}


def nom_mlflow(texte):
    """Nom accepté par MLflow : « Erreurs critiques (2→0) » devient « erreurs_critiques_2_0 »."""
    texte = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", texte.lower()).strip("_")


def recherche_sur_grille(X_train, y_train):
    """Rejoue les deux tours de la section 5.4 ; un run imbriqué par combinaison."""
    for tour, grille in GRILLES.items():
        recherche = GridSearchCV(construire_pipeline(reglages=None), grille, scoring=METRIQUES,
                                 refit=False, cv=CV)
        recherche.fit(X_train, y_train)
        resultats = pd.DataFrame(recherche.cv_results_)
        for _, ligne in resultats.iterrows():
            parametres = ligne["params"]
            reglages = {k.replace("modele__", ""): v for k, v in parametres.items()}
            nom_run = f"{tour} · " + " · ".join(f"{k}={v}" for k, v in reglages.items())
            with mlflow.start_run(run_name=nom_run, nested=True):
                mlflow.log_param("tour", tour)
                mlflow.log_params(reglages)
                # Le scorer des erreurs critiques renvoie un taux négatif (à minimiser) : remis en positif
                mlflow.log_metrics({f"cv_{nom_mlflow(m)}": abs(ligne[f"mean_test_{m}"]) for m in METRIQUES})
                mlflow.log_metric("cv_f1_macro_ecart_type", ligne["std_test_F1 macro"])
        print(f"Recherche sur grille, {tour} : {len(resultats)} combinaisons tracées")


def enregistrer_matrice_confusion(y_test, classes, dossier):
    """Matrice de confusion du jeu de test, règle retenue ; le cadre orange entoure les erreurs critiques."""
    fig = Figure(figsize=(5, 4.5))   # Figure seule : aucune fenêtre graphique n'est ouverte
    ax = fig.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, classes, cmap="Blues", colorbar=False, ax=ax,
                                            display_labels=["0 · Rapide", "1 · Moyen", "2 · Longue"])
    ax.add_patch(Rectangle((-0.5, 1.5), 1, 1, fill=False, edgecolor="#eb6834", linewidth=3))
    ax.set(title="Jeu de test · règle retenue", xlabel="Classe prédite", ylabel="Classe réelle")
    chemin = Path(dossier) / "matrice_confusion.png"
    fig.savefig(chemin, bbox_inches="tight")
    return chemin


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--donnees", default=DONNEES_PAR_DEFAUT, help="fichier CSV d'entraînement")
    parser.add_argument("--sans-recherche", action="store_true",
                        help="ne rejoue pas la recherche sur grille (plus rapide, utilisé dans la CI)")
    args = parser.parse_args()

    # Par défaut : base SQLite locale à la racine du projet (ignorée par Git)
    suivi_par_defaut = f"sqlite:///{(RACINE / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", suivi_par_defaut))
    utiliser_registre = os.getenv("USE_REGISTRY", "true").lower() == "true"
    mlflow.set_experiment("cisia-orientation")

    # 1. Données
    entrainement, test = decouper(charger_donnees(args.donnees))
    X_train, y_train = separer_x_y(nettoyer(entrainement))
    X_test, y_test = separer_x_y(nettoyer(test))
    regle = charger_regle(CHEMIN_REGLE)

    with mlflow.start_run(run_name=f"entrainement-{datetime.now():%Y%m%d-%H%M}") as run:
        mlflow.log_params({
            "donnees": Path(args.donnees).name,
            "n_entrainement": len(X_train),
            "n_test": len(X_test),
            **{k.replace("modele__", ""): v for k, v in REGLAGES_RETENUS.items()},
            "methode_recalibration": regle["methode_recalibration"],
            "seuil_classe_2": regle["seuil_classe_2"],
            "seuil_garde_fou_classe_0": regle["seuil_garde_fou_classe_0"],
        })

        # 2. Recherche sur grille (traçabilité du choix des réglages)
        if not args.sans_recherche:
            recherche_sur_grille(X_train, y_train)

        # 3. Modèle final avec les réglages retenus dans le notebook, et correction du risque
        modele, correction = entrainer(X_train, y_train, regle)

        # 4. Évaluation sur le jeu de test
        probas, risque, classes = predire(modele, correction, regle, X_test)
        resultats = evaluer(y_test, probas, risque, classes)
        mlflow.log_metrics({f"test_{nom_mlflow(k)}": v for k, v in resultats.items()})
        with tempfile.TemporaryDirectory() as dossier_temporaire:
            mlflow.log_artifact(enregistrer_matrice_confusion(y_test, classes, dossier_temporaire))
        for nom, valeur in resultats.items():
            print(f"  {nom:<30} {valeur}" if isinstance(valeur, int) else f"  {nom:<30} {valeur:.3f}")

        # 5. Quality gate
        echecs = verifier_seuils_qualite(resultats)
        mlflow.log_metric("quality_gate_ok", int(not echecs))

        # 6. Sauvegarde dans models/ : modèle, correction et règle ensemble
        infos = {"run_id": run.info.run_id, "date": datetime.now().isoformat(timespec="seconds"),
                 "donnees": Path(args.donnees).name, "quality_gate_ok": not echecs,
                 "resultats_test": {k: round(float(v), 4) for k, v in resultats.items()}}
        # Les anciens fichiers de models/ sont remplacés (sauf .gitkeep, qui garde le dossier dans Git)
        for ancien in DOSSIER_MODELES.glob("*"):
            if ancien.name == ".gitkeep":
                continue
            if ancien.is_dir():
                shutil.rmtree(ancien)
            else:
                ancien.unlink()
        artefacts.sauvegarder(DOSSIER_MODELES, modele, correction, regle, infos)
        mlflow.log_artifacts(DOSSIER_MODELES, artifact_path="models")

        # 7. Registre de modèles : nouvelle version, alias « production » seulement si le quality gate passe
        if utiliser_registre:
            mlflow.sklearn.log_model(modele, name="modele", registered_model_name=NOM_MODELE,
                                     serialization_format="cloudpickle")
            client = MlflowClient()
            version = max(int(v.version) for v in client.search_model_versions(f"name='{NOM_MODELE}'"))
            if echecs:
                print(f"Quality gate non respecté, version {version} non promue : " + " ; ".join(echecs))
            else:
                client.set_registered_model_alias(NOM_MODELE, "production", str(version))
                print(f"Quality gate respecté : version {version} promue avec l'alias « production »")

    print(f"Modèle enregistré dans {DOSSIER_MODELES} (run MLflow {run.info.run_id})")
    if echecs:
        print("ATTENTION : quality gate non respecté : " + " ; ".join(echecs))


if __name__ == "__main__":
    main()
