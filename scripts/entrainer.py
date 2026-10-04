"""Entraînement du modèle, instrumenté avec MLflow.

Reprend la logique du notebook (sections 5.4 à 6.6) avec le code partagé de src/cisia/ :
1. lecture, découpage 80 / 20 et nettoyage des données ;
2. recherche sur grille en deux tours (un run MLflow imbriqué par combinaison), pour la traçabilité ;
3. entraînement avec les réglages retenus dans le notebook, correction du risque, règle de décision ;
4. évaluation de la chaîne complète sur le jeu de test, matrice de confusion, quality gate ;
5. enregistrement du CANDIDAT (dossier models/candidats/<run>/ et version dans le registre MLflow) ;
6. mise en production seulement si le quality gate passe ET si l'option --promouvoir est donnée :
   modèle complet dans models/production/<run>/, déclaré en service dans models/production/actuelle.json,
   puis alias « production » dans le registre.

Garde-fous :
- quality gate non respecté : la production en place n'est pas touchée et le script s'arrête
  avec le code 1 (une CI qui l'appelle s'arrête aussi) ;
- données factices (chemin contenant « factice » ou identifiants « FACTICE_ ») : expérience et modèle séparés
  (« cisia-orientation-controle »), et mise en production refusée.

Ce que le script ne refait pas : le choix des réglages, de la méthode de recalibration et des seuils.
Ils ont été décidés dans le notebook (validation croisée, puis validation croisée imbriquée pour la
recalibration) ; le script les applique tels quels (REGLAGES_RETENUS, config/regle_decision.json).
La correction est apprise sur des probabilités HORS PLI, jamais sur les prédictions du modèle
sur ses propres données d'entraînement. La recherche sur grille est rejouée pour la traçabilité
de l'étude ; pour un réentraînement de routine, --sans-recherche évite ce calcul inutile.

Réentraînement avec les feedbacks des conseillers (--feedbacks, utilisé par la route /retrain) :
les exemples corrigés sont ajoutés à la partie ENTRAÎNEMENT seulement, après le découpage ; le jeu
de test reste exactement celui du notebook, donc le quality gate compare les modèles à armes égales.

Usage :
    python scripts/entrainer.py --promouvoir                       # vraies données, mise en production
    python scripts/entrainer.py                                    # vraies données, candidat seulement
    python scripts/entrainer.py --donnees data/factice/entrainement.csv --sans-recherche   # contrôle (CI)
    python scripts/entrainer.py --feedbacks feedbacks.csv --sans-recherche --promouvoir    # réentraînement
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pyfunc
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
from cisia.modele_mlflow import options_modele, publier_en_production
from cisia.preparation import charger_donnees, decouper, nettoyer, separer_x_y

RACINE = Path(__file__).resolve().parents[1]
DONNEES_PAR_DEFAUT = RACINE / "data" / "raw" / "dataset_trajectoire_emploi.csv"
CHEMIN_REGLE = RACINE / "config" / "regle_decision.json"
DOSSIER_MODELES = RACINE / "models"
NOM_MODELE = "cisia-orientation"
NOM_CONTROLE = "cisia-orientation-controle"   # entraînements sur données factices (CI, tests)
PREFIXE_FACTICE = "FACTICE_"                  # identifiants créés par generer_donnees_factices.py

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


def etat_du_code():
    """Commit Git et présence de modifications non enregistrées (le run doit pouvoir être retracé)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RACINE, capture_output=True,
                                text=True, check=True).stdout.strip()
        modifie = bool(subprocess.run(["git", "status", "--porcelain"], cwd=RACINE, capture_output=True,
                                      text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return "inconnu", True
    return commit, modifie


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--donnees", default=DONNEES_PAR_DEFAUT, help="fichier CSV d'entraînement")
    parser.add_argument("--regle", default=CHEMIN_REGLE, help="fichier de la règle de décision")
    parser.add_argument("--sortie", default=DOSSIER_MODELES,
                        help="dossier des modèles (candidats, production)")
    parser.add_argument("--sans-recherche", action="store_true",
                        help="ne rejoue pas la recherche sur grille (plus rapide : CI, réentraînement)")
    parser.add_argument("--feedbacks", default=None,
                        help="CSV des feedbacks des conseillers, ajoutés à l'entraînement seulement")
    parser.add_argument("--promouvoir", action="store_true",
                        help="met le modèle en production s'il passe le quality gate (désactivé par défaut)")
    args = parser.parse_args()

    # Sorties en UTF-8 : sous Windows, quand la sortie est redirigée (tests, CI), l'encodage par défaut
    # (cp1252) ne connaît pas certains caractères comme « → » et le script s'arrêtait en plein affichage
    for flux in (sys.stdout, sys.stderr):
        flux.reconfigure(encoding="utf-8")

    # Données factices : repérées par leur chemin ET par leur contenu (identifiants « FACTICE_ »),
    # pour qu'un fichier factice copié ailleurs ou renommé ne puisse pas être mis en production
    donnees_brutes = charger_donnees(args.donnees)
    donnees_factices = ("factice" in Path(args.donnees).as_posix().lower()
                        or donnees_brutes["usager_id"].str.startswith(PREFIXE_FACTICE).any())
    if args.promouvoir and donnees_factices:
        sys.exit("Refusé : un modèle entraîné sur des données factices ne peut pas être mis en production.")
    nom_modele = NOM_CONTROLE if donnees_factices else NOM_MODELE

    # Par défaut : base SQLite locale à la racine du projet (ignorée par Git)
    suivi_par_defaut = f"sqlite:///{(RACINE / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", suivi_par_defaut))
    utiliser_registre = os.getenv("USE_REGISTRY", "true").lower() == "true"
    mlflow.set_experiment(nom_modele)

    # 1. Données
    entrainement, test = decouper(donnees_brutes)
    n_feedbacks = 0
    if args.feedbacks:   # après le découpage : les feedbacks n'entrent jamais dans le jeu de test
        feedbacks = charger_donnees(args.feedbacks).reindex(columns=entrainement.columns)
        n_feedbacks = len(feedbacks)
        entrainement = pd.concat([entrainement, feedbacks], ignore_index=True)
    X_train, y_train = separer_x_y(nettoyer(entrainement))
    X_test, y_test = separer_x_y(nettoyer(test))
    regle = charger_regle(args.regle)
    commit, code_modifie = etat_du_code()

    with mlflow.start_run(run_name=f"entrainement-{datetime.now():%Y%m%d-%H%M}") as run:
        mlflow.log_params({
            "donnees": Path(args.donnees).name,
            "donnees_sha256": artefacts.empreinte(args.donnees),
            "donnees_factices": donnees_factices,
            "git_commit": commit,
            "code_modifie_non_commite": code_modifie,
            "n_feedbacks": n_feedbacks,
            "feedbacks_sha256": artefacts.empreinte(args.feedbacks) if args.feedbacks else "aucun",
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

        # 4. Évaluation de la chaîne complète sur le jeu de test, et quality gate
        probas, risque, classes = predire(modele, correction, regle, X_test)
        resultats = evaluer(y_test, probas, risque, classes)
        mlflow.log_metrics({f"test_{nom_mlflow(k)}": v for k, v in resultats.items()})
        with tempfile.TemporaryDirectory() as dossier_temporaire:
            mlflow.log_artifact(enregistrer_matrice_confusion(y_test, classes, dossier_temporaire))
        for nom, valeur in resultats.items():
            print(f"  {nom:<30} {valeur}" if isinstance(valeur, int) else f"  {nom:<30} {valeur:.3f}")
        echecs = verifier_seuils_qualite(resultats)
        mlflow.log_metric("quality_gate_ok", int(not echecs))

        # 5. Candidat : dossier propre à ce run (jamais écrasé), puis modèle MLflow complet dans le registre
        infos = {"run_id": run.info.run_id, "date": datetime.now().isoformat(timespec="seconds"),
                 "donnees": Path(args.donnees).name, "donnees_sha256": artefacts.empreinte(args.donnees),
                 "git_commit": commit, "code_modifie_non_commite": code_modifie,
                 "n_feedbacks": n_feedbacks, "n_entrainement": len(X_train), "n_test": len(X_test),
                 "quality_gate_ok": not echecs, "echecs_quality_gate": echecs,
                 "resultats_test": {k: round(float(v), 4) for k, v in resultats.items()}}
        dossier_candidat = Path(args.sortie) / "candidats" / run.info.run_id
        chemins = artefacts.sauvegarder(dossier_candidat, modele, correction, regle, infos)
        info_modele = mlflow.pyfunc.log_model(name="modele", **options_modele(chemins),
                                              registered_model_name=nom_modele if utiliser_registre else None)
        version = info_modele.registered_model_version   # la version créée par CET enregistrement

        # 6. Mise en production : seulement si le quality gate passe et si --promouvoir est demandé
        if echecs:
            print("Quality gate NON respecté : " + " ; ".join(echecs))
            print(f"Candidat conservé pour analyse dans {dossier_candidat} ; "
                  "la production n'est pas modifiée.")
        elif not args.promouvoir:
            print(f"Quality gate respecté. Candidat enregistré dans {dossier_candidat}"
                  + (f" et dans le registre (version {version})" if version else "")
                  + ", non mis en production (option --promouvoir).")
        else:
            # Dossier, puis actuelle.json, puis alias ; si l'alias échoue, l'ancienne version reste en service
            def publier_alias():
                MlflowClient().set_registered_model_alias(nom_modele, "production", str(version))

            dossier = publier_en_production(chemins, Path(args.sortie) / "production", run.info.run_id,
                                            {"run_id": run.info.run_id, "version_registre": version},
                                            publier_alias if utiliser_registre else None)
            print(f"Quality gate respecté : version {version} mise en production "
                  f"(alias « production », dossier {dossier}).")

    print(f"Run MLflow : {run.info.run_id}")
    if echecs:
        sys.exit(1)   # code de sortie non nul : une CI qui appelle ce script s'arrête


if __name__ == "__main__":
    main()
