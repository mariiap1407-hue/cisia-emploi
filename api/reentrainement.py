"""Réentraînement monitoré à partir des feedbacks des conseillers (route /retrain).

Principe : la route ne réentraîne pas « à sa façon ». Elle lance le MÊME script que l'entraînement
manuel (scripts/entrainer.py), avec les mêmes garde-fous : quality gate sur le même jeu de test,
candidat enregistré dans MLflow, mise en production seulement si le gate passe (dossier, puis
actuelle.json, puis alias « production »), production inchangée sinon.

Étapes :
1. les feedbacks qui font foi (le plus récent par prédiction) sont mis au format du fichier
   d'origine : entrées validées de la prédiction + classe OBSERVÉE ; usager_id = « FEEDBACK_<id> » ;
   les colonnes non collectées (commune, allocataire, nationalité) restent vides ;
2. ce fichier est écrit dans un dossier temporaire (données personnelles : supprimé à la fin) ;
3. le script est lancé dans un processus séparé : python scripts/entrainer.py --feedbacks ...
   --sans-recherche [--promouvoir] ; un échec du script ne fait pas tomber l'API ;
4. le résultat est lu dans le dossier du candidat (infos_entrainement.json) et dans actuelle.json.
Le script publie dans le MÊME dossier de production que celui de l'API (--production).
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from cisia.modele_mlflow import FICHIER_ACTUELLE
from cisia.preparation import CIBLE, COLONNES_BRUTES

RACINE = Path(__file__).resolve().parents[1]
SCRIPT = RACINE / "scripts" / "entrainer.py"
DELAI_MAXIMUM_S = 1800   # au-delà, le réentraînement est considéré comme bloqué
NOM_MODELE = "cisia-orientation"


class EchecReentrainement(Exception):
    """Le script n'a pas produit de candidat (erreur technique, données refusées...)."""


class AucunFeedbackUtilisable(Exception):
    """Tous les feedbacks ont été écartés (profil déjà dans le jeu de test) : rien à réentraîner."""


def vers_tableau_entrainement(feedbacks):
    """Feedbacks au format du fichier d'origine (une ligne par prédiction corrigée)."""
    lignes = []
    for feedback in feedbacks:
        ligne = dict.fromkeys(COLONNES_BRUTES)
        ligne.update(feedback["entrees"])
        ligne["usager_id"] = f"FEEDBACK_{feedback['id_prediction']}"
        ligne[CIBLE] = int(feedback["classe_reelle"])
        lignes.append(ligne)
    return pd.DataFrame(lignes, columns=[*COLONNES_BRUTES, CIBLE])


def lancer_script(chemin_feedbacks, donnees, dossier_modeles, dossier_production, promouvoir):
    """Lance scripts/entrainer.py ; renvoie (code de retour, sorties texte)."""
    commande = [sys.executable, str(SCRIPT), "--donnees", str(donnees), "--feedbacks", str(chemin_feedbacks),
                "--sortie", str(dossier_modeles), "--production", str(dossier_production), "--sans-recherche"]
    if promouvoir:
        commande.append("--promouvoir")
    resultat = subprocess.run(commande, cwd=RACINE, capture_output=True, encoding="utf-8",
                              timeout=DELAI_MAXIMUM_S)
    return resultat.returncode, resultat.stdout + "\n" + resultat.stderr


def run_en_service(dossier_production):
    chemin = Path(dossier_production) / FICHIER_ACTUELLE
    if not chemin.exists():
        return None
    return json.loads(chemin.read_text(encoding="utf-8")).get("run_id")


def reentrainer(feedbacks, donnees, dossier_production, promouvoir=True, lanceur=lancer_script):
    """Réentraîne avec les feedbacks ; renvoie le statut et les indicateurs du nouveau candidat.

    `lanceur` est remplaçable dans les tests (même interface que lancer_script).
    """
    dossier_production = Path(dossier_production)
    dossier_modeles = dossier_production.parent   # candidats : <parent>/candidats/<run>/
    with tempfile.TemporaryDirectory() as dossier_temporaire:
        chemin_feedbacks = Path(dossier_temporaire) / "feedbacks.csv"
        vers_tableau_entrainement(feedbacks).to_csv(chemin_feedbacks, index=False)
        code_retour, sorties = lanceur(chemin_feedbacks, donnees, dossier_modeles, dossier_production,
                                       promouvoir)

    if "Aucun feedback utilisable" in sorties:
        raise AucunFeedbackUtilisable()
    trouve = re.search(r"Run MLflow : (\S+)", sorties)
    run_id = trouve.group(1) if trouve else "?"
    chemin_infos = dossier_modeles / "candidats" / run_id / "infos_entrainement.json"
    if not chemin_infos.exists():
        raise EchecReentrainement(f"code {code_retour} ; fin des sorties : {sorties[-2000:]}")

    infos = json.loads(chemin_infos.read_text(encoding="utf-8"))
    comparaison = infos.get("comparaison_production")
    if not infos["quality_gate_ok"]:
        statut = "refuse_quality_gate"
    elif run_en_service(dossier_production) == infos["run_id"]:
        statut = "mis_en_production"
    elif comparaison is not None and not comparaison["meilleur"]:
        statut = "non_promu_pas_meilleur"   # champion / challenger : le modèle en place reste
    elif promouvoir:   # gate respecté mais mise en production non faite : anomalie, pas un refus
        raise EchecReentrainement(f"mise en production non effectuée (code {code_retour}) ; "
                                  f"fin des sorties : {sorties[-2000:]}")
    else:
        statut = "candidat_non_promu"
    return {"statut": statut, "n_feedbacks": infos.get("n_feedbacks", len(feedbacks)),
            "n_feedbacks_ecartes": infos.get("n_feedbacks_ecartes", 0), "run_id": infos["run_id"],
            "quality_gate_ok": bool(infos["quality_gate_ok"]),
            "echecs_quality_gate": infos.get("echecs_quality_gate", []),
            "resultats_test": infos.get("resultats_test", {}),
            "comparaison_production": comparaison}


def retablir_alias(version):
    """Remet l'alias « production » du registre sur `version` (retour arrière après un échec d'activation).

    Renvoie True si l'alias a été remis, None s'il n'y a rien à remettre (pas de version du registre,
    ou registre désactivé). Une erreur du registre est levée à l'appelant.
    """
    if not version or os.getenv("USE_REGISTRY", "true").lower() != "true":
        return None
    suivi_par_defaut = f"sqlite:///{(RACINE / 'mlflow.db').as_posix()}"   # comme scripts/entrainer.py
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", suivi_par_defaut))
    mlflow.MlflowClient().set_registered_model_alias(NOM_MODELE, "production", str(version))
    return True
