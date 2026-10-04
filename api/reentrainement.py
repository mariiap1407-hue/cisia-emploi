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
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from cisia.modele_mlflow import FICHIER_ACTUELLE
from cisia.preparation import CIBLE, COLONNES_BRUTES

RACINE = Path(__file__).resolve().parents[1]
SCRIPT = RACINE / "scripts" / "entrainer.py"
DELAI_MAXIMUM_S = 1800   # au-delà, le réentraînement est considéré comme bloqué


class EchecReentrainement(Exception):
    """Le script n'a pas produit de candidat (erreur technique, données refusées...)."""


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


def lancer_script(chemin_feedbacks, donnees, dossier_modeles, promouvoir):
    """Lance scripts/entrainer.py ; renvoie (code de retour, sorties texte)."""
    commande = [sys.executable, str(SCRIPT), "--donnees", str(donnees), "--feedbacks", str(chemin_feedbacks),
                "--sortie", str(dossier_modeles), "--sans-recherche"]
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
    dossier_modeles = dossier_production.parent   # le script écrit dans <sortie>/candidats et /production
    with tempfile.TemporaryDirectory() as dossier_temporaire:
        chemin_feedbacks = Path(dossier_temporaire) / "feedbacks.csv"
        vers_tableau_entrainement(feedbacks).to_csv(chemin_feedbacks, index=False)
        code_retour, sorties = lanceur(chemin_feedbacks, donnees, dossier_modeles, promouvoir)

    trouve = re.search(r"Run MLflow : (\S+)", sorties)
    run_id = trouve.group(1) if trouve else "?"
    chemin_infos = dossier_modeles / "candidats" / run_id / "infos_entrainement.json"
    if not chemin_infos.exists():
        raise EchecReentrainement(f"code {code_retour} ; fin des sorties : {sorties[-2000:]}")

    infos = json.loads(chemin_infos.read_text(encoding="utf-8"))
    if not infos["quality_gate_ok"]:
        statut = "refuse_quality_gate"
    elif run_en_service(dossier_production) == infos["run_id"]:
        statut = "mis_en_production"
    elif promouvoir:   # gate respecté mais mise en production non faite : anomalie, pas un refus
        raise EchecReentrainement(f"mise en production non effectuée (code {code_retour}) ; "
                                  f"fin des sorties : {sorties[-2000:]}")
    else:
        statut = "candidat_non_promu"
    return {"statut": statut, "n_feedbacks": len(feedbacks), "run_id": infos["run_id"],
            "quality_gate_ok": bool(infos["quality_gate_ok"]),
            "echecs_quality_gate": infos.get("echecs_quality_gate", []),
            "resultats_test": infos.get("resultats_test", {})}
