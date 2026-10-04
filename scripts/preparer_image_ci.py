"""CI uniquement : prépare le modèle de CONTRÔLE (données factices) à copier dans l'image Docker de test.

Dans la CI, les vraies données ne sont jamais disponibles (RGPD) : le pipeline entraîne un modèle de
contrôle sur des données factices, le teste, puis construit l'image avec lui pour vérifier que
l'image fonctionne (démarrage, /health, /predict). Ce modèle ne vaut rien en conditions réelles.

Ce script le range dans un dossier SÉPARÉ (models/ci par défaut) ; par sécurité il refuse un dossier
nommé « production » (simple garde-fou sur le nom, pas une protection de tous les chemins possibles :
c'est la CI qui fixe la destination, models/ci, sur un runner éphémère). Le format est celui
qu'attend l'API (dossier du modèle + actuelle.json) ; l'image le copie grâce à l'argument de
construction MODELE (docker build --build-arg MODELE=models/ci).

Usage : python scripts/preparer_image_ci.py models/candidats/<run> [--sortie models/ci]
"""

import argparse
import json
import sys
from pathlib import Path

from cisia import artefacts
from cisia.modele_mlflow import publier_en_production


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidat", help="dossier du candidat de contrôle (models/candidats/<run>)")
    parser.add_argument("--sortie", default="models/ci", help="dossier du modèle pour l'image de test")
    args = parser.parse_args()
    for flux in (sys.stdout, sys.stderr):   # Windows : sorties en UTF-8 (accents), comme entrainer.py
        flux.reconfigure(encoding="utf-8")

    candidat = Path(args.candidat)
    sortie = Path(args.sortie)
    if sortie.resolve().name == "production":
        sys.exit("Refusé : le modèle de contrôle ne va jamais dans un dossier de production.")
    infos = json.loads((candidat / artefacts.FICHIERS["infos"]).read_text(encoding="utf-8"))
    chemins = {nom: candidat / fichier for nom, fichier in artefacts.FICHIERS.items()}
    identifiant = f"controle-{infos['run_id'][:8]}"
    infos_ci = {"run_id": identifiant, "modele_de_controle": True, "donnees": infos.get("donnees")}
    publier_en_production(chemins, sortie, identifiant, infos_ci)
    print(f"Modèle de contrôle prêt pour l'image : {sortie / identifiant}")


if __name__ == "__main__":
    main()
