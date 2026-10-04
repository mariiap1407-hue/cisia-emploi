"""Explicabilité GLOBALE du modèle en service : sur quelles informations s'appuie-t-il en général ?

Moyenne, sur le jeu de test, de la valeur absolue des contributions SHAP de chaque information saisie,
pour chacune des trois classes. Sert au notebook (section 7) et à l'analyse éthique (section 8).
Écrit un tableau à l'écran et un graphique dans outputs/importance_shap.png (non versionné).

Usage (depuis la racine du projet) :
    python scripts/expliquer_modele.py [--donnees data/raw/dataset_trajectoire_emploi.csv]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from cisia.decision import RECOMMANDATIONS
from cisia.explication import importance_globale
from cisia.modele_mlflow import charger_modele_en_service
from cisia.preparation import charger_donnees, decouper, nettoyer, separer_x_y

RACINE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--donnees", default=str(RACINE / "data" / "raw" / "dataset_trajectoire_emploi.csv"))
    parser.add_argument("--production", default=str(RACINE / "models" / "production"))
    parser.add_argument("--sortie", default=str(RACINE / "outputs" / "importance_shap.png"))
    args = parser.parse_args()

    modele = charger_modele_en_service(args.production).unwrap_python_model().composants["modele"]
    _, test = decouper(charger_donnees(args.donnees))
    X_test, _ = separer_x_y(nettoyer(test))

    tableau = pd.DataFrame({RECOMMANDATIONS[classe]: importance_globale(modele, X_test, classe)
                            for classe in (0, 1, 2)})
    tableau = tableau.sort_values(RECOMMANDATIONS[2], ascending=False)
    print(f"Importance moyenne |SHAP| (score brut, logit multiclasse), jeu de test : {len(X_test)} usagers")
    print(tableau.round(3).to_string())

    Path(args.sortie).parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(8, 4), layout="tight")
    ax = figure.subplots()
    tableau.iloc[::-1].plot.barh(ax=ax, color=["#18753c", "#000091", "#f2a900"])
    ax.set_xlabel("Moyenne de |contribution SHAP| au score brut (logit multiclasse)")
    ax.set_title("Sur quoi le modèle s'appuie-t-il ? (jeu de test)")
    figure.savefig(args.sortie, dpi=150)
    print(f"Graphique : {args.sortie}")


if __name__ == "__main__":
    for flux in (sys.stdout, sys.stderr):
        flux.reconfigure(encoding="utf-8")
    main()
