"""Lecture et nettoyage des données (notebook, sections 2.1, 2.4 et 4.2)."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CIBLE = "classe_retour_emploi"
COL_TEXTE = "synthese_entretien"

# Colonnes du fichier source, hors variable cible
COLONNES_BRUTES = [
    "usager_id", "age", "niveau_diplome", "anciennete_poste_ans", "code_rome_vise",
    "code_insee_commune", "est_allocataire", "nationalite_hors_ue", "synthese_entretien",
]

# Les codes sont lus comme du texte pour garder les zéros initiaux (« 01405 ») et la Corse (« 2A »)
TYPES_LECTURE = {"usager_id": "string", "code_insee_commune": "string", "code_rome_vise": "string"}

ORDRE_DIPLOME = {"Sans diplôme": 0, "Bac": 1, "Bac+2": 2, "Bac+5": 3}

REGIONS = {
    "Auvergne-Rhône-Alpes": ["01", "03", "07", "15", "26", "38", "42", "43", "63", "69", "73", "74"],
    "Bourgogne-Franche-Comté": ["21", "25", "39", "58", "70", "71", "89", "90"],
    "Bretagne": ["22", "29", "35", "56"],
    "Centre-Val de Loire": ["18", "28", "36", "37", "41", "45"],
    "Corse": ["2A", "2B"],
    "Grand Est": ["08", "10", "51", "52", "54", "55", "57", "67", "68", "88"],
    "Hauts-de-France": ["02", "59", "60", "62", "80"],
    "Île-de-France": ["75", "77", "78", "91", "92", "93", "94", "95"],
    "Normandie": ["14", "27", "50", "61", "76"],
    "Nouvelle-Aquitaine": ["16", "17", "19", "23", "24", "33", "40", "47", "64", "79", "86", "87"],
    "Occitanie": ["09", "11", "12", "30", "31", "32", "34", "46", "48", "65", "66", "81", "82"],
    "Pays de la Loire": ["44", "49", "53", "72", "85"],
    "Provence-Alpes-Côte d'Azur": ["04", "05", "06", "13", "83", "84"],
    "Outre-mer": ["971", "972", "973", "974", "976"],
}
DEP_VERS_REGION = {dep: region for region, deps in REGIONS.items() for dep in deps}


def charger_donnees(chemin):
    """Lit le fichier CSV avec les bons types (section 2.1)."""
    return pd.read_csv(chemin, dtype=TYPES_LECTURE)


def decouper(df_brut):
    """Découpage stratifié 80 / 20, avec la même graine que le notebook (section 2.4)."""
    return train_test_split(df_brut, test_size=0.2, stratify=df_brut[CIBLE], random_state=42)


def nettoyer(df_entree):
    """Nettoyage et création des variables (section 4.2), identique au notebook."""
    df = df_entree.copy()

    # Diplôme : encodage ordinal, « Non renseigné » = -1
    df["niveau_diplome_ord"] = df["niveau_diplome"].map(ORDRE_DIPLOME).fillna(-1).astype(int)

    # Ancienneté invraisemblable -> valeur manquante (imputée plus tard par la médiane)
    impossible = df["anciennete_poste_ans"] > df["age"] - 15
    df.loc[impossible, "anciennete_poste_ans"] = np.nan

    # Code commune -> région
    code_commune = df["code_insee_commune"].str.strip().str.upper()
    departement = code_commune.str[:2].where(~code_commune.str.startswith("97"), code_commune.str[:3])
    df["region"] = departement.map(DEP_VERS_REGION).fillna("Inconnue")

    # Code ROME -> domaine (1re lettre)
    df["domaine_rome"] = df["code_rome_vise"].str.strip().str.upper().str[0]

    # Synthèse absente -> texte vide
    df["synthese_entretien"] = df["synthese_entretien"].fillna("")

    return df


def separer_x_y(df):
    """Variables explicatives et cible (section 4.3)."""
    return df.drop(columns=[CIBLE]), df[CIBLE]
