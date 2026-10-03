"""Tests du nettoyage des données (section 4.2 du notebook)."""

import numpy as np
import pandas as pd

from cisia.preparation import nettoyer


def un_usager(**valeurs):
    """Un usager fictif valide ; on remplace seulement les valeurs à tester."""
    usager = {
        "usager_id": "TEST_1", "age": 40.0, "niveau_diplome": "Bac", "anciennete_poste_ans": 3.0,
        "code_rome_vise": "M1607", "code_insee_commune": "69123", "est_allocataire": 1.0,
        "nationalite_hors_ue": 0, "synthese_entretien": "Projet clair.",
    }
    usager.update(valeurs)
    return pd.DataFrame([usager]).astype({"code_insee_commune": "string", "code_rome_vise": "string"})


def test_diplome_encode_dans_l_ordre():
    assert nettoyer(un_usager(niveau_diplome="Sans diplôme"))["niveau_diplome_ord"].iloc[0] == 0
    assert nettoyer(un_usager(niveau_diplome="Bac+5"))["niveau_diplome_ord"].iloc[0] == 3


def test_diplome_manquant_vaut_moins_un():
    assert nettoyer(un_usager(niveau_diplome=np.nan))["niveau_diplome_ord"].iloc[0] == -1


def test_anciennete_invraisemblable_devient_manquante():
    # 7 ans d'ancienneté à 18 ans : plus que « âge - 15 »
    assert np.isnan(nettoyer(un_usager(age=18.0, anciennete_poste_ans=7.0))["anciennete_poste_ans"].iloc[0])


def test_anciennete_plausible_conservee():
    assert nettoyer(un_usager(age=40.0, anciennete_poste_ans=12.0))["anciennete_poste_ans"].iloc[0] == 12.0


def test_region_depuis_le_code_commune():
    assert nettoyer(un_usager(code_insee_commune="69123"))["region"].iloc[0] == "Auvergne-Rhône-Alpes"
    assert nettoyer(un_usager(code_insee_commune="2A004"))["region"].iloc[0] == "Corse"
    assert nettoyer(un_usager(code_insee_commune="97411"))["region"].iloc[0] == "Outre-mer"
    assert nettoyer(un_usager(code_insee_commune="99999"))["region"].iloc[0] == "Inconnue"


def test_domaine_rome_et_synthese_vide():
    df = nettoyer(un_usager(code_rome_vise=" m1607", synthese_entretien=np.nan))
    assert df["domaine_rome"].iloc[0] == "M"
    assert df["synthese_entretien"].iloc[0] == ""


def test_les_donnees_d_entree_ne_sont_pas_modifiees():
    entree = un_usager(age=18.0, anciennete_poste_ans=7.0)
    nettoyer(entree)
    assert entree["anciennete_poste_ans"].iloc[0] == 7.0
