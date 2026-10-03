"""Validation des entrées de l'API : ce qui est accepté, ce qui est refusé (sans lancer l'API)."""

import json
import math

import pytest
from pydantic import ValidationError

from api.schemas import DemandePrediction, Feedback, Usager, erreurs_sans_valeurs, vers_tableau

USAGER = {"age": 35, "niveau_diplome": "Bac", "anciennete_poste_ans": 4.5,
          "code_rome_vise": "M1607", "synthese_entretien": "Projet clair, mobilité possible."}


def test_usager_valide():
    assert Usager(**USAGER).code_rome_vise == "M1607"


def test_informations_manquantes_acceptees():
    # Âge, diplôme et ancienneté inconnus : le modèle sait les traiter (médiane, code -1)
    usager = Usager(code_rome_vise="K2204", synthese_entretien="Synthèse courte.")
    tableau = vers_tableau(usager)
    assert math.isnan(tableau["age"].iloc[0]) and tableau["niveau_diplome"].isna().iloc[0]


@pytest.mark.parametrize("champ, valeur", [
    ("age", -5), ("age", 120), ("age", float("inf")), ("anciennete_poste_ans", -1),
    ("niveau_diplome", "Doctorat"), ("code_rome_vise", "M16"), ("code_rome_vise", "1607M"),
    ("synthese_entretien", ""), ("synthese_entretien", "   "),
])
def test_informations_invalides_refusees(champ, valeur):
    with pytest.raises(ValidationError):
        Usager(**{**USAGER, champ: valeur})


def test_donnee_non_demandee_refusee():
    # Minimisation des données : la nationalité n'est pas collectée
    with pytest.raises(ValidationError):
        Usager(**USAGER, nationalite_hors_ue=1)


def test_erreurs_sans_les_valeurs_recues():
    with pytest.raises(ValidationError) as erreur:
        Usager(**{**USAGER, "age": float("inf")}, nationalite_hors_ue=1)
    erreurs = erreurs_sans_valeurs(erreur.value.errors())
    json.dumps(erreurs, allow_nan=False)   # toujours représentable en JSON
    assert all(set(e) == {"champ", "type", "message"} for e in erreurs)


def test_demande_et_feedback():
    assert DemandePrediction(usager=USAGER, id_session="S-1").id_session == "S-1"
    with pytest.raises(ValidationError):
        Feedback(id_prediction="abc", classe_reelle=3)
