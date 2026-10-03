"""Tests de la recalibration et de la règle de décision (sections 6.3 à 6.6 du notebook)."""

import json
from pathlib import Path

import numpy as np
import pytest

from cisia.decision import (
    ajuster_recalibrage,
    appliquer_recalibrage,
    appliquer_regle,
    charger_regle,
    presenter,
)

# Probabilités brutes (classes 0, 1, 2) de trois usagers
PROBAS = np.array([
    [0.70, 0.20, 0.10],
    [0.30, 0.60, 0.10],
    [0.10, 0.20, 0.70],
])


def test_sans_seuil_regle_par_defaut():
    assert appliquer_regle(PROBAS, np.array([0.1, 0.1, 0.7])).tolist() == [0, 1, 2]


def test_seuil_classe_2():
    # Le risque recalibré du premier usager atteint 0,25 : il passe en classe 2
    risque = np.array([0.30, 0.10, 0.20])
    assert appliquer_regle(PROBAS, risque, seuil_classe_2=0.25).tolist() == [2, 1, 1]


def test_garde_fou_remplace_la_classe_0_par_la_classe_1():
    risque = np.array([0.09, 0.09, 0.05])
    assert appliquer_regle(PROBAS, risque, seuil_garde_fou=0.08).tolist() == [1, 1, 2]


def test_garde_fou_ne_touche_pas_un_risque_faible():
    risque = np.array([0.05, 0.05, 0.05])
    assert appliquer_regle(PROBAS, risque, seuil_garde_fou=0.08).tolist() == [0, 1, 2]


def test_seuil_absent_none_ou_nan():
    risque = np.array([0.1, 0.1, 0.7])
    assert appliquer_regle(PROBAS, risque, None, None).tolist() == [0, 1, 2]
    assert appliquer_regle(PROBAS, risque, np.nan, np.nan).tolist() == [0, 1, 2]


def test_niveaux_d_alerte():
    assert presenter(PROBAS[0], 0.30, 0.25, 0.08)["Niveau d'alerte"] == "Élevé"
    assert presenter(PROBAS[0], 0.10, 0.25, 0.08)["Niveau d'alerte"] == "Modéré"
    assert presenter(PROBAS[0], 0.05, 0.25, 0.08)["Niveau d'alerte"] == "Faible"


def test_presenter_sans_seuil_classe_2():
    # Cas relevé en relecture : sans seuil de classe 2, « Élevé » doit rester possible
    resultat = presenter(np.array([0.1, 0.2, 0.7]), 0.50, None, 0.08)
    assert resultat["Recommandation"] == "Accompagnement renforcé"
    assert resultat["Niveau d'alerte"] == "Élevé"


def test_presenter_format_du_risque():
    assert presenter(PROBAS[1], 0.093, 0.25, 0.08)["Risque de longue durée"] == "9%"


def test_recalibration_logistique_reste_entre_0_et_1():
    rng = np.random.default_rng(0)
    p = rng.random(200)
    y = (rng.random(200) < p * 0.7).astype(int)
    modele = ajuster_recalibrage("Logistique", p, y)
    risque = appliquer_recalibrage("Logistique", modele, p)
    assert risque.shape == p.shape
    assert ((risque > 0) & (risque < 1)).all()


def test_aucune_correction_renvoie_la_probabilite_brute():
    p = np.array([0.1, 0.5])
    assert ajuster_recalibrage("Aucune correction", p, np.array([0, 1])) is None
    assert appliquer_recalibrage("Aucune correction", None, p) is p


def test_methode_inconnue_refusee():
    with pytest.raises(ValueError):
        ajuster_recalibrage("Sigmoide", np.array([0.1, 0.5]), np.array([0, 1]))


def test_fichier_de_regle_du_depot():
    regle = charger_regle(Path(__file__).resolve().parents[1] / "config" / "regle_decision.json")
    assert regle["seuil_classe_2"] == 0.25
    assert regle["seuil_garde_fou_classe_0"] == 0.08
    assert regle["methode_recalibration"] == "Logistique"


REGLE_VALIDE = {"seuil_classe_2": 0.25, "seuil_garde_fou_classe_0": 0.08,
                "methode_recalibration": "Logistique"}


def ecrire_regle(dossier, **modifications):
    regle = {**REGLE_VALIDE, **modifications}
    chemin = dossier / "regle.json"
    chemin.write_text(json.dumps(regle), encoding="utf-8")
    return chemin


def test_regle_avec_methode_inconnue_refusee(tmp_path):
    with pytest.raises(ValueError):
        charger_regle(ecrire_regle(tmp_path, methode_recalibration="Sigmoide"))


def test_regle_avec_seuil_hors_bornes_refusee(tmp_path):
    with pytest.raises(ValueError):
        charger_regle(ecrire_regle(tmp_path, seuil_classe_2=1.5))
    with pytest.raises(ValueError):
        charger_regle(ecrire_regle(tmp_path, seuil_garde_fou_classe_0=-0.1))


def test_regle_incomplete_refusee(tmp_path):
    chemin = tmp_path / "regle.json"
    chemin.write_text(json.dumps({"methode_recalibration": "Logistique"}), encoding="utf-8")
    with pytest.raises(ValueError):
        charger_regle(chemin)


def test_regle_avec_seuils_absents_acceptee(tmp_path):
    regle = charger_regle(ecrire_regle(tmp_path, seuil_classe_2=None, seuil_garde_fou_classe_0=None))
    assert regle["seuil_classe_2"] is None


def test_recalibration_refuse_une_cible_a_trois_classes():
    # Erreur relevée en relecture : avec la cible 0/1/2, la correction renverrait le risque de la classe 1
    with pytest.raises(ValueError):
        ajuster_recalibrage("Logistique", np.array([0.1, 0.5, 0.9]), np.array([0, 1, 2]))


def test_recalibration_refuse_une_cible_sans_classe_2():
    with pytest.raises(ValueError):
        ajuster_recalibrage("Logistique", np.array([0.1, 0.5]), np.array([0, 0]))
