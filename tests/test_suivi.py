"""Suivi du service (B10) : indicateurs depuis le journal, seuils, alertes.

Test d'acceptation du B10 : sur un journal FACTICE qui franchit les seuils, les alertes attendues sont
produites (code de sortie 2) ; sur un journal sain, aucune alerte (code 0).
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from api.journal import Journal
from cisia.suivi import analyser, code_de_sortie, psi_categoriel, psi_numerique

RACINE = Path(__file__).resolve().parents[1]
SEUILS = json.loads((RACINE / "config" / "seuils_suivi.json").read_text(encoding="utf-8"))
MAINTENANT = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)


def lancer(script, *options):
    return subprocess.run([sys.executable, f"scripts/{script}", *options], cwd=RACINE,
                          capture_output=True, encoding="utf-8")


@pytest.fixture(scope="module")
def journaux(donnees_factices, tmp_path_factory):
    """Les deux journaux factices (sain, dégradé), générés une seule fois pour le module."""
    dossier = tmp_path_factory.mktemp("journaux")
    resultat = lancer("generer_journal_factice.py", "--sortie", str(dossier))
    assert resultat.returncode == 0, resultat.stdout + resultat.stderr
    return dossier


def test_acceptation_journal_sain_puis_degrade(journaux, tmp_path):

    sain = lancer("suivi.py", "--base", str(journaux / "journal_sain.db"),
                  "--sortie", str(tmp_path / "rapports"))
    assert sain.returncode == 0, sain.stdout + sain.stderr
    assert "0 alerte(s)" in sain.stdout

    degrade = lancer("suivi.py", "--base", str(journaux / "journal_degrade.db"),
                     "--sortie", str(tmp_path / "rapports"))
    assert degrade.returncode == 2, degrade.stdout + degrade.stderr   # au moins une alerte critique
    rapport = json.loads(max((tmp_path / "rapports").glob("rapport_*.json")).read_text(encoding="utf-8"))
    indicateurs = {alerte["indicateur"] for alerte in rapport["alertes"]}
    assert indicateurs == {"erreurs du service", "latence", "répartition des prédictions",
                           "accord conseillers / modèle", "biais des étiquettes", "performance réelle",
                           "dérive des synthèses", "négations dans les synthèses", "registre MLflow"}
    sources = rapport["indicateurs"]["performance_observee"]["par_source"]
    assert set(sources) == {"référentiel", "saisie manuelle"}
    # Les alertes sont aussi écrites, horodatées, dans le journal des alertes
    alertes = (tmp_path / "rapports" / "alertes.log").read_text(encoding="utf-8")
    assert "CRITICAL registre MLflow" in alertes and "WARNING latence" in alertes


def test_peu_de_donnees_insuffisant_sans_alerte(tmp_path):
    journal = Journal(tmp_path / "journal.db")
    date = MAINTENANT.isoformat()
    for i in range(3):
        journal.enregistrer_inference(f"p{i}", date, None, "ok", entrees={"synthese_entretien": "Projet."},
                                      sorties={"classe": 2}, duree_ms=20)
        journal.enregistrer_feedback(f"p{i}", date, 0)          # trois « erreurs », mais trop peu de cas
        journal.enregistrer_avis(f"p{i}", date, "corrige", 0, "Autre")
    rapport = analyser(tmp_path / "journal.db", SEUILS, maintenant=MAINTENANT)
    assert rapport["alertes"] == [] and code_de_sortie(rapport) == 0
    assert rapport["indicateurs"]["performance_observee"]["statut"] == "insuffisant"
    assert rapport["indicateurs"]["accord_conseillers"]["statut"] == "insuffisant"


def test_fenetre_glissante(tmp_path):
    journal = Journal(tmp_path / "journal.db")
    ancien = (MAINTENANT - timedelta(days=60)).isoformat()
    for i in range(10):
        journal.enregistrer_inference(f"a{i}", ancien, None, "erreur", message_erreur="ancienne panne")
    journal.enregistrer_inference("r1", MAINTENANT.isoformat(), None, "ok", sorties={"classe": 1},
                                  duree_ms=20)
    rapport = analyser(tmp_path / "journal.db", SEUILS, maintenant=MAINTENANT)   # 30 jours
    assert rapport["indicateurs"]["service"]["requetes"] == 1 and rapport["alertes"] == []
    rapport = analyser(tmp_path / "journal.db", SEUILS, jours=90, maintenant=MAINTENANT)
    assert rapport["indicateurs"]["service"]["erreur"] == 10 and code_de_sortie(rapport) == 2


def test_retour_arriere_pointeur_et_alias_distingues(tmp_path):
    """Relecture B6 : distinguer « pointeur rétabli » et « alias du registre rétabli »."""
    journal = Journal(tmp_path / "journal.db")
    date = MAINTENANT.isoformat()
    cas = {
        "verifie_sans_registre": {"necessaire": True, "pointeur_retabli": True, "alias_retabli": None,
                                  "verifie": True},
        "alias_non_remis": {"necessaire": True, "pointeur_retabli": True, "alias_retabli": False,
                            "verifie": True},
        "non_verifie": {"necessaire": True, "pointeur_retabli": True, "alias_retabli": True,
                        "verifie": False},
    }
    for etat in cas.values():
        journal.enregistrer_reentrainement(date, date, "erreur", 3, resultats={"retour_arriere": etat})
    journal.enregistrer_reentrainement(date, date, "erreur_activation", 3)   # journal antérieur au B10
    alertes = analyser(tmp_path / "journal.db", SEUILS, maintenant=MAINTENANT)["alertes"]
    assert [(a["niveau"], a["indicateur"]) for a in alertes] == [
        ("attention", "réentraînement"),      # échec, mais ancienne version restée en service (vérifié)
        ("critique", "registre MLflow"),      # pointeur rétabli, alias NON remis : registre et API divergent
        ("critique", "réentraînement"),       # retour arrière non vérifié
        ("attention", "réentraînement"),      # état non tracé
    ]


def test_psi_et_niveaux():
    generateur = np.random.default_rng(0)
    reference = generateur.normal(40, 10, 2000)
    assert psi_numerique(reference, generateur.normal(40, 10, 500)) < 0.10          # même population
    assert psi_numerique(reference, generateur.normal(55, 10, 500)) >= 0.25         # population plus âgée
    assert psi_categoriel(["Bac"] * 50 + ["Bac+2"] * 50, ["Bac"] * 50 + ["Bac+2"] * 50) == 0
    assert psi_categoriel(["Bac"] * 90 + ["Bac+5"] * 10, ["Bac"] * 10 + ["Bac+5"] * 90) >= 0.25


def test_derive_des_donnees_psi_et_ks(journaux, tmp_path, donnees_factices):
    """Avec la référence (données factices d'entraînement) : population saine = stable, aucune alerte ;
    population dégradée = dérive globale de la synthèse ET dérive LOCALISÉE (âge, métiers du transport),
    invisible au niveau global : seul le suivi par métier et par semaine la révèle (comme le M6)."""
    reference = ["--reference", str(donnees_factices / "entrainement.csv")]

    sain = lancer("suivi.py", "--base", str(journaux / "journal_sain.db"), "--sortie", str(tmp_path / "a"),
                  *reference)
    assert sain.returncode == 0, sain.stdout + sain.stderr
    rapport = json.loads((tmp_path / "a" / "dernier_rapport.json").read_text(encoding="utf-8"))
    derive = rapport["indicateurs"]["derive_des_donnees"]
    assert {v["derive"] for v in derive.values()} == {"stable"}
    assert "ks_p_valeur" in derive["age"]
    # Semaine × métier : ~140 tests sur de petites cellules ; quelques fausses alertes isolées restent
    # possibles (c'est pourquoi seule la DERNIÈRE semaine déclenche une alerte), jamais au niveau global
    cellules = rapport["derive_par_semaine"]
    assert not any(c["derive"] for c in cellules if c["perimetre"] == "__global__")
    assert sum(c["derive"] for c in cellules) <= 0.03 * len(cellules)

    degrade = lancer("suivi.py", "--base", str(journaux / "journal_degrade.db"),
                     "--sortie", str(tmp_path / "b"), *reference)
    assert degrade.returncode == 2, degrade.stdout + degrade.stderr
    rapport = json.loads((tmp_path / "b" / "dernier_rapport.json").read_text(encoding="utf-8"))
    derive = rapport["indicateurs"]["derive_des_donnees"]
    assert derive["longueur_synthese"]["derive"] == "forte"
    assert derive["age"]["derive"] != "forte"                              # noyée dans la moyenne...
    localisees = [a for a in rapport["alertes"] if a["indicateur"] == "dérive localisée"]
    assert localisees and all("« N »" in a["message"] and "age" in a["message"] for a in localisees)
    # ... mais visible métier par métier : tableau semaine × métier écrit pour le tableau de bord
    tableau = pd.read_csv(tmp_path / "b" / "derive_par_semaine.csv")
    transport = tableau[(tableau["perimetre"] == "N") & (tableau["variable"] == "age")]
    assert transport["derive"].any()
