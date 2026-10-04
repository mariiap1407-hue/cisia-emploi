"""Journal SQLite : migration d'une ancienne base, ordre de l'historique (sans lancer l'API)."""

import json
import sqlite3
from contextlib import closing

from api.journal import Journal

ANCIEN_SCHEMA = """
CREATE TABLE inferences (
    id_prediction TEXT PRIMARY KEY, date TEXT NOT NULL, id_session TEXT, statut TEXT NOT NULL,
    entrees TEXT, sorties TEXT, version_modele TEXT, duree_ms REAL, message_erreur TEXT
);
"""


def test_anciennes_requetes_refusees_nettoyees(tmp_path):
    # Base créée avant le bloc 5b : la requête refusée y était gardée AVEC ses valeurs
    chemin = tmp_path / "ancienne.db"
    with closing(sqlite3.connect(chemin)) as connexion, connexion:
        connexion.executescript(ANCIEN_SCHEMA)
        connexion.execute("INSERT INTO inferences VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          ("ancienne", "2026-10-03T23:44:44.795", "demo-1", "invalide",
                           json.dumps({"usager": {"age": -5}}), None, None, None,
                           json.dumps([{"loc": ["age"], "input": -5}])))

    journal = Journal(chemin)   # ouverture = migration

    with closing(sqlite3.connect(chemin)) as connexion:
        brut = connexion.execute("SELECT entrees, sorties, message_erreur FROM inferences").fetchone()
    assert brut == (None, None, None)   # les valeurs ont disparu de la base, pas seulement de l'affichage
    element, = journal.historique()
    assert element["statut"] == "invalide" and element["id_session"] == "demo-1"
    assert element["entrees"] is None


def test_historique_dans_l_ordre_d_enregistrement(tmp_path):
    journal = Journal(tmp_path / "journal.db")
    # Ancien format de date (heure locale, sans fuseau) : trié comme du texte, il passerait devant
    journal.enregistrer_inference("ancienne", "2026-10-03T23:44:44.795", None, "ok", entrees={"age": 30})
    journal.enregistrer_inference("recente", "2026-10-03T22:47:11.251+00:00", None, "ok", entrees={"age": 40})
    assert [e["id_prediction"] for e in journal.historique()] == ["recente", "ancienne"]


def test_reentrainements_traces(tmp_path):
    journal = Journal(tmp_path / "journal.db")
    journal.enregistrer_reentrainement("debut", "fin", "refuse_quality_gate", 3, "run-1", "3", "3",
                                       {"echecs": ["F1 macro = 0.300 (seuil min : 0.62)"]}, "diagnostic")
    trace, = journal.reentrainements()
    assert trace["statut"] == "refuse_quality_gate" and trace["n_feedbacks"] == 3
    assert "message_erreur" not in trace   # le diagnostic interne n'est pas renvoyé


def test_historique_avec_avis_et_situation_observee(tmp_path):
    journal = Journal(tmp_path / "journal.db")
    sorties = {"classe": 2, "risque_longue_duree": 0.46}
    journal.enregistrer_inference("p1", "2026-10-04T10:00:00.000+00:00", None, "ok", entrees={"age": 42},
                                  sorties=sorties, id_usager="DE-0248")
    journal.enregistrer_inference("p2", "2026-10-04T11:00:00.000+00:00", None, "ok", entrees={"age": 30},
                                  sorties={"classe": 0, "risque_longue_duree": 0.05})
    assert journal.classe_predite("p1") == 2 and journal.classe_predite("inconnue") is None
    journal.enregistrer_avis("p1", "d1", "confirme", 2)
    _, remplace = journal.enregistrer_avis("p1", "d2", "corrige", 1, "Autre", "précisions")
    journal.enregistrer_feedback("p1", "d3", 1)

    p2, p1 = journal.historique()
    assert p1["id_usager"] == "DE-0248" and remplace is True
    assert p1["avis_conseiller"] == {"avis": "corrige", "classe_proposee": 1, "motif": "Autre",
                                     "precisions": "précisions", "date": "d2"}   # précisions relisibles
    assert p1["situation_observee"] == {"classe_reelle": 1, "date": "d3"}
    assert p2["avis_conseiller"] is None and p2["situation_observee"] is None
    # L'avis du conseiller n'entre jamais dans les données de réentraînement : seuls les feedbacks y vont
    assert [f["id_prediction"] for f in journal.feedbacks_actuels()] == ["p1"]


def test_ancienne_base_sans_id_usager(tmp_path):
    chemin = tmp_path / "ancienne.db"
    with closing(sqlite3.connect(chemin)) as connexion, connexion:
        connexion.executescript(ANCIEN_SCHEMA)
    Journal(chemin).enregistrer_inference("p", "d", None, "ok", entrees={}, id_usager="DE-1")
    assert Journal(chemin).historique()[0]["id_usager"] == "DE-1"
