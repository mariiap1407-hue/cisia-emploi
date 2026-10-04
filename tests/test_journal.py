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
