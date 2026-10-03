"""Journal des requêtes, historique des inférences et feedbacks des conseillers (base SQLite).

La base (outputs/cisia.db par défaut) contient des profils d'usagers : elle reste sur le serveur,
n'est jamais versionnée (.gitignore) et devrait avoir une durée de conservation définie.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS inferences (
    id_prediction TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    id_session TEXT,
    statut TEXT NOT NULL,          -- ok, invalide, indisponible, erreur
    entrees TEXT,                  -- JSON
    sorties TEXT,                  -- JSON
    version_modele TEXT,
    duree_ms REAL,
    message_erreur TEXT
);
CREATE TABLE IF NOT EXISTS feedbacks (
    id_feedback INTEGER PRIMARY KEY AUTOINCREMENT,
    id_prediction TEXT NOT NULL REFERENCES inferences(id_prediction),
    date TEXT NOT NULL,
    classe_reelle INTEGER NOT NULL,
    commentaire TEXT
);
"""


class Journal:
    def __init__(self, chemin):
        self.chemin = Path(chemin)
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connexion()) as connexion, connexion:
            connexion.executescript(SCHEMA)

    def _connexion(self):
        connexion = sqlite3.connect(self.chemin)
        connexion.row_factory = sqlite3.Row
        return connexion

    def enregistrer_inference(self, id_prediction, date, id_session, statut, entrees=None, sorties=None,
                              version_modele=None, duree_ms=None, message_erreur=None):
        with closing(self._connexion()) as connexion, connexion:
            connexion.execute(
                "INSERT INTO inferences VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (id_prediction, date, id_session, statut,
                 json.dumps(entrees, ensure_ascii=False) if entrees is not None else None,
                 json.dumps(sorties, ensure_ascii=False) if sorties is not None else None,
                 version_modele, duree_ms, message_erreur),
            )

    def historique(self, limite=50):
        with closing(self._connexion()) as connexion:
            lignes = connexion.execute(
                "SELECT * FROM inferences ORDER BY date DESC LIMIT ?", (limite,)).fetchall()
        historique = []
        for ligne in lignes:
            element = dict(ligne)
            for champ in ["entrees", "sorties"]:
                element[champ] = json.loads(element[champ]) if element[champ] else None
            historique.append(element)
        return historique

    def prediction_existe(self, id_prediction):
        with closing(self._connexion()) as connexion:
            ligne = connexion.execute("SELECT 1 FROM inferences WHERE id_prediction = ? AND statut = 'ok'",
                                      (id_prediction,)).fetchone()
        return ligne is not None

    def enregistrer_feedback(self, id_prediction, date, classe_reelle, commentaire=None):
        with closing(self._connexion()) as connexion, connexion:
            curseur = connexion.execute(
                "INSERT INTO feedbacks (id_prediction, date, classe_reelle, commentaire) VALUES (?, ?, ?, ?)",
                (id_prediction, date, classe_reelle, commentaire))
            return curseur.lastrowid
