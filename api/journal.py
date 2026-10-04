"""Journal des requêtes, historique des inférences et feedbacks des conseillers (base SQLite).

La base (outputs/cisia.db par défaut) contient des profils d'usagers : elle reste sur le serveur,
n'est jamais versionnée (.gitignore) et devrait avoir une durée de conservation définie.

Politique de journalisation :
- requête acceptée : entrées validées (seulement les champs demandés), sorties, version, durée ;
- requête refusée : date, statut, identifiant de session et erreurs SANS les valeurs reçues ;
- erreur interne : le message technique est conservé pour le diagnostic, mais jamais montré
  dans l'historique.
Les dates sont en UTC, avec le fuseau (ex. 2026-10-04T08:15:00.000+00:00). L'historique est trié
par ordre d'enregistrement (et non par la date en texte, qui dépend du format).

Réentraînements : chaque demande est tracée (début, fin, statut, nombre de feedbacks, run MLflow,
version avant / après, indicateurs du quality gate). C'est la base du suivi (B10).
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
    message_erreur TEXT,           -- diagnostic interne, jamais affiché
    erreurs_validation TEXT        -- JSON, sans les valeurs reçues
);
CREATE TABLE IF NOT EXISTS reentrainements (
    id_reentrainement INTEGER PRIMARY KEY AUTOINCREMENT,
    date_debut TEXT NOT NULL,
    date_fin TEXT,
    statut TEXT NOT NULL,          -- mis_en_production, candidat_non_promu, refuse_quality_gate,
                                   -- erreur, erreur_activation (retour à la version précédente)
    n_feedbacks INTEGER,
    run_id TEXT,
    version_avant TEXT,
    version_apres TEXT,
    resultats TEXT,                -- JSON : indicateurs du jeu de test et échecs du quality gate
    message_erreur TEXT            -- diagnostic interne, jamais renvoyé par l'API
);
CREATE TABLE IF NOT EXISTS feedbacks (
    id_feedback INTEGER PRIMARY KEY AUTOINCREMENT,
    id_prediction TEXT NOT NULL REFERENCES inferences(id_prediction),
    date TEXT NOT NULL,
    classe_reelle INTEGER NOT NULL,
    commentaire TEXT
);
"""
COLONNES_PUBLIQUES = ["id_prediction", "date", "id_session", "statut", "entrees", "sorties",
                      "version_modele", "duree_ms", "erreurs_validation"]


def en_json(valeur):
    return json.dumps(valeur, ensure_ascii=False, allow_nan=False) if valeur is not None else None


class Journal:
    def __init__(self, chemin):
        self.chemin = Path(chemin)
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connexion()) as connexion, connexion:
            connexion.executescript(SCHEMA)
            # Base créée par une version précédente : ajout de la colonne manquante
            colonnes = [ligne["name"] for ligne in connexion.execute("PRAGMA table_info(inferences)")]
            if "erreurs_validation" not in colonnes:
                connexion.execute("ALTER TABLE inferences ADD COLUMN erreurs_validation TEXT")
            # Requêtes refusées enregistrées par une version précédente (avant le bloc 5b) avec les
            # valeurs reçues : ces valeurs (et l'ancien diagnostic qui les contenait) sont effacées
            connexion.execute("UPDATE inferences SET entrees = NULL, sorties = NULL, message_erreur = NULL "
                              "WHERE statut = 'invalide' AND (entrees IS NOT NULL OR sorties IS NOT NULL "
                              "OR message_erreur IS NOT NULL)")

    def _connexion(self):
        connexion = sqlite3.connect(self.chemin)
        connexion.row_factory = sqlite3.Row
        return connexion

    def enregistrer_inference(self, id_prediction, date, id_session, statut, entrees=None, sorties=None,
                              version_modele=None, duree_ms=None, message_erreur=None,
                              erreurs_validation=None):
        with closing(self._connexion()) as connexion, connexion:
            connexion.execute(
                "INSERT INTO inferences (id_prediction, date, id_session, statut, entrees, sorties, "
                "version_modele, duree_ms, message_erreur, erreurs_validation) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (id_prediction, date, id_session, statut, en_json(entrees), en_json(sorties),
                 version_modele, duree_ms, message_erreur, en_json(erreurs_validation)),
            )

    def historique(self, limite=50):
        """Dernières requêtes, au format public (sans le diagnostic interne)."""
        with closing(self._connexion()) as connexion:
            lignes = connexion.execute(
                f"SELECT {', '.join(COLONNES_PUBLIQUES)} FROM inferences ORDER BY rowid DESC LIMIT ?",
                (limite,)).fetchall()
        historique = []
        for ligne in lignes:
            element = dict(ligne)
            if element["statut"] == "invalide":   # sécurité : jamais de valeur d'une requête refusée
                element["entrees"] = element["sorties"] = None
            for champ in ["entrees", "sorties", "erreurs_validation"]:
                element[champ] = json.loads(element[champ]) if element[champ] else None
            historique.append(element)
        return historique

    def prediction_existe(self, id_prediction):
        with closing(self._connexion()) as connexion:
            ligne = connexion.execute("SELECT 1 FROM inferences WHERE id_prediction = ? AND statut = 'ok'",
                                      (id_prediction,)).fetchone()
        return ligne is not None

    def enregistrer_feedback(self, id_prediction, date, classe_reelle, commentaire=None):
        """Enregistre un feedback ; renvoie son identifiant et s'il remplace un feedback précédent."""
        with closing(self._connexion()) as connexion, connexion:
            deja = connexion.execute("SELECT COUNT(*) FROM feedbacks WHERE id_prediction = ?",
                                     (id_prediction,)).fetchone()[0]
            curseur = connexion.execute(
                "INSERT INTO feedbacks (id_prediction, date, classe_reelle, commentaire) VALUES (?, ?, ?, ?)",
                (id_prediction, date, classe_reelle, commentaire))
            return curseur.lastrowid, deja > 0

    def feedbacks_actuels(self):
        """Le feedback qui fait foi pour chaque prédiction (le plus récent), avec les entrées associées.

        C'est ce que le réentraînement utilisera : un exemple par prédiction, jamais de doublon.
        """
        with closing(self._connexion()) as connexion:
            lignes = connexion.execute("""
                SELECT f.id_prediction, f.date, f.classe_reelle, i.entrees
                FROM feedbacks f JOIN inferences i ON i.id_prediction = f.id_prediction
                WHERE f.id_feedback = (SELECT MAX(g.id_feedback) FROM feedbacks g
                                       WHERE g.id_prediction = f.id_prediction)
                ORDER BY f.date""").fetchall()
        return [{**dict(ligne), "entrees": json.loads(ligne["entrees"])} for ligne in lignes]

    def enregistrer_reentrainement(self, date_debut, date_fin, statut, n_feedbacks, run_id=None,
                                   version_avant=None, version_apres=None, resultats=None,
                                   message_erreur=None):
        with closing(self._connexion()) as connexion, connexion:
            connexion.execute(
                "INSERT INTO reentrainements (date_debut, date_fin, statut, n_feedbacks, run_id, "
                "version_avant, version_apres, resultats, message_erreur) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (date_debut, date_fin, statut, n_feedbacks, run_id, version_avant, version_apres,
                 en_json(resultats), message_erreur))

    def reentrainements(self):
        """Tous les réentraînements demandés, du plus récent au plus ancien (sans le diagnostic interne)."""
        with closing(self._connexion()) as connexion:
            lignes = connexion.execute(
                "SELECT id_reentrainement, date_debut, date_fin, statut, n_feedbacks, run_id, version_avant, "
                "version_apres, resultats FROM reentrainements ORDER BY id_reentrainement DESC").fetchall()
        return [{**dict(ligne), "resultats": json.loads(ligne["resultats"]) if ligne["resultats"] else None}
                for ligne in lignes]
