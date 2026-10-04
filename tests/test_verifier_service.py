"""scripts/verifier_service.py (smoke test de la CI) : il doit refuser un service incohérent.

Un faux service HTTP local (bibliothèque standard) répond comme l'API ; on vérifie que le script
accepte un service correct et rejette une mauvaise version, une classe impossible ou un service absent.
"""

import json
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[1]
CLE = "cle-de-test"


def faux_service(version_sante, version_prediction, classe, code_sante=200):
    class Gestionnaire(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def repondre(self, code, contenu):
            corps = json.dumps(contenu).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(corps)))
            self.end_headers()
            self.wfile.write(corps)

        def do_GET(self):
            self.repondre(code_sante, {"statut": "ok" if code_sante == 200 else "indisponible",
                                       "version_modele": version_sante})

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.headers.get("X-API-Key") != CLE:
                return self.repondre(401, {"detail": "Clé d'API absente ou invalide."})
            if self.path == "/retrain":
                return self.repondre(503, {"detail": "Données absentes."})
            self.repondre(200, {"id_prediction": "1", "classe": classe, "recommandation": "r",
                                "niveau_alerte": "Faible", "risque_longue_duree": 0.2,
                                "risque_affiche": "20%", "version_modele": version_prediction})

    serveur = HTTPServer(("127.0.0.1", 0), Gestionnaire)
    threading.Thread(target=serveur.serve_forever, daemon=True).start()
    return serveur


def verifier(url, *options):
    return subprocess.run([sys.executable, "scripts/verifier_service.py", url, "--cle", CLE, *options],
                          cwd=RACINE, capture_output=True, encoding="utf-8")


@pytest.mark.parametrize("version_prediction, classe, attendu", [
    ("controle-1", 1, 0),          # service correct
    ("AUTRE-VERSION", 1, 1),       # /predict ne sert pas la version attendue
    ("controle-1", 7, 1),          # classe impossible
])
def test_verification_du_service(version_prediction, classe, attendu):
    serveur = faux_service("controle-1", version_prediction, classe)
    try:
        url = f"http://127.0.0.1:{serveur.server_address[1]}"
        resultat = verifier(url, "--version", "controle-1", "--essais", "2", "--pause", "0.1",
                            "--retrain", "503")
        assert resultat.returncode == attendu, resultat.stdout + resultat.stderr
    finally:
        serveur.shutdown()


def test_service_absent():
    with socket.socket() as s:   # port libre, sur lequel rien n'écoute
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    resultat = verifier(f"http://127.0.0.1:{port}", "--essais", "2", "--pause", "0.1")
    assert resultat.returncode == 1 and "ÉCHEC" in resultat.stdout


def test_delai_maximal_depasse():
    """Service jamais prêt (/health 503) : au bout de --duree-max, échec, sans passer à la prédiction."""
    serveur = faux_service("controle-1", "controle-1", 1, code_sante=503)
    try:
        url = f"http://127.0.0.1:{serveur.server_address[1]}"
        resultat = verifier(url, "--essais", "1000", "--pause", "0.1", "--duree-max", "1")
        assert resultat.returncode == 1, resultat.stdout + resultat.stderr
        assert "délai maximal" in resultat.stdout and "Prédiction" not in resultat.stdout
    finally:
        serveur.shutdown()
