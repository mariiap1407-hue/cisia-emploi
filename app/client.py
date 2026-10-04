"""Appels à l'API CISIA depuis l'interface (bibliothèque standard : aucune dépendance en plus).

La clé d'API reste CÔTÉ SERVEUR (variable CISIA_CLE_API, ou fichier .env) : le navigateur du
conseiller ne la reçoit jamais. Adresse de l'API : CISIA_API_URL (défaut : API locale).
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

URL_API = os.getenv("CISIA_API_URL", "http://127.0.0.1:8000").rstrip("/")


class ErreurAPI(Exception):
    """Réponse d'erreur de l'API, avec un message lisible pour le conseiller."""

    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details or []


def appeler(methode, chemin, corps=None, delai=30):
    entetes = {"Content-Type": "application/json", "X-API-Key": os.getenv("CISIA_CLE_API", "")}
    donnees = json.dumps(corps).encode("utf-8") if corps is not None else None
    requete = urllib.request.Request(f"{URL_API}{chemin}", data=donnees, headers=entetes, method=methode)
    try:
        with urllib.request.urlopen(requete, timeout=delai) as reponse:
            return json.loads(reponse.read() or b"null")
    except urllib.error.HTTPError as erreur:
        try:
            detail = json.loads(erreur.read() or b"null").get("detail")
        except (ValueError, AttributeError):
            detail = None
        if isinstance(detail, list):   # 422 : liste des champs refusés (format de l'API)
            raise ErreurAPI(erreur.code, "Certaines informations sont invalides.", detail) from erreur
        raise ErreurAPI(erreur.code, detail or f"Erreur {erreur.code} du service de prédiction.") from erreur
    except (urllib.error.URLError, TimeoutError, OSError) as erreur:
        raise ErreurAPI(0, "Le service de prédiction est injoignable pour le moment.") from erreur


def sante():
    try:
        return appeler("GET", "/health", delai=5)
    except ErreurAPI:
        return None


def predire(usager, id_usager):
    return appeler("POST", "/predict", {"usager": usager, "id_usager": id_usager or None})


def envoyer_avis(id_prediction, avis, classe_proposee=None, motif=None, precisions=None):
    corps = {"id_prediction": id_prediction, "avis": avis}
    if avis == "corrige":
        corps.update(classe_proposee=classe_proposee, motif=motif, precisions=precisions or None)
    return appeler("POST", "/avis", corps)


def envoyer_situation_observee(id_prediction, classe_reelle):
    return appeler("POST", "/feedback", {"id_prediction": id_prediction, "classe_reelle": classe_reelle})


def historique(limite=500):
    return appeler("GET", f"/history?limite={limite}")
