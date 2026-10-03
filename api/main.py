"""API FastAPI : santé, prédiction, historique et feedback des conseillers.

Lancement (depuis la racine du projet) :
    uvicorn api.main:app --reload
Documentation interactive : http://127.0.0.1:8000/docs (bouton « Authorize » pour la clé d'API)

Configuration (variables d'environnement, ou fichier .env à la racine, jamais versionné) :
- CISIA_CLE_API : clé exigée dans l'en-tête X-API-Key (toutes les routes sauf /health) ;
  sans clé configurée, ces routes répondent 503 : l'API n'est jamais ouverte par défaut ;
- CISIA_PRODUCTION : dossier des modèles en production (défaut : models/production) ;
- CISIA_BASE : base SQLite du journal et des feedbacks (défaut : outputs/cisia.db).

Limite assumée : une clé unique, partagée par les postes autorisés. Dans le SI de l'agence,
l'authentification passerait par l'annuaire des conseillers (SSO), avec des droits par rôle :
qui peut consulter l'historique, qui peut saisir un feedback.
"""

import json
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from api.journal import Journal
from api.schemas import (
    DemandePrediction,
    ElementHistorique,
    Feedback,
    Prediction,
    erreurs_sans_valeurs,
    vers_tableau,
)
from cisia.modele_mlflow import FICHIER_ACTUELLE, charger_modele_en_service

RACINE = Path(__file__).resolve().parents[1]
load_dotenv(RACINE / ".env")

ENTETE_CLE = APIKeyHeader(name="X-API-Key", auto_error=False)


def maintenant():
    """Date et heure en UTC, avec le fuseau : sans ambiguïté dans les journaux."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def creer_application(dossier_production=None, chemin_base=None, cle_api=None):
    """Construit l'API. Les paramètres servent aux tests (modèle, base et clé temporaires)."""
    dossier_production = Path(dossier_production
                              or os.getenv("CISIA_PRODUCTION", RACINE / "models" / "production"))
    journal = Journal(chemin_base or os.getenv("CISIA_BASE", RACINE / "outputs" / "cisia.db"))
    cle_attendue = cle_api or os.getenv("CISIA_CLE_API")

    def verifier_cle(cle_fournie: str | None = Security(ENTETE_CLE)):
        if not cle_attendue:
            raise HTTPException(status_code=503, detail="API non configurée : CISIA_CLE_API absente.")
        if not cle_fournie or not secrets.compare_digest(cle_fournie, cle_attendue):
            raise HTTPException(status_code=401, detail="Clé d'API absente ou invalide.")

    @asynccontextmanager
    async def cycle_de_vie(application):
        # Chargement du modèle au démarrage. S'il est absent, l'API démarre quand même :
        # /health répond 503 (pas prête) et /predict répond 503.
        application.state.modele, application.state.infos_modele = None, {}
        try:
            application.state.modele = charger_modele_en_service(dossier_production)
            application.state.infos_modele = json.loads(
                (dossier_production / FICHIER_ACTUELLE).read_text(encoding="utf-8"))
        except FileNotFoundError:
            pass
        yield

    application = FastAPI(title="CISIA · Orientation des demandeurs d'emploi", version="1.1",
                          lifespan=cycle_de_vie)

    def version_modele():
        infos = application.state.infos_modele
        return str(infos.get("version_registre") or infos.get("run_id") or "inconnue")

    @application.exception_handler(RequestValidationError)
    async def entree_invalide(request: Request, erreur: RequestValidationError):
        # Réponse et journal SANS les valeurs reçues (données non demandées, nombres non finis...)
        erreurs = erreurs_sans_valeurs(erreur.errors())
        if request.url.path == "/predict":
            corps = erreur.body if isinstance(erreur.body, dict) else {}
            id_session = corps.get("id_session")
            id_session = id_session if isinstance(id_session, str) and len(id_session) <= 64 else None
            journal.enregistrer_inference(str(uuid.uuid4()), maintenant(), id_session, "invalide",
                                          erreurs_validation=erreurs)
        return JSONResponse(status_code=422, content={"detail": erreurs})

    @application.get("/health")
    def sante():
        """Le service est-il prêt à prédire ? 200 si un modèle est chargé, 503 sinon."""
        pret = application.state.modele is not None
        contenu = {"statut": "ok" if pret else "indisponible", "modele_charge": pret,
                   "version_modele": version_modele() if pret else None,
                   "run_id": application.state.infos_modele.get("run_id")}
        return JSONResponse(status_code=200 if pret else 503, content=contenu)

    @application.post("/predict", response_model=Prediction, dependencies=[Depends(verifier_cle)])
    def predire(demande: DemandePrediction):
        """Recommandation, niveau d'alerte et risque de chômage de longue durée pour un usager."""
        id_prediction = str(uuid.uuid4())
        entrees = demande.usager.model_dump()
        if application.state.modele is None:
            journal.enregistrer_inference(id_prediction, maintenant(), demande.id_session, "indisponible",
                                          entrees=entrees, message_erreur="aucun modèle en production")
            raise HTTPException(status_code=503, detail="Aucun modèle en production.")

        debut = time.perf_counter()
        try:
            sortie = application.state.modele.predict(vers_tableau(demande.usager)).iloc[0]
        except Exception as erreur:   # erreur inattendue : journalisée, puis réponse 500 sans détail interne
            journal.enregistrer_inference(id_prediction, maintenant(), demande.id_session, "erreur",
                                          entrees=entrees, version_modele=version_modele(),
                                          message_erreur=repr(erreur))
            raise HTTPException(status_code=500, detail="Erreur lors de la prédiction.") from erreur
        duree_ms = (time.perf_counter() - debut) * 1000

        reponse = Prediction(
            id_prediction=id_prediction,
            classe=int(sortie["classe"]),
            recommandation=sortie["recommandation"],
            niveau_alerte=sortie["niveau_alerte"],
            risque_longue_duree=round(float(sortie["risque_longue_duree"]), 4),
            risque_affiche=f"{float(sortie['risque_longue_duree']):.0%}",
            version_modele=version_modele(),
        )
        journal.enregistrer_inference(id_prediction, maintenant(), demande.id_session, "ok",
                                      entrees=entrees, sorties=reponse.model_dump(),
                                      version_modele=reponse.version_modele, duree_ms=round(duree_ms, 2))
        return reponse

    @application.get("/history", response_model=list[ElementHistorique],
                     dependencies=[Depends(verifier_cle)])
    def historique(limite: int = Query(50, ge=1, le=500)):
        """Dernières requêtes de prédiction (les plus récentes en premier)."""
        return journal.historique(limite)

    @application.post("/feedback", status_code=201, dependencies=[Depends(verifier_cle)])
    def feedback(retour: Feedback):
        """Classe de retour à l'emploi OBSERVÉE pour une prédiction ; le plus récent feedback fait foi."""
        if not journal.prediction_existe(retour.id_prediction):
            raise HTTPException(status_code=404, detail="Prédiction inconnue.")
        id_feedback, remplace = journal.enregistrer_feedback(retour.id_prediction, maintenant(),
                                                             retour.classe_reelle, retour.commentaire)
        return {"id_feedback": id_feedback, "id_prediction": retour.id_prediction,
                "remplace_un_feedback_precedent": remplace}

    return application


app = creer_application()
