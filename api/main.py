"""API FastAPI : santé, prédiction, historique et feedback des conseillers.

Lancement (depuis la racine du projet) :
    uvicorn api.main:app --reload
Documentation interactive : http://127.0.0.1:8000/docs

Variables d'environnement (facultatives) :
- CISIA_PRODUCTION : dossier des modèles en production (défaut : models/production) ;
- CISIA_BASE : base SQLite du journal et des feedbacks (défaut : outputs/cisia.db).
"""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.journal import Journal
from api.schemas import DemandePrediction, Feedback, Prediction, vers_tableau
from cisia.modele_mlflow import FICHIER_ACTUELLE, charger_modele_en_service

RACINE = Path(__file__).resolve().parents[1]


def maintenant():
    return datetime.now().isoformat(timespec="milliseconds")


def creer_application(dossier_production=None, chemin_base=None):
    """Construit l'API. Les paramètres permettent aux tests d'utiliser un modèle et une base temporaires."""
    dossier_production = Path(dossier_production
                              or os.getenv("CISIA_PRODUCTION", RACINE / "models" / "production"))
    journal = Journal(chemin_base or os.getenv("CISIA_BASE", RACINE / "outputs" / "cisia.db"))

    @asynccontextmanager
    async def cycle_de_vie(application):
        # Chargement du modèle au démarrage. S'il est absent, l'API démarre quand même :
        # /health le signale et /predict répond 503 (service indisponible).
        application.state.modele, application.state.infos_modele = None, {}
        try:
            application.state.modele = charger_modele_en_service(dossier_production)
            application.state.infos_modele = json.loads(
                (dossier_production / FICHIER_ACTUELLE).read_text(encoding="utf-8"))
        except FileNotFoundError:
            pass
        yield

    application = FastAPI(title="CISIA · Orientation des demandeurs d'emploi", version="1.0",
                          lifespan=cycle_de_vie)

    def version_modele():
        infos = application.state.infos_modele
        return str(infos.get("version_registre") or infos.get("run_id") or "inconnue")

    @application.exception_handler(RequestValidationError)
    async def entree_invalide(request: Request, erreur: RequestValidationError):
        # Les requêtes refusées sont aussi journalisées (statut « invalide »)
        if request.url.path == "/predict":
            journal.enregistrer_inference(str(uuid.uuid4()), maintenant(), None, "invalide",
                                          entrees=jsonable_encoder(erreur.body),
                                          message_erreur=json.dumps(jsonable_encoder(erreur.errors()),
                                                                    ensure_ascii=False))
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(erreur.errors())})

    @application.get("/health")
    def sante():
        """État du service : le modèle est-il chargé, et lequel ?"""
        return {"statut": "ok", "modele_charge": application.state.modele is not None,
                "version_modele": version_modele() if application.state.modele is not None else None,
                "run_id": application.state.infos_modele.get("run_id")}

    @application.post("/predict", response_model=Prediction)
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

    @application.get("/history")
    def historique(limite: int = 50):
        """Dernières requêtes de prédiction (les plus récentes en premier)."""
        if not 1 <= limite <= 500:
            raise HTTPException(status_code=422, detail="limite doit être comprise entre 1 et 500.")
        return journal.historique(limite)

    @application.post("/feedback", status_code=201)
    def feedback(retour: Feedback):
        """Classe constatée par le conseiller pour une prédiction : servira au réentraînement."""
        if not journal.prediction_existe(retour.id_prediction):
            raise HTTPException(status_code=404, detail="Prédiction inconnue.")
        id_feedback = journal.enregistrer_feedback(retour.id_prediction, maintenant(),
                                                   retour.classe_reelle, retour.commentaire)
        return {"id_feedback": id_feedback, "id_prediction": retour.id_prediction}

    return application


app = creer_application()
