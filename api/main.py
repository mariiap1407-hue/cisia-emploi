"""API FastAPI : santé, prédiction, historique, feedback des conseillers et réentraînement monitoré.

Lancement (depuis la racine du projet) :
    uvicorn api.main:app --reload
Documentation interactive : http://127.0.0.1:8000/docs (bouton « Authorize » pour la clé d'API)

Configuration (variables d'environnement, ou fichier .env à la racine, jamais versionné) :
- CISIA_CLE_API : clé exigée dans l'en-tête X-API-Key (toutes les routes sauf /health) ;
  sans clé configurée, ces routes répondent 503 : l'API n'est jamais ouverte par défaut ;
- CISIA_PRODUCTION : dossier des modèles en production (défaut : models/production) ;
- CISIA_BASE : base SQLite du journal et des feedbacks (défaut : outputs/cisia.db) ;
- CISIA_DONNEES : données d'entraînement d'origine pour /retrain
  (défaut : data/raw/dataset_trajectoire_emploi.csv ; absentes = /retrain répond 503) ;
- CISIA_MIN_FEEDBACKS : nombre minimal de feedbacks pour lancer /retrain (défaut : 1).

/health = disponibilité : 200 seulement si un modèle est chargé ET une clé d'API configurée.

Limite assumée : une clé unique, partagée par les postes autorisés. Dans le SI de l'agence,
l'authentification passerait par l'annuaire des conseillers (SSO), avec des droits par rôle :
qui peut consulter l'historique, qui peut saisir un feedback.
"""

import json
import os
import secrets
import subprocess
import threading
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
from api.reentrainement import EchecReentrainement, lancer_script, reentrainer
from api.schemas import (
    DemandePrediction,
    ElementHistorique,
    Erreur,
    ErreurValidation,
    Feedback,
    Prediction,
    Reentrainement,
    ResultatReentrainement,
    Sante,
    erreurs_sans_valeurs,
    vers_tableau,
)
from cisia.modele_mlflow import FICHIER_ACTUELLE, charger_modele_en_service

RACINE = Path(__file__).resolve().parents[1]
load_dotenv(RACINE / ".env")

ENTETE_CLE = APIKeyHeader(name="X-API-Key", auto_error=False)

# Réponses d'erreur documentées dans /docs (format réellement renvoyé)
ERREUR_401 = {401: {"model": Erreur, "description": "Clé d'API absente ou invalide"}}
ERREUR_422 = {422: {"model": ErreurValidation, "description": "Requête invalide : champ, type et message, "
                                                              "sans la valeur reçue"}}
ERREUR_503 = {503: {"model": Erreur, "description": "Service non prêt (clé d'API non configurée, "
                                                    "aucun modèle en production...)"}}


def maintenant():
    """Date et heure en UTC, avec le fuseau : sans ambiguïté dans les journaux."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def creer_application(dossier_production=None, chemin_base=None, cle_api=None, donnees_entrainement=None,
                      lanceur=lancer_script):
    """Construit l'API. Les paramètres servent aux tests (modèle, base, clé et données temporaires ;
    `lanceur` remplace le script d'entraînement). cle_api="" simule une clé non configurée."""
    dossier_production = Path(dossier_production
                              or os.getenv("CISIA_PRODUCTION", RACINE / "models" / "production"))
    journal = Journal(chemin_base or os.getenv("CISIA_BASE", RACINE / "outputs" / "cisia.db"))
    cle_attendue = cle_api if cle_api is not None else os.getenv("CISIA_CLE_API")
    donnees_entrainement = Path(donnees_entrainement or os.getenv(
        "CISIA_DONNEES", RACINE / "data" / "raw" / "dataset_trajectoire_emploi.csv"))
    min_feedbacks = int(os.getenv("CISIA_MIN_FEEDBACKS", "1"))
    verrou_reentrainement = threading.Lock()   # un seul réentraînement à la fois

    def verifier_cle(cle_fournie: str | None = Security(ENTETE_CLE)):
        if not cle_attendue:
            raise HTTPException(status_code=503, detail="API non configurée : CISIA_CLE_API absente.")
        # Comparaison en octets : une clé reçue avec des caractères accentués est simplement fausse
        # (401), au lieu de faire échouer la comparaison (500)
        if not cle_fournie or not secrets.compare_digest(cle_fournie.encode("utf-8"),
                                                         cle_attendue.encode("utf-8")):
            raise HTTPException(status_code=401, detail="Clé d'API absente ou invalide.")

    def charger_modele(application):
        """Charge le modèle déclaré en service (au démarrage, puis après une mise en production)."""
        modele = charger_modele_en_service(dossier_production)
        infos = json.loads((dossier_production / FICHIER_ACTUELLE).read_text(encoding="utf-8"))
        application.state.modele, application.state.infos_modele = modele, infos

    @asynccontextmanager
    async def cycle_de_vie(application):
        # Chargement du modèle au démarrage. S'il est absent, l'API démarre quand même :
        # /health répond 503 (pas prête) et /predict répond 503.
        application.state.modele, application.state.infos_modele = None, {}
        try:
            charger_modele(application)
        except FileNotFoundError:
            pass
        yield

    application = FastAPI(title="CISIA · Orientation des demandeurs d'emploi", version="1.2",
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

    @application.get("/health", response_model=Sante,
                     responses={503: {"model": Sante, "description": "Service non prêt"}})
    def sante():
        """Le service est-il prêt à prédire ? 200 si un modèle est chargé ET une clé configurée, 503 sinon."""
        modele_charge = application.state.modele is not None
        cle_configuree = bool(cle_attendue)
        pret = modele_charge and cle_configuree
        contenu = {"statut": "ok" if pret else "indisponible", "modele_charge": modele_charge,
                   "cle_configuree": cle_configuree,
                   "version_modele": version_modele() if modele_charge else None,
                   "run_id": application.state.infos_modele.get("run_id")}
        return JSONResponse(status_code=200 if pret else 503, content=contenu)

    @application.post("/predict", response_model=Prediction, dependencies=[Depends(verifier_cle)],
                      responses={**ERREUR_401, **ERREUR_422, **ERREUR_503,
                                 500: {"model": Erreur, "description": "Erreur interne (journalisée)"}})
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
                     dependencies=[Depends(verifier_cle)],
                     responses={**ERREUR_401, **ERREUR_422, **ERREUR_503})
    def historique(limite: int = Query(50, ge=1, le=500)):
        """Dernières requêtes de prédiction (les plus récentes en premier)."""
        return journal.historique(limite)

    @application.post("/feedback", status_code=201, dependencies=[Depends(verifier_cle)],
                      responses={**ERREUR_401, **ERREUR_422, **ERREUR_503,
                                 404: {"model": Erreur, "description": "Prédiction inconnue"}})
    def feedback(retour: Feedback):
        """Classe de retour à l'emploi OBSERVÉE pour une prédiction ; le plus récent feedback fait foi."""
        if not journal.prediction_existe(retour.id_prediction):
            raise HTTPException(status_code=404, detail="Prédiction inconnue.")
        id_feedback, remplace = journal.enregistrer_feedback(retour.id_prediction, maintenant(),
                                                             retour.classe_reelle, retour.commentaire)
        return {"id_feedback": id_feedback, "id_prediction": retour.id_prediction,
                "remplace_un_feedback_precedent": remplace}

    @application.post("/retrain", response_model=ResultatReentrainement, dependencies=[Depends(verifier_cle)],
                      responses={**ERREUR_401, **ERREUR_422, **ERREUR_503,
                                 409: {"model": Erreur, "description": "Pas assez de feedbacks, ou "
                                                                      "réentraînement déjà en cours"},
                                 500: {"model": Erreur,
                                       "description": "Échec du réentraînement (journalisé) ; "
                                                      "la production n'est pas modifiée"}})
    def retrain(demande: Reentrainement | None = None):
        """Réentraîne le modèle avec les feedbacks des conseillers (réentraînement monitoré).

        Données : fichier d'origine + feedbacks qui font foi (le plus récent par prédiction), ajoutés à
        l'entraînement seulement. Même script et mêmes garde-fous que l'entraînement manuel : quality gate
        sur le même jeu de test, version enregistrée dans MLflow, mise en service seulement si le gate
        passe (et si promouvoir = true). Chaque demande est tracée dans le journal.
        """
        demande = demande or Reentrainement()
        if not donnees_entrainement.exists():
            raise HTTPException(status_code=503,
                                detail="Données d'entraînement d'origine absentes sur ce serveur.")
        feedbacks = journal.feedbacks_actuels()
        if len(feedbacks) < min_feedbacks:
            raise HTTPException(status_code=409, detail=f"Pas assez de feedbacks : {len(feedbacks)} "
                                                        f"(minimum {min_feedbacks}).")
        if not verrou_reentrainement.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Un réentraînement est déjà en cours.")
        try:
            debut, date_debut = time.perf_counter(), maintenant()
            version_avant = version_modele() if application.state.modele is not None else None
            try:
                resultat = reentrainer(feedbacks, donnees_entrainement, dossier_production,
                                       demande.promouvoir, lanceur)
            except (EchecReentrainement, OSError, subprocess.SubprocessError) as erreur:
                journal.enregistrer_reentrainement(date_debut, maintenant(), "erreur", len(feedbacks),
                                                   version_avant=version_avant, version_apres=version_avant,
                                                   message_erreur=repr(erreur))
                raise HTTPException(status_code=500, detail="Échec du réentraînement ; la production "
                                                            "n'est pas modifiée.") from erreur
            if resultat["statut"] == "mis_en_production":
                charger_modele(application)   # le nouveau modèle répond dès maintenant
            version_apres = version_modele() if application.state.modele is not None else None
            journal.enregistrer_reentrainement(
                date_debut, maintenant(), resultat["statut"], len(feedbacks), resultat["run_id"],
                version_avant, version_apres,
                {"resultats_test": resultat["resultats_test"], "echecs": resultat["echecs_quality_gate"]})
            return {**resultat, "version_modele_avant": version_avant, "version_modele": version_apres,
                    "duree_s": round(time.perf_counter() - debut, 1)}
        finally:
            verrou_reentrainement.release()

    return application


app = creer_application()
