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
- CISIA_MIN_FEEDBACKS : nombre minimal de feedbacks pour lancer /retrain (défaut : 1) ;
- CISIA_SUIVI : dossier des rapports de suivi servis par /suivi (défaut : outputs/suivi).

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
from api.reentrainement import (
    AucunFeedbackUtilisable,
    EchecReentrainement,
    lancer_script,
    reentrainer,
    retablir_alias,
)
from api.schemas import (
    AvisConseiller,
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
from cisia.modele_mlflow import FICHIER_ACTUELLE, charger_modele_en_service, ecrire_en_une_operation

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
                      lanceur=lancer_script, dossier_suivi=None):
    """Construit l'API. Les paramètres servent aux tests (modèle, base, clé et données temporaires ;
    `lanceur` remplace le script d'entraînement). cle_api="" simule une clé non configurée."""
    dossier_production = Path(dossier_production
                              or os.getenv("CISIA_PRODUCTION", RACINE / "models" / "production"))
    journal = Journal(chemin_base or os.getenv("CISIA_BASE", RACINE / "outputs" / "cisia.db"))
    cle_attendue = cle_api if cle_api is not None else os.getenv("CISIA_CLE_API")
    donnees_entrainement = Path(donnees_entrainement or os.getenv(
        "CISIA_DONNEES", RACINE / "data" / "raw" / "dataset_trajectoire_emploi.csv"))
    min_feedbacks = int(os.getenv("CISIA_MIN_FEEDBACKS", "1"))
    dossier_suivi = Path(dossier_suivi or os.getenv("CISIA_SUIVI", RACINE / "outputs" / "suivi"))
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
        """Charge le modèle déclaré en service (au démarrage, puis après une mise en production).

        Le modèle et ses informations (version, run) forment UN seul objet, remplacé d'un coup :
        une prédiction en cours garde le couple qu'elle a lu au départ (jamais l'ancien modèle
        avec le numéro de la nouvelle version).
        """
        modele = charger_modele_en_service(dossier_production)
        infos = json.loads((dossier_production / FICHIER_ACTUELLE).read_text(encoding="utf-8"))
        application.state.service = (modele, infos)

    @asynccontextmanager
    async def cycle_de_vie(application):
        # Chargement du modèle au démarrage. S'il est absent, l'API démarre quand même :
        # /health répond 503 (pas prête) et /predict répond 503.
        application.state.service = None
        try:
            charger_modele(application)
        except FileNotFoundError:
            pass
        yield

    application = FastAPI(title="CISIA · Orientation des demandeurs d'emploi", version="1.2",
                          lifespan=cycle_de_vie)

    def version_de(service):
        if service is None:
            return None
        infos = service[1]
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
        service = application.state.service
        modele_charge = service is not None
        cle_configuree = bool(cle_attendue)
        pret = modele_charge and cle_configuree
        contenu = {"statut": "ok" if pret else "indisponible", "modele_charge": modele_charge,
                   "cle_configuree": cle_configuree,
                   "version_modele": version_de(service),
                   "run_id": service[1].get("run_id") if modele_charge else None}
        return JSONResponse(status_code=200 if pret else 503, content=contenu)

    @application.post("/predict", response_model=Prediction, dependencies=[Depends(verifier_cle)],
                      responses={**ERREUR_401, **ERREUR_422, **ERREUR_503,
                                 500: {"model": Erreur, "description": "Erreur interne (journalisée)"}})
    def predire(demande: DemandePrediction):
        """Recommandation, niveau d'alerte, risque de chômage de longue durée et explication (SHAP)."""
        id_prediction = str(uuid.uuid4())
        entrees = demande.usager.model_dump()
        service = application.state.service   # lu UNE fois : modèle et version restent cohérents
        if service is None:
            journal.enregistrer_inference(id_prediction, maintenant(), demande.id_session, "indisponible",
                                          entrees=entrees, message_erreur="aucun modèle en production",
                                          id_usager=demande.id_usager)
            raise HTTPException(status_code=503, detail="Aucun modèle en production.")

        debut = time.perf_counter()
        try:
            sortie = service[0].predict(vers_tableau(demande.usager)).iloc[0]
        except Exception as erreur:   # erreur inattendue : journalisée, puis réponse 500 sans détail interne
            journal.enregistrer_inference(id_prediction, maintenant(), demande.id_session, "erreur",
                                          entrees=entrees, version_modele=version_de(service),
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
            version_modele=version_de(service),
            explication=sortie.get("explication"),   # absente si le modèle date d'avant l'ajout de SHAP
        )
        journal.enregistrer_inference(id_prediction, maintenant(), demande.id_session, "ok",
                                      entrees=entrees, sorties=reponse.model_dump(),
                                      version_modele=reponse.version_modele, duree_ms=round(duree_ms, 2),
                                      id_usager=demande.id_usager)
        return reponse

    @application.get("/history", response_model=list[ElementHistorique],
                     dependencies=[Depends(verifier_cle)],
                     responses={**ERREUR_401, **ERREUR_422, **ERREUR_503})
    def historique(limite: int = Query(50, ge=1, le=500)):
        """Dernières requêtes de prédiction (les plus récentes en premier)."""
        return journal.historique(limite)

    @application.get("/suivi", dependencies=[Depends(verifier_cle)],
                     responses={**ERREUR_401, **ERREUR_503,
                                404: {"model": Erreur, "description": "Aucun rapport de suivi produit"}})
    def suivi():
        """Dernier rapport de suivi (indicateurs, alertes, dérive par semaine et par métier).

        Le rapport est produit par scripts/suivi.py (planifié chaque jour, à côté du journal et des
        données d'entraînement) ; l'API le sert tel quel. Réservé à l'équipe data en production
        (droits par rôle)."""
        chemin = dossier_suivi / "dernier_rapport.json"
        if not chemin.exists():
            raise HTTPException(status_code=404, detail="Aucun rapport de suivi : lancer scripts/suivi.py.")
        return json.loads(chemin.read_text(encoding="utf-8"))

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

    @application.post("/avis", status_code=201, dependencies=[Depends(verifier_cle)],
                      responses={**ERREUR_401, **ERREUR_422, **ERREUR_503,
                                 404: {"model": Erreur, "description": "Prédiction inconnue"}})
    def avis(retour: AvisConseiller):
        """Appréciation du conseiller à l'entretien : « confirme » ou « corrige » (classe et motif).

        Conservée pour le suivi de l'accord conseiller / modèle, JAMAIS utilisée pour réentraîner :
        seule la situation observée (/feedback) sert au réentraînement.
        """
        classe_predite = journal.classe_predite(retour.id_prediction)
        if classe_predite is None:
            raise HTTPException(status_code=404, detail="Prédiction inconnue.")
        classe = classe_predite if retour.avis == "confirme" else retour.classe_proposee
        id_avis, remplace = journal.enregistrer_avis(retour.id_prediction, maintenant(), retour.avis, classe,
                                                     retour.motif, retour.precisions)
        return {"id_avis": id_avis, "id_prediction": retour.id_prediction, "classe_proposee": classe,
                "remplace_un_avis_precedent": remplace}

    @application.post("/retrain", response_model=ResultatReentrainement, dependencies=[Depends(verifier_cle)],
                      responses={**ERREUR_401, **ERREUR_422, **ERREUR_503,
                                 409: {"model": Erreur, "description": "Pas assez de feedbacks, ou "
                                                                      "réentraînement déjà en cours"},
                                 500: {"model": Erreur,
                                       "description": "Échec (journalisé) ; la version précédente "
                                                      "reste en service"}})
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
            service_avant = application.state.service
            version_avant = version_de(service_avant)
            chemin_actuelle = dossier_production / FICHIER_ACTUELLE
            pointeur_avant = chemin_actuelle.read_bytes() if chemin_actuelle.exists() else None

            def lire():
                return chemin_actuelle.read_bytes() if chemin_actuelle.exists() else None

            def revenir_a_l_etat_precedent():
                """Après un échec : si le pointeur a changé (publication faite), le rétablir, remettre
                l'alias, puis VÉRIFIER. Renvoie (retour arrière confirmé ?, message, état structuré).

                L'état (pointeur rétabli ? alias rétabli ? vérifié ?) est enregistré dans le journal :
                le suivi (B10) distingue ainsi « pointeur rétabli » et « alias du registre rétabli »."""
                if lire() == pointeur_avant:
                    etat = {"necessaire": False, "pointeur_retabli": None, "alias_retabli": None,
                            "verifie": True}
                    return True, "pointeur inchangé : aucun retour arrière nécessaire", etat
                etat = {"necessaire": True, "pointeur_retabli": False, "alias_retabli": None,
                        "verifie": False}
                try:
                    if pointeur_avant is None:
                        chemin_actuelle.unlink(missing_ok=True)
                    else:
                        ecrire_en_une_operation(chemin_actuelle, pointeur_avant)
                except OSError as erreur:
                    return False, f"pointeur NON rétabli : {erreur!r}", etat
                etat["pointeur_retabli"] = True
                message = "pointeur rétabli"
                try:
                    # True : alias remis ; None : sans objet (pas de version du registre)
                    etat["alias_retabli"] = retablir_alias(
                        service_avant[1].get("version_registre") if service_avant else None)
                except Exception as erreur:
                    etat["alias_retabli"] = False
                    message += f" ; alias du registre NON rétabli : {erreur!r}"
                confirme = lire() == pointeur_avant
                etat["verifie"] = confirme
                return confirme, message + (" et vérifié" if confirme else " mais NON vérifié"), etat

            def echec(statut, erreur, run_id=None):
                confirme, retour, etat = revenir_a_l_etat_precedent()
                journal.enregistrer_reentrainement(date_debut, maintenant(), statut, len(feedbacks), run_id,
                                                   version_avant, version_de(application.state.service),
                                                   resultats={"retour_arriere": etat},
                                                   message_erreur=f"{erreur!r} ; {retour}")
                if confirme:
                    detail = f"Échec du réentraînement : la version {version_avant} reste en service."
                else:
                    detail = "Échec du réentraînement ET retour arrière non confirmé : voir le journal."
                return HTTPException(status_code=500, detail=detail)

            try:
                resultat = reentrainer(feedbacks, donnees_entrainement, dossier_production,
                                       demande.promouvoir, lanceur)
            except AucunFeedbackUtilisable as erreur:
                journal.enregistrer_reentrainement(date_debut, maintenant(), "aucun_feedback_utilisable", 0,
                                                   version_avant=version_avant,
                                                   version_apres=version_avant)
                raise HTTPException(status_code=409, detail="Aucun feedback utilisable : tous ont un profil "
                                                            "déjà présent dans le jeu de test.") from erreur
            except (EchecReentrainement, OSError, subprocess.SubprocessError) as erreur:
                # Y compris un échec APRÈS la publication (délai dépassé...) : retour arrière vérifié
                raise echec("erreur", erreur) from erreur
            if resultat["statut"] == "mis_en_production":
                try:
                    charger_modele(application)   # le nouveau modèle répond dès maintenant
                except Exception as erreur:
                    # Nouveau modèle publié mais impossible à charger : même retour arrière
                    raise echec("erreur_activation", erreur, resultat["run_id"]) from erreur
            version_apres = version_de(application.state.service)
            journal.enregistrer_reentrainement(
                date_debut, maintenant(), resultat["statut"], resultat["n_feedbacks"], resultat["run_id"],
                version_avant, version_apres,
                {"resultats_test": resultat["resultats_test"], "echecs": resultat["echecs_quality_gate"],
                 "comparaison_production": resultat["comparaison_production"]})
            return {**resultat, "version_modele_avant": version_avant, "version_modele": version_apres,
                    "duree_s": round(time.perf_counter() - debut, 1)}
        finally:
            verrou_reentrainement.release()

    return application


app = creer_application()
