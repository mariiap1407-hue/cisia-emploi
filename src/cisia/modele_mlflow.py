"""Le système de décision complet, sous la forme d'un modèle MLflow (« pyfunc »), et sa mise en production.

Le modèle enregistré dans le registre MLflow n'est pas seulement le classifieur LightGBM :
c'est toute la chaîne de décision (nettoyage, probabilités, recalibration, règle), avec les
fichiers du même entraînement. Le charger avec mlflow.pyfunc.load_model() donne donc exactement
le système évalué sur le jeu de test.

Mise en production : chaque version validée est écrite dans son propre dossier
models/production/<identifiant>/, qui n'est plus jamais modifié ensuite. Le fichier
models/production/actuelle.json indique la version en service. Changer de version revient à
réécrire ce petit fichier (pas de dossier renommé ni supprimé : sous Windows, l'antivirus ou
OneDrive peuvent bloquer ces opérations sur des fichiers tout juste écrits). La version
précédente reste disponible pour un retour en arrière.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import mlflow.pyfunc

from cisia import artefacts
from cisia.inference import predire_usagers

RACINE = Path(__file__).resolve().parents[2]
FICHIER_ACTUELLE = "actuelle.json"


class ModeleCISIA(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # context.artifacts : chemin local de chaque fichier enregistré avec le modèle
        self.composants = artefacts.charger_fichiers(context.artifacts)

    def predict(self, context, model_input, params=None):
        return predire_usagers(self.composants, model_input)


def options_modele(chemins):
    """Paramètres communs à mlflow.pyfunc.log_model() et save_model()."""
    return {
        "python_model": ModeleCISIA(),
        "artifacts": {nom: str(chemin) for nom, chemin in chemins.items()},
        "code_paths": [str(RACINE / "src" / "cisia")],
        "pip_requirements": str(RACINE / "requirements.txt"),
    }


def publier_en_production(chemins, dossier_production, identifiant, infos):
    """Écrit le modèle complet dans dossier_production/<identifiant>/, puis le déclare en service."""
    dossier_production = Path(dossier_production)
    dossier_production.mkdir(parents=True, exist_ok=True)
    mlflow.pyfunc.save_model(path=dossier_production / identifiant, **options_modele(chemins))

    # Déclaration de la version en service : fichier temporaire puis remplacement en une opération
    actuelle = {"dossier": identifiant, "date": datetime.now().isoformat(timespec="seconds"), **infos}
    temporaire = dossier_production / (FICHIER_ACTUELLE + ".tmp")
    temporaire.write_text(json.dumps(actuelle, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporaire, dossier_production / FICHIER_ACTUELLE)
    return dossier_production / identifiant


def dossier_en_service(dossier_production):
    """Dossier du modèle actuellement en service (lu dans actuelle.json)."""
    dossier_production = Path(dossier_production)
    chemin = dossier_production / FICHIER_ACTUELLE
    if not chemin.exists():
        raise FileNotFoundError(f"Aucun modèle en production dans {dossier_production} "
                                "(lancer scripts/entrainer.py --promouvoir).")
    actuelle = json.loads(chemin.read_text(encoding="utf-8"))
    return dossier_production / actuelle["dossier"]


def charger_modele_en_service(dossier_production):
    """Charge le modèle MLflow complet actuellement en service (utilisé par l'API)."""
    return mlflow.pyfunc.load_model(str(dossier_en_service(dossier_production)))
