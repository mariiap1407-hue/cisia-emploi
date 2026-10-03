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
import tempfile
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


def publier_en_production(chemins, dossier_production, identifiant, infos, publier_alias=None):
    """Met en service le modèle complet : dossier_production/<identifiant>/, puis actuelle.json, puis alias.

    Le fichier actuelle.json fait foi (c'est lui que lit l'API). L'alias du registre MLflow doit
    désigner la même version : si sa mise à jour échoue (publier_alias lève une erreur), l'ancien
    actuelle.json est rétabli, pour que l'API et le registre ne désignent jamais deux versions
    différentes. Une seule mise en production à la fois est prévue (entraînements successifs).
    """
    dossier_production = Path(dossier_production)
    dossier_production.mkdir(parents=True, exist_ok=True)
    mlflow.pyfunc.save_model(path=dossier_production / identifiant, **options_modele(chemins))

    chemin_actuelle = dossier_production / FICHIER_ACTUELLE
    ancienne = chemin_actuelle.read_bytes() if chemin_actuelle.exists() else None
    actuelle = {"dossier": identifiant, "date": datetime.now().isoformat(timespec="seconds"), **infos}
    contenu = json.dumps(actuelle, ensure_ascii=False, indent=2).encode("utf-8")
    ecrire_en_une_operation(chemin_actuelle, contenu)

    if publier_alias is not None:
        try:
            publier_alias()
        except Exception:
            # Retour à la situation précédente : l'ancienne version reste en service
            if ancienne is None:
                chemin_actuelle.unlink()
            else:
                ecrire_en_une_operation(chemin_actuelle, ancienne)
            raise
    return dossier_production / identifiant


def ecrire_en_une_operation(chemin, contenu):
    """Écrit un fichier temporaire au nom unique, puis remplace le fichier visé en une seule opération."""
    descripteur, temporaire = tempfile.mkstemp(dir=chemin.parent, prefix=chemin.name + ".", suffix=".tmp")
    with os.fdopen(descripteur, "wb") as fichier:
        fichier.write(contenu)
    os.replace(temporaire, chemin)


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
