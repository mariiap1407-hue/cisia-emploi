"""Format des requêtes et des réponses de l'API, avec la validation des entrées (Pydantic).

Minimisation des données (RGPD) : l'API ne demande que les informations utilisées par le modèle
retenu (candidat B) : âge, diplôme, ancienneté, métier visé et synthèse de l'entretien.
Ni la nationalité, ni la commune, ni le statut d'allocataire ne sont collectés.

Une information MANQUANTE est acceptée quand le modèle sait la traiter (âge, diplôme, ancienneté :
imputées comme à l'entraînement). Une information INVALIDE (âge négatif, code ROME mal formé,
diplôme inconnu...) est refusée avec une erreur 422 qui précise le champ en cause.
"""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

Diplome = Literal["Sans diplôme", "Bac", "Bac+2", "Bac+5"]


class Usager(BaseModel):
    model_config = ConfigDict(extra="forbid")   # un champ inconnu (ex. « nationalite ») est refusé

    age: float | None = Field(None, ge=16, le=70, description="Âge en années (vide si inconnu)")
    niveau_diplome: Diplome | None = Field(None, description="Plus haut diplôme (vide si inconnu)")
    anciennete_poste_ans: float | None = Field(None, ge=0, le=50,
                                               description="Ancienneté dans le dernier poste, en années")
    code_rome_vise: str = Field(pattern=r"^[A-Za-z]\d{4}$", description="Code ROME visé, ex. M1607")
    synthese_entretien: str = Field(min_length=1, max_length=5000,
                                    description="Synthèse rédigée par le conseiller")


class DemandePrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    usager: Usager
    id_session: str | None = Field(None, max_length=64, description="Identifiant de la session du conseiller")


class Prediction(BaseModel):
    id_prediction: str
    classe: int
    recommandation: str
    niveau_alerte: str
    risque_longue_duree: float = Field(description="Probabilité recalibrée de la classe 2 (entre 0 et 1)")
    risque_affiche: str = Field(description="Risque arrondi, tel qu'affiché au conseiller")
    version_modele: str


class Feedback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id_prediction: str = Field(min_length=1, max_length=64)
    classe_reelle: Literal[0, 1, 2] = Field(description="Classe constatée par le conseiller")
    commentaire: str | None = Field(None, max_length=1000)


def vers_tableau(usager: Usager) -> pd.DataFrame:
    """Une ligne au format du fichier d'origine, prête pour predire_usagers().

    La commune n'est pas collectée : la colonne est laissée vide (le modèle retenu ne l'utilise pas,
    seul le nettoyage la lit pour calculer une région qui n'entre pas dans le modèle).
    """
    ligne = {**usager.model_dump(), "code_insee_commune": None}
    return pd.DataFrame([ligne]).astype({"age": float, "anciennete_poste_ans": float})
