"""Format des requêtes et des réponses de l'API, avec la validation des entrées (Pydantic).

Minimisation des données (RGPD) : l'API ne demande que les informations utilisées par le modèle
retenu (candidat B) : âge, diplôme, ancienneté, métier visé et synthèse de l'entretien.
Ni la nationalité, ni la commune, ni le statut d'allocataire ne sont collectés : un champ inconnu
fait refuser la requête, et une requête refusée n'est jamais conservée avec ses valeurs.

Information MANQUANTE ou INVALIDE :
- manquante et acceptée : âge et ancienneté (remplacés par la médiane, comme à l'entraînement),
  diplôme (codé -1, « non renseigné », comme à l'entraînement) ;
- invalide et refusée (erreur 422 qui précise le champ) : âge négatif, code ROME mal formé,
  diplôme inconnu, synthèse vide...

Politiques d'entrée (choix du projet, pas des propriétés des données) : âge accepté de 16 à 70 ans
(les données d'entraînement vont de 18 à 63 ans) ; synthèse obligatoire, alors que le modèle
saurait traiter une synthèse absente : une recommandation sans entretien n'a pas de sens métier.
"""

from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Diplome = Literal["Sans diplôme", "Bac", "Bac+2", "Bac+5"]
# Espaces retirés avant la vérification : une synthèse faite seulement d'espaces est refusée
Synthese = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=5000)]
CodeRome = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z]\d{4}$")]


class Usager(BaseModel):
    model_config = ConfigDict(extra="forbid")   # un champ inconnu (ex. « nationalite ») est refusé

    age: float | None = Field(None, ge=16, le=70, description="Âge en années (vide si inconnu)")
    niveau_diplome: Diplome | None = Field(None, description="Plus haut diplôme (vide si inconnu)")
    anciennete_poste_ans: float | None = Field(None, ge=0, le=50,
                                               description="Ancienneté dans le dernier poste, en années")
    code_rome_vise: CodeRome = Field(description="Code ROME visé, ex. M1607")
    synthese_entretien: Synthese = Field(description="Synthèse rédigée par le conseiller")


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
    """Retour du conseiller sur une prédiction, une fois la situation de l'usager CONNUE.

    classe_reelle est la classe de retour à l'emploi OBSERVÉE (0 : moins de 6 mois, 1 : 6 à 12 mois,
    2 : plus de 12 mois), et non l'accompagnement recommandé ou choisi. Un nouveau feedback sur la
    même prédiction remplace le précédent (le plus récent fait foi) ; les anciens restent tracés.
    """
    model_config = ConfigDict(extra="forbid")

    id_prediction: str = Field(min_length=1, max_length=64)
    classe_reelle: Literal[0, 1, 2] = Field(description="Classe de retour à l'emploi observée")
    commentaire: str | None = Field(None, max_length=1000)


class ElementHistorique(BaseModel):
    """Ce que l'historique montre : pas de détail technique interne, pas de valeur d'une requête refusée."""
    id_prediction: str
    date: str
    id_session: str | None
    statut: str
    entrees: dict | None
    sorties: dict | None
    version_modele: str | None
    duree_ms: float | None
    erreurs_validation: list | None


def vers_tableau(usager: Usager) -> pd.DataFrame:
    """Une ligne au format du fichier d'origine, prête pour predire_usagers().

    La commune n'est pas collectée : la colonne est laissée vide (le modèle retenu ne l'utilise pas,
    seul le nettoyage la lit pour calculer une région qui n'entre pas dans le modèle).
    """
    ligne = {**usager.model_dump(), "code_insee_commune": None}
    return pd.DataFrame([ligne]).astype({"age": float, "anciennete_poste_ans": float})


def erreurs_sans_valeurs(erreurs):
    """Erreurs de validation SANS les valeurs reçues : champ, type d'erreur et message seulement.

    Les valeurs refusées ne sont ni renvoyées ni conservées : elles peuvent être des données non
    demandées (nationalité...) ou des nombres non représentables en JSON (1e999 = infini).
    """
    return [{"champ": [str(partie) for partie in erreur.get("loc", ())],
             "type": str(erreur.get("type", "")),
             "message": str(erreur.get("msg", ""))} for erreur in erreurs]
