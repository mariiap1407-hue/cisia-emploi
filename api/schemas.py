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

Erreurs : messages en français pour les cas courants (repli sur le message anglais de Pydantic),
chemin du champ sans le préfixe technique « body » (ex. ["usager", "age"]).
"""

from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Diplome = Literal["Sans diplôme", "Bac", "Bac+2", "Bac+5"]
# Espaces retirés avant la vérification : une synthèse faite seulement d'espaces est refusée
Synthese = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=5000)]
CodeRome = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z]\d{4}$")]
IdUsager = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z0-9_-]{1,32}$")]
MotifCorrection = Literal["Informations complémentaires issues de l'entretien",
                          "Situation personnelle non prise en compte", "Projet professionnel en évolution",
                          "Autre"]


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
    id_usager: IdUsager | None = Field(None, description="Identifiant PSEUDONYME du dossier (ex. DE-0248), "
                                                         "jamais un nom : lettres, chiffres, tiret")
    id_session: str | None = Field(None, max_length=64, description="Identifiant de la session du conseiller")


class Facteur(BaseModel):
    facteur: str = Field(description="Information saisie (âge, diplôme, ancienneté, métier, synthèse)")
    contribution: float = Field(description="Contribution SHAP au score brut de la classe retenue (logit "
                                            "multiclasse) : positive = augmente ce score, "
                                            "négative = le diminue")


class Mot(BaseModel):
    """Repère indicatif : contributions des fragments de caractères réparties entre les mots (approximé)."""
    mot: str
    contribution: float


class DecisionRegle(BaseModel):
    """Ce que SHAP n'explique pas : la règle de décision appliquée au risque recalibré."""
    motif: Literal["seuil_classe_2", "garde_fou", "plus_probable_0_1", "plus_probable"]
    classe_la_plus_probable: int = Field(description="Classe la plus probable du modèle (avant la règle)")
    risque_recalibre: float
    seuil_classe_2: float | None
    seuil_garde_fou: float | None


class Explication(BaseModel):
    """Pourquoi cette classe ? Deux parties distinctes.

    1. facteurs : contributions SHAP exactes (TreeSHAP, calculées par LightGBM) au score brut de la classe
       retenue ; valeur de base + somme des contributions = score brut. N'explique pas la règle de décision.
    2. regle : comment la règle (seuil de la classe 2, garde-fou) a choisi la classe à partir du risque.
    Décrit le fonctionnement du modèle, pas une cause dans la situation de l'usager.
    """
    classe_expliquee: int
    valeur_de_base: float
    facteurs: list[Facteur]
    mots: list[Mot] = Field(description="Repères indicatifs dans la synthèse (pas un SHAP par mot)")
    regle: DecisionRegle | None = None


class Prediction(BaseModel):
    id_prediction: str
    classe: int
    recommandation: str
    niveau_alerte: str
    risque_longue_duree: float = Field(description="Probabilité recalibrée de la classe 2 (entre 0 et 1)")
    risque_affiche: str = Field(description="Risque arrondi, tel qu'affiché au conseiller")
    version_modele: str
    explication: Explication | None = Field(
        None, description="Explication de la classe retenue (absente pour un modèle d'avant l'ajout de SHAP)")


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


class AvisConseiller(BaseModel):
    """Appréciation du conseiller AU MOMENT de l'entretien (« confirmer » ou « proposer une correction »).

    Distincte du feedback : l'avis sert au suivi de l'accord entre conseillers et modèle, il n'est
    JAMAIS utilisé pour réentraîner (le modèle apprendrait l'opinion, pas la réalité). Le
    réentraînement n'utilise que la situation OBSERVÉE (route /feedback), connue des mois plus tard.
    Un nouvel avis sur la même prédiction remplace le précédent ; les anciens restent tracés.
    """
    model_config = ConfigDict(extra="forbid")

    id_prediction: str = Field(min_length=1, max_length=64)
    avis: Literal["confirme", "corrige"]
    classe_proposee: Literal[0, 1, 2] | None = Field(None, description="Obligatoire pour une correction")
    motif: MotifCorrection | None = Field(None, description="Obligatoire pour une correction")
    precisions: Annotated[str, StringConstraints(strip_whitespace=True, max_length=1000)] | None = None

    @model_validator(mode="after")
    def correction_complete(self):
        if self.avis == "corrige" and (self.classe_proposee is None or self.motif is None):
            raise ValueError("Une correction exige la classe proposée et le motif.")
        return self


class ElementHistorique(BaseModel):
    """Ce que l'historique montre : pas de détail technique interne, pas de valeur d'une requête refusée."""
    id_prediction: str
    date: str
    id_usager: str | None = None
    id_session: str | None
    statut: str
    entrees: dict | None
    sorties: dict | None
    version_modele: str | None
    duree_ms: float | None
    erreurs_validation: list | None
    avis_conseiller: dict | None = Field(None, description="Dernier avis du conseiller à l'entretien")
    situation_observee: dict | None = Field(None, description="Dernière situation observée (feedback)")


def vers_tableau(usager: Usager) -> pd.DataFrame:
    """Une ligne au format du fichier d'origine, prête pour predire_usagers().

    La commune n'est pas collectée : la colonne est laissée vide (le modèle retenu ne l'utilise pas,
    seul le nettoyage la lit pour calculer une région qui n'entre pas dans le modèle).
    """
    ligne = {**usager.model_dump(), "code_insee_commune": None}
    return pd.DataFrame([ligne]).astype({"age": float, "anciennete_poste_ans": float})


class Reentrainement(BaseModel):
    """Demande de réentraînement. Par défaut, le nouveau modèle est mis en service s'il passe le gate."""
    model_config = ConfigDict(extra="forbid")

    promouvoir: bool = Field(True, description="Mettre en service le nouveau modèle s'il passe le gate")


class ResultatReentrainement(BaseModel):
    statut: Literal["mis_en_production", "candidat_non_promu", "refuse_quality_gate",
                    "non_promu_pas_meilleur"]
    n_feedbacks: int = Field(description="Feedbacks ajoutés aux données d'entraînement")
    n_feedbacks_ecartes: int = Field(0, description="Feedbacks écartés (profil déjà dans le test)")
    run_id: str = Field(description="Run MLflow du nouvel entraînement")
    quality_gate_ok: bool
    echecs_quality_gate: list[str]
    resultats_test: dict = Field(description="Indicateurs sur le jeu de test (inchangé : comparable)")
    comparaison_production: dict | None = Field(
        None, description="Champion / challenger : candidat comparé au modèle en production (même jeu de "
                          "test ; meme_regle indique si les deux règles de décision sont identiques) ; "
                          "promu seulement s'il est meilleur")
    version_modele_avant: str | None
    version_modele: str | None = Field(description="Version en service APRÈS la demande")
    duree_s: float


class Sante(BaseModel):
    statut: Literal["ok", "indisponible"]
    modele_charge: bool
    cle_configuree: bool
    version_modele: str | None
    run_id: str | None


# Format des erreurs, pour la documentation (/docs) : c'est le format réellement renvoyé
class ErreurChamp(BaseModel):
    champ: list[str] = Field(description="Chemin du champ en cause, ex. [\"usager\", \"age\"]")
    type: str = Field(description="Type d'erreur (code Pydantic)")
    message: str


class ErreurValidation(BaseModel):
    detail: list[ErreurChamp]


class Erreur(BaseModel):
    detail: str


# Messages en français pour les erreurs courantes ; {ge}, {le}... sont les LIMITES du schéma
# (jamais la valeur reçue). Type absent de la liste : message anglais de Pydantic.
MESSAGES = {
    "missing": "Champ obligatoire manquant.",
    "extra_forbidden": "Champ non autorisé : cette information n'est pas collectée.",
    "greater_than_equal": "Doit être supérieur ou égal à {ge}.",
    "less_than_equal": "Doit être inférieur ou égal à {le}.",
    "greater_than": "Doit être strictement supérieur à {gt}.",
    "less_than": "Doit être strictement inférieur à {lt}.",
    "literal_error": "Valeur non autorisée. Valeurs possibles : {expected}.",
    "string_pattern_mismatch": "Format invalide.",
    "string_too_short": "Texte vide ou trop court (au moins {min_length} caractère(s)).",
    "string_too_long": "Texte trop long (au plus {max_length} caractères).",
    "string_type": "Texte attendu.",
    "float_parsing": "Nombre attendu.",
    "float_type": "Nombre attendu.",
    "int_parsing": "Nombre entier attendu.",
    "int_type": "Nombre entier attendu.",
    "int_from_float": "Nombre entier attendu.",
    "finite_number": "Nombre fini attendu.",
    "bool_parsing": "Valeur attendue : true ou false.",
    "bool_type": "Valeur attendue : true ou false.",
    "json_invalid": "JSON mal formé.",
    "value_error": "{error}",
    "model_attributes_type": "Objet JSON attendu.",
    "dict_type": "Objet JSON attendu.",
}
MESSAGE_ID_USAGER = ("Identifiant pseudonyme attendu (ex. DE-0248) : lettres, chiffres, tiret, "
                     "32 caractères au plus, jamais de nom.")
MESSAGE_CODE_ROME = "Format invalide : une lettre suivie de 4 chiffres attendue (ex. M1607)."
PREFIXES_TECHNIQUES = {"body", "query", "header", "path"}


def lisible(valeur):
    """16.0 devient « 16 » ; « 'Bac' or 'Bac+2' » devient « 'Bac' ou 'Bac+2' »."""
    if isinstance(valeur, float) and valeur.is_integer():
        valeur = int(valeur)
    return str(valeur).replace(" or ", " ou ")


def message_francais(erreur, champ):
    type_erreur = erreur.get("type", "")
    if type_erreur == "string_pattern_mismatch" and champ and champ[-1] == "code_rome_vise":
        return MESSAGE_CODE_ROME
    if type_erreur == "string_pattern_mismatch" and champ and champ[-1] == "id_usager":
        return MESSAGE_ID_USAGER
    modele = MESSAGES.get(type_erreur)
    if modele is None:
        return str(erreur.get("msg", ""))
    limites = {cle: lisible(valeur) for cle, valeur in (erreur.get("ctx") or {}).items()}
    try:
        return modele.format(**limites)
    except (KeyError, IndexError):
        return str(erreur.get("msg", ""))


def erreurs_sans_valeurs(erreurs):
    """Erreurs de validation SANS les valeurs reçues : champ, type d'erreur et message seulement.

    Les valeurs refusées ne sont ni renvoyées ni conservées : elles peuvent être des données non
    demandées (nationalité...) ou des nombres non représentables en JSON (1e999 = infini).
    """
    resultat = []
    for erreur in erreurs:
        champ = [str(partie) for partie in erreur.get("loc", ())]
        if len(champ) > 1 and champ[0] in PREFIXES_TECHNIQUES:
            champ = champ[1:]
        resultat.append({"champ": champ, "type": str(erreur.get("type", "")),
                         "message": message_francais(erreur, champ)})
    return resultat
