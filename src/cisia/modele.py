"""Construction du modèle retenu (notebook, sections 4.4, 4.5, 5.3, 5.4 et 5.5)."""

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from cisia.preparation import COL_TEXTE

# Candidat B : sans nationalité, région ni statut d'allocataire (section 5.3)
VARIABLES_CANDIDAT_B = dict(
    colonnes_num=["age", "anciennete_poste_ans", "niveau_diplome_ord"],
    colonnes_bin=[], colonnes_cat=["domaine_rome"], avec_texte=True,
)

# Réglages retenus après la recherche sur grille en deux tours (section 5.4)
REGLAGES_RETENUS = {"modele__num_leaves": 7, "modele__learning_rate": 0.1, "modele__n_estimators": 100}


def construire_preprocesseur(colonnes_num, colonnes_bin, colonnes_cat, avec_texte):
    """Préprocesseur de la section 4.4, identique au notebook."""
    blocs = []
    if colonnes_num:
        blocs.append(("num", SimpleImputer(strategy="median"), colonnes_num))
    if colonnes_bin:
        blocs.append(("bin", SimpleImputer(strategy="most_frequent"), colonnes_bin))
    if colonnes_cat:
        blocs.append(("cat", OneHotEncoder(handle_unknown="ignore"), colonnes_cat))
    if avec_texte:
        blocs.append(("texte", TfidfVectorizer(ngram_range=(1, 2)), COL_TEXTE))
    return ColumnTransformer(blocs, sparse_threshold=0)


def construire_pipeline(reglages=REGLAGES_RETENUS):
    """Pipeline LightGBM pondéré du candidat B, avec TF-IDF sur les caractères (sections 5.4 et 5.5).

    Sans argument : le modèle final du notebook. `reglages=None` : réglages par défaut de LightGBM,
    point de départ de la recherche sur grille.
    """
    pipeline = Pipeline([
        ("preparation", construire_preprocesseur(**VARIABLES_CANDIDAT_B)),
        ("modele", LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1)),
    ])
    # TF-IDF sur des groupes de caractères, retenu pour sa robustesse
    pipeline.set_params(preparation__texte=TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                                           strip_accents="unicode"))
    if reglages:
        pipeline.set_params(**reglages)
    return pipeline
