"""Validation croisée et indicateurs (notebook, sections 5.1 et 6.1)."""

import numpy as np
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

# Les mêmes 5 plis que dans le notebook
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def taux_erreurs_critiques(y_vrai, y_pred):
    """Part des usagers réellement en classe 2 prédits en classe 0."""
    y_pred = np.ravel(y_pred)
    classe_2 = np.asarray(y_vrai) == 2   # usagers réellement en classe 2
    return ((y_pred == 0) & classe_2).sum() / classe_2.sum()


# Métriques de la recherche sur grille (section 5.1)
METRIQUES = {
    "F1 macro": "f1_macro",
    "Accuracy": "accuracy",
    "Rappel classe 2": make_scorer(recall_score, labels=[2], average="macro"),
    "AUC ROC": "roc_auc_ovr",
    "Erreurs critiques (2→0)": make_scorer(taux_erreurs_critiques),
}


def mesurer(y_vrai, y_pred):
    """Indicateurs de l'étape 6 : performance globale, détection de la classe 2 et coût des erreurs."""
    y_vrai, y_pred = np.asarray(y_vrai), np.asarray(y_pred)
    return {
        "F1 macro": f1_score(y_vrai, y_pred, average="macro"),
        "Rappel classe 2": recall_score(y_vrai, y_pred, labels=[2], average="macro"),
        "Précision classe 2": precision_score(y_vrai, y_pred, labels=[2], average="macro", zero_division=0),
        "Erreurs critiques (2→0)": taux_erreurs_critiques(y_vrai, y_pred),
        "Nb erreurs critiques": int(((y_pred == 0) & (y_vrai == 2)).sum()),
        "Nb fausses alertes classe 2": int(((y_pred == 2) & (y_vrai != 2)).sum()),
        "Rappel classe 0": recall_score(y_vrai, y_pred, labels=[0], average="macro"),
        "Précision classe 0": precision_score(y_vrai, y_pred, labels=[0], average="macro", zero_division=0),
        "Nb orientés en classe 1": int((y_pred == 1).sum()),
    }
