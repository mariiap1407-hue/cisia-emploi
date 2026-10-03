"""Entraînement complet, prédiction et évaluation, comme dans le notebook (sections 5.5, 5.6, 6.3 et 6.6).

Ces fonctions ne dépendent pas de MLflow : le script d'entraînement les appelle et trace les résultats,
le test de cohérence les appelle directement.
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict

from cisia.decision import ajuster_recalibrage, appliquer_recalibrage, appliquer_regle
from cisia.evaluation import CV, mesurer
from cisia.modele import REGLAGES_RETENUS, construire_pipeline


def entrainer(X_train, y_train, regle, reglages=REGLAGES_RETENUS):
    """Renvoie le modèle entraîné sur tout le jeu d'entraînement et la correction du risque de la classe 2.

    1. probabilités hors pli sur les 5 plis (section 5.6) ;
    2. correction apprise sur la probabilité hors pli de la classe 2 (section 6.3) ;
    3. modèle final entraîné sur tout le jeu d'entraînement (section 5.5).
    """
    modele = construire_pipeline(reglages)
    probas_hors_pli = cross_val_predict(clone(modele), X_train, y_train, cv=CV, method="predict_proba")
    cible = (np.asarray(y_train) == 2).astype(int)
    correction = ajuster_recalibrage(regle["methode_recalibration"], probas_hors_pli[:, 2], cible)
    modele.fit(X_train, y_train)
    return modele, correction


def predire(modele, correction, regle, X):
    """Probabilités brutes, risque recalibré de la classe 2 et classe retenue par la règle."""
    probas = modele.predict_proba(X)
    risque = appliquer_recalibrage(regle["methode_recalibration"], correction, probas[:, 2])
    classes = appliquer_regle(probas, risque, regle["seuil_classe_2"], regle["seuil_garde_fou_classe_0"])
    return probas, risque, classes


def evaluer(y_vrai, probas, risque, classes):
    """Indicateurs du jeu de test (section 6.6), pour la règle retenue."""
    y_vrai = np.asarray(y_vrai)
    return {
        "Accuracy": accuracy_score(y_vrai, classes),
        **mesurer(y_vrai, classes),
        "AUC ROC (3 classes)": roc_auc_score(y_vrai, probas, multi_class="ovr"),
        "AUC ROC classe 2": roc_auc_score(y_vrai == 2, probas[:, 2]),
        "Brier classe 2 (brut)": brier_score_loss(y_vrai == 2, probas[:, 2]),
        "Brier classe 2 (recalibré)": brier_score_loss(y_vrai == 2, risque),
    }
