"""Validation croisée et indicateurs (notebook, sections 5.1 et 6.1)."""

import numpy as np
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

# Les mêmes 5 plis que dans le notebook
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def taux_erreurs_critiques(y_vrai, y_pred):
    """Part des usagers réellement en classe 2 prédits en classe 0.

    Sans aucun usager de classe 2, le taux n'est pas défini : la fonction renvoie NaN (sans avertissement),
    et le quality gate le traite comme un échec.
    """
    y_pred = np.ravel(y_pred)
    classe_2 = np.asarray(y_vrai) == 2   # usagers réellement en classe 2
    if classe_2.sum() == 0:
        return np.nan
    return ((y_pred == 0) & classe_2).sum() / classe_2.sum()


# Métriques de la recherche sur grille (section 5.1).
# Les erreurs critiques sont un taux à minimiser : greater_is_better=False, donc scikit-learn renvoie
# le taux en NÉGATIF (convention « plus grand = meilleur »). Le choix se fait sur le F1 macro.
METRIQUES = {
    "F1 macro": "f1_macro",
    "Accuracy": "accuracy",
    "Rappel classe 2": make_scorer(recall_score, labels=[2], average="macro"),
    "AUC ROC": "roc_auc_ovr",
    "Erreurs critiques (2→0)": make_scorer(taux_erreurs_critiques, greater_is_better=False),
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


# Quality gate : seuils d'alerte proposés pour le suivi en production (section 6.6)
SEUILS_QUALITE = {
    "Erreurs critiques (2→0)": ("max", 0.10),
    "Rappel classe 2": ("min", 0.60),
    "F1 macro": ("min", 0.62),
}


def verifier_seuils_qualite(resultats):
    """Renvoie la liste des seuils non respectés (liste vide : le modèle passe le quality gate)."""
    echecs = []
    for indicateur, (sens, seuil) in SEUILS_QUALITE.items():
        valeur = resultats[indicateur]
        if valeur is None or np.isnan(valeur):
            echecs.append(f"{indicateur} non calculable (aucun usager concerné)")
        elif (sens == "max" and valeur > seuil) or (sens == "min" and valeur < seuil):
            echecs.append(f"{indicateur} = {valeur:.3f} (seuil {sens} : {seuil})")
    return echecs


# Champion / challenger : le candidat est comparé au modèle EN PRODUCTION, sur le même jeu de test. Chacun
# applique SA règle de décision (celle sauvegardée avec le champion, celle fournie au candidat) : si elles
# diffèrent, on compare deux systèmes de décision complets, et c'est tracé (meme_regle, scripts/entrainer.py).
# Critères par ordre de priorité (comparaison lexicographique) : une erreur critique (chômeur de longue
# durée orienté en accompagnement léger) coûte bien plus cher qu'une fausse alerte ; on ne regarde le
# critère suivant qu'en cas d'égalité sur le précédent. Conséquence voulue : une erreur critique évitée
# peut justifier une baisse du rappel ou du F1, tant que les seuils minimaux (quality gate) sont respectés.
CRITERES_CHAMPION = [("Nb erreurs critiques", "min"), ("Rappel classe 2", "max"), ("F1 macro", "max")]


def comparer_au_modele_en_production(candidat, production):
    """Renvoie {meilleur, motif, criteres} ; à égalité sur les trois critères, le modèle en place reste."""
    criteres = [{"critere": nom, "sens": sens, "candidat": round(float(candidat[nom]), 4),
                 "production": round(float(production[nom]), 4)} for nom, sens in CRITERES_CHAMPION]
    for ligne in criteres:
        c, p = ligne["candidat"], ligne["production"]
        if c == p:
            continue
        meilleur = c < p if ligne["sens"] == "min" else c > p
        verdict = "Candidat meilleur que le modèle en production" if meilleur else \
            "Candidat moins bon que le modèle en production"
        return {"meilleur": meilleur, "criteres": criteres,
                "motif": f"{verdict} — {ligne['critere']} : {c:g} contre {p:g}."}
    return {"meilleur": False, "criteres": criteres,
            "motif": "Candidat équivalent au modèle en production sur les trois critères : "
                     "le modèle en place est conservé."}
