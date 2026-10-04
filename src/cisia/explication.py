"""Explicabilité : pourquoi le modèle a-t-il donné cette estimation pour CET usager ?

Méthode : valeurs de SHAP exactes pour les arbres (TreeSHAP), calculées par LightGBM lui-même
(`pred_contrib=True`) : pas de dépendance en plus, ni dans l'image Docker.

Pour un usager et une classe, chaque variable reçoit une contribution au score brut du modèle
(en log-odds). Propriété d'additivité : valeur de base + somme des contributions = score brut.
Une contribution positive rapproche l'usager de la classe expliquée, une contribution négative l'en éloigne.

Lecture pour le conseiller :
- les contributions des variables sont regroupées par information saisie (âge, ancienneté, diplôme,
  métier visé, synthèse) : le TF-IDF compte des milliers de colonnes, illisibles une à une ;
- pour la synthèse, les contributions des groupes de caractères PRÉSENTS dans le texte sont réparties
  entre les mots qui les contiennent : on obtient les mots qui ont le plus pesé.

Limite : SHAP décrit le fonctionnement du modèle, pas une cause dans la vie de l'usager.
"""

import re

import numpy as np
import pandas as pd

SYNTHESE = "Synthèse de l'entretien"
GROUPES = {
    "num__age": "Âge",
    "num__anciennete_poste_ans": "Ancienneté dans le dernier poste",
    "num__niveau_diplome_ord": "Niveau de diplôme",
}
MANQUANTES = {"Âge": "age", "Ancienneté dans le dernier poste": "anciennete_poste_ans"}
NOMBRE_MOTS = 6


def groupe(nom_colonne):
    """Colonne après préparation → information saisie par le conseiller."""
    if nom_colonne.startswith("cat__"):
        return "Métier visé (domaine)"
    if nom_colonne.startswith("texte__"):
        return SYNTHESE
    return GROUPES.get(nom_colonne, nom_colonne)


def contributions(modele, X):
    """Contributions SHAP au score brut : tableau (usagers, classes, colonnes + 1) et noms des colonnes.

    La dernière valeur de chaque ligne est la valeur de base (score moyen du modèle).
    """
    preparation, classifieur = modele.named_steps["preparation"], modele.named_steps["modele"]
    Xt = preparation.transform(X)
    brut = np.asarray(classifieur.booster_.predict(Xt, pred_contrib=True))
    n_classes = len(classifieur.classes_)
    return Xt, brut.reshape(Xt.shape[0], n_classes, Xt.shape[1] + 1), preparation.get_feature_names_out()


def indices_par_groupe(noms):
    """Information saisie → indices des colonnes correspondantes après préparation."""
    indices = {}
    for j, nom in enumerate(noms):
        indices.setdefault(groupe(nom), []).append(j)
    return {g: np.array(js) for g, js in indices.items()}


def mots_de_la_synthese(texte, analyser, ngrammes, contribs):
    """Répartit la contribution de chaque groupe de caractères PRÉSENT entre les mots qui le contiennent.

    ngrammes et contribs : groupes de caractères présents dans le texte et leur contribution.
    """
    # Même découpage que le TF-IDF « char_wb » : mots séparés par des espaces, ponctuation comprise
    jetons = (texte or "").split()
    if not jetons:
        return []
    ngrammes_par_mot = [set(analyser(jeton)) for jeton in jetons]
    mots = [re.sub(r"[^\w'-]", "", jeton).strip("'-") for jeton in jetons]   # mot affiché, sans ponctuation
    effet = np.zeros(len(mots))
    for nom, contribution in zip(ngrammes, contribs):
        porteurs = [i for i, ensemble in enumerate(ngrammes_par_mot) if nom in ensemble]
        for i in porteurs:
            effet[i] += contribution / len(porteurs)
    cumul = {}
    for mot, valeur in zip(mots, effet):
        if len(mot) >= 3:                         # mots outils (« de », « à ») non affichés
            cumul[mot.lower()] = cumul.get(mot.lower(), 0.0) + valeur
    tries = sorted(cumul.items(), key=lambda paire: -abs(paire[1]))
    return [{"mot": mot, "contribution": round(float(v), 4)} for mot, v in tries[:NOMBRE_MOTS] if v != 0]


def libelle_facteur(libelle, usager):
    """Signale une information manquante : le modèle a alors utilisé la valeur médiane (ou « inconnu »)."""
    colonne = MANQUANTES.get(libelle)
    manquant = (colonne is not None and pd.isna(usager[colonne])) or \
        (libelle == "Niveau de diplôme" and usager.get("niveau_diplome_ord", 0) == -1)
    accord = "e" if libelle.startswith("Ancienneté") else ""
    return f"{libelle} (non renseigné{accord})" if manquant else libelle


def expliquer(modele, X, classes, taille_lot=200):
    """Explication par lots (la matrice des contributions grandit avec le vocabulaire du TF-IDF)."""
    classes = np.asarray(classes)
    return [explication for debut in range(0, len(X), taille_lot)
            for explication in expliquer_lot(modele, X.iloc[debut:debut + taille_lot],
                                             classes[debut:debut + taille_lot])]


def expliquer_lot(modele, X, classes):
    """Explication locale de chaque usager, pour la classe retenue par la règle de décision.

    X : données NETTOYÉES (sortie de nettoyer()) ; classes : classe retenue pour chaque usager.
    Renvoie une liste de dictionnaires (sérialisables en JSON) :
    {classe_expliquee, valeur_de_base, facteurs: [{facteur, contribution}], mots: [{mot, contribution}]}.
    """
    Xt, contribs, noms = contributions(modele, X)
    groupes = indices_par_groupe(noms)
    vectoriseur = modele.named_steps["preparation"].named_transformers_.get("texte")
    analyser = vectoriseur.build_analyzer() if vectoriseur is not None else None
    indices_texte = groupes.get(SYNTHESE, np.array([], dtype=int))
    ngrammes = np.array([nom[len("texte__"):] for nom in np.asarray(noms)[indices_texte]], dtype=object)

    explications = []
    for i, classe in enumerate(np.asarray(classes)):
        ligne = contribs[i, int(classe)]
        usager = X.iloc[i]
        facteurs = {libelle_facteur(g, usager): float(ligne[js].sum()) for g, js in groupes.items()}
        mots = []
        if analyser is not None and len(indices_texte):
            presents = np.nonzero(Xt[i, indices_texte])[0]     # absents : restent dans le total « synthèse »
            mots = mots_de_la_synthese(usager.get("synthese_entretien"), analyser, ngrammes[presents],
                                       ligne[indices_texte][presents])
        explications.append({
            "classe_expliquee": int(classe),
            "valeur_de_base": round(float(ligne[-1]), 4),
            "facteurs": [{"facteur": f, "contribution": round(v, 4)}
                         for f, v in sorted(facteurs.items(), key=lambda paire: -abs(paire[1]))],
            "mots": mots,
        })
    return explications


def importance_globale(modele, X, classe=2):
    """Importance moyenne (moyenne des |contributions|) de chaque information, pour une classe.

    Sert au notebook et à l'analyse éthique : sur quoi le modèle s'appuie-t-il en général ?
    """
    _, contribs, noms = contributions(modele, X)
    importance = {g: float(np.abs(contribs[:, classe, js].sum(axis=1)).mean())
                  for g, js in indices_par_groupe(noms).items()}
    return dict(sorted(importance.items(), key=lambda paire: -paire[1]))
