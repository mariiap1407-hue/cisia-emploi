"""Génère des jeux de données FACTICES, au même format que le vrai fichier.

Ils servent aux tests et à la CI (GitHub Actions), où le vrai fichier ne doit jamais aller :
il contient des données personnelles. Aucune ligne n'est issue des vraies données.

Trois fichiers sont créés dans data/factice/ :
- entrainement.csv : pour entraîner un modèle de contrôle ;
- reference.csv : même lien entre variables et classe, pour vérifier le quality gate ;
- reference_derivee.csv : le lien entre le profil, la synthèse et la classe a changé (dérive),
  pour vérifier que le quality gate bloque bien un modèle devenu mauvais.

Usage : python scripts/generer_donnees_factices.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DOSSIER = Path("data/factice")
GRAINE = 42

# Phrases inventées, typiques de chaque classe (0 : rapide, 1 : moyen, 2 : longue durée)
PHRASES = {
    0: ["Profil recherché dans son secteur, disponible immédiatement.",
        "Expérience récente et réseau professionnel actif.",
        "Plusieurs entretiens déjà programmés, projet très clair."],
    1: ["Projet professionnel à préciser avec le conseiller.",
        "Quelques compétences à remettre à niveau, motivation présente.",
        "Recherche en cours, mobilité limitée au département."],
    2: ["Éloignement durable de l'emploi et isolement social.",
        "Difficultés de mobilité et de garde d'enfants cumulées.",
        "Maîtrise insuffisante de l'écrit, besoin d'accompagnement global."],
}
DIPLOMES = ["Sans diplôme", "Bac", "Bac+2", "Bac+5"]
DOMAINES_ROME = list("ACDGHKMN")
DEPARTEMENTS = ["01", "13", "21", "2A", "33", "35", "57", "59", "62", "69", "75", "93", "971", "974"]


def generer(n, rng, derive=False):
    classe = rng.choice([0, 1, 2], size=n, p=[0.37, 0.45, 0.18])

    # Variables tabulaires liées à la classe (lien volontairement simple).
    # Dérive : le lien entre le profil et la classe s'inverse pour les classes 0 et 2
    profil = 2 - classe if derive else classe
    age = np.clip(rng.normal(36 + 6 * profil, 11), 18, 63).round()
    anciennete = np.clip(rng.exponential(3.5 - profil, n), 0, 20).round(1)
    indice_diplome = np.clip(rng.normal(2.2 - 0.8 * profil, 0.9), 0, 3).round().astype(int)
    diplome = np.array(DIPLOMES, dtype=object)[indice_diplome]

    # Synthèse : 80 % du temps, une phrase typique de la classe ; sinon, une phrase au hasard
    classe_du_texte = np.where(rng.random(n) < 0.8, classe, rng.choice([0, 1, 2], size=n))
    if derive:
        # Dérive : les phrases des classes 0 et 2 ont changé de sens
        classe_du_texte = np.select([classe_du_texte == 0, classe_du_texte == 2], [2, 0], classe_du_texte)
    synthese = [rng.choice(PHRASES[c]) for c in classe_du_texte]

    departement = rng.choice(DEPARTEMENTS, size=n)
    # Code commune à 5 caractères : département (2 ou 3 caractères) + numéro de commune
    commune = [d + str(rng.integers(1, 100)).zfill(5 - len(d)) for d in departement]

    df = pd.DataFrame({
        "usager_id": [f"FACTICE_{i:05d}" for i in range(n)],
        "age": age,
        "niveau_diplome": diplome,
        "anciennete_poste_ans": anciennete,
        "code_rome_vise": [f"{rng.choice(DOMAINES_ROME)}{rng.integers(1100, 1900)}" for _ in range(n)],
        "code_insee_commune": commune,
        "est_allocataire": rng.integers(0, 2, n).astype(float),
        "nationalite_hors_ue": (rng.random(n) < 0.12).astype(int),
        "synthese_entretien": synthese,
        "classe_retour_emploi": classe,
    })

    # Valeurs manquantes, dans les mêmes proportions que le vrai fichier
    df.loc[rng.random(n) < 0.05, "age"] = np.nan
    df.loc[rng.random(n) < 0.03, "niveau_diplome"] = np.nan
    df.loc[rng.random(n) < 0.02, "est_allocataire"] = np.nan
    df.loc[rng.random(n) < 0.03, "synthese_entretien"] = np.nan
    return df


def main():
    rng = np.random.default_rng(GRAINE)
    DOSSIER.mkdir(parents=True, exist_ok=True)
    jeux = {
        "entrainement.csv": generer(1000, rng),
        "reference.csv": generer(300, rng),
        "reference_derivee.csv": generer(300, rng, derive=True),
    }
    for nom, df in jeux.items():
        df.to_csv(DOSSIER / nom, index=False)
        print(f"{DOSSIER / nom} : {len(df)} lignes")


if __name__ == "__main__":
    main()
