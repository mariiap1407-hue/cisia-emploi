"""Prédiction pour de nouveaux usagers, à partir de leurs données brutes.

C'est la séquence complète, la même que dans le notebook (section 6.6) :
données brutes -> nettoyer() -> probabilités du modèle -> recalibration du risque de la classe 2
-> règle de décision -> recommandation et niveau d'alerte.
Elle est utilisée par le modèle enregistré dans MLflow et par l'API.
"""

import pandas as pd

from cisia.decision import RECOMMANDATIONS, niveau_alerte
from cisia.entrainement import predire
from cisia.preparation import TYPES_LECTURE, nettoyer


def predire_usagers(composants, donnees_brutes):
    """composants : dictionnaire renvoyé par artefacts.charger() (modèle, correction, règle).

    Renvoie une ligne par usager : classe, recommandation, niveau d'alerte et risque de longue durée.
    """
    types = {colonne: type_ for colonne, type_ in TYPES_LECTURE.items() if colonne in donnees_brutes}
    X = nettoyer(pd.DataFrame(donnees_brutes).astype(types))
    regle = composants["regle"]
    _, risque, classes = predire(composants["modele"], composants["correction"], regle, X)
    return pd.DataFrame({
        "classe": classes,
        "recommandation": [RECOMMANDATIONS[c] for c in classes],
        "niveau_alerte": [niveau_alerte(classe, r, regle["seuil_garde_fou_classe_0"])
                          for classe, r in zip(classes, risque)],
        "risque_longue_duree": risque,
    }, index=X.index)
