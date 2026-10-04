"""Journaux FACTICES pour démontrer le suivi (B10) : un service sain et un service dégradé.

Aucune donnée réelle : profils et synthèses inventés, identifiants FACTICE-xxxx. Les journaux sont écrits
avec la même classe Journal que l'API, donc au même format que le vrai journal.

Usage (depuis la racine du projet) :
    python scripts/generer_journal_factice.py              # les deux scénarios
    python scripts/suivi.py --base outputs/suivi_demo/journal_sain.db      → aucune alerte, code 0
    python scripts/suivi.py --base outputs/suivi_demo/journal_degrade.db   → alertes, code 2
    (ajouter --reference data/factice/entrainement.csv pour la dérive des données : PSI, KS)

Scénario « dégradé » : erreurs du service, latence élevée, beaucoup d'accompagnements renforcés,
conseillers souvent en désaccord, usagers de longue durée orientés en « retour rapide » (erreurs
critiques), synthèses longues avec négations, dérive localisée sur les métiers du transport (âge),
situations observées biaisées (classe 1 rarement recontactée), réentraînement en échec dont l'alias
n'a pas été remis.
"""

import argparse
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))   # le module api/ n'est pas installé : il est lu depuis la racine

from api.journal import Journal  # noqa: E402
from cisia.decision import RECOMMANDATIONS  # noqa: E402
from cisia.preparation import charger_donnees, decouper  # noqa: E402

COLONNES = ["age", "niveau_diplome", "anciennete_poste_ans", "code_rome_vise", "synthese_entretien"]

SYNTHESE_COURTE = "Projet clair, mobilité possible, expérience en accueil."
SYNTHESE_LONGUE = ("Projet : reconversion vers le secrétariat médical. Expériences : dix ans en accueil. "
                   "Freins : pas de permis, aucune solution de garde le mercredi, pas de mobilité.")


def profils_factices():
    """Profils d'usagers tirés des données FACTICES d'entraînement (s'ils existent) : le scénario sain
    ressemble alors aux données d'entraînement (pas de dérive), le scénario dégradé s'en écarte."""
    chemin = RACINE / "data" / "factice" / "entrainement.csv"
    if not chemin.exists():
        return None
    entrainement, _ = decouper(charger_donnees(chemin))
    return entrainement[COLONNES].to_dict("records")


def remplir(journal, scenario, maintenant, graine=0, profils=None, nombre=3200, jours=42):
    """Remplit `journal` avec `nombre` prédictions sur les `jours` derniers jours (6 semaines par défaut :
    assez d'usagers par semaine et par métier pour le suivi segmenté, comme le M6)."""
    alea = random.Random(graine)
    degrade = scenario == "degrade"
    poids_classes = [0.2, 0.2, 0.6] if degrade else [0.27, 0.52, 0.21]
    for i in range(nombre):
        id_prediction = f"FACTICE-{scenario}-{i:04d}"
        anciennete_jours = alea.uniform(0, jours)
        date = (maintenant - timedelta(days=anciennete_jours)).isoformat(timespec="milliseconds")
        if degrade and i % 10 == 0:              # 10 % de requêtes en erreur
            journal.enregistrer_inference(id_prediction, date, None, "erreur", message_erreur="factice")
            continue
        if i % 50 == 49:                         # quelques saisies refusées (normal)
            journal.enregistrer_inference(id_prediction, date, None, "invalide",
                                          erreurs_validation=[{"champ": ["age"], "type": "factice"}])
            continue
        classe = alea.choices([0, 1, 2], poids_classes)[0]
        garde_fou = classe == 1 and alea.random() < 0.15
        sorties = {"classe": classe, "recommandation": RECOMMANDATIONS[classe],
                   "explication": {"regle": {"motif": "garde_fou" if garde_fou else "plus_probable_0_1"}}}
        if profils:
            profil = alea.choice(profils)
            entrees = {c: (None if isinstance(profil[c], float) and profil[c] != profil[c] else profil[c])
                       for c in COLONNES}                                       # NaN du CSV → None
        else:
            entrees = {"age": alea.randint(18, 64), "niveau_diplome": "Bac", "anciennete_poste_ans": 2.0,
                       "code_rome_vise": "M1607", "synthese_entretien": SYNTHESE_COURTE}
        if degrade:                              # synthèses longues partout (dérive GLOBALE)
            entrees["synthese_entretien"] = SYNTHESE_LONGUE
            # Dérive LOCALISÉE : métiers du transport (N), 3 dernières semaines, population plus âgée.
            # Elle pèse peu dans la moyenne globale : seul le suivi par métier la révèle.
            if str(entrees.get("code_rome_vise") or "").upper().startswith("N") and anciennete_jours < 21:
                entrees["age"] = min(70, (entrees["age"] or 40) + 20)
        duree = alea.uniform(600, 1200) if degrade else alea.uniform(12, 30)
        journal.enregistrer_inference(id_prediction, date, None, "ok", entrees=entrees, sorties=sorties,
                                      version_modele="factice-1", duree_ms=round(duree, 2),
                                      id_usager=f"FACTICE-{i:04d}")

        # Avis du conseiller (environ une prédiction sur deux)
        if i % 2 == 0:
            confirme = alea.random() < (0.5 if degrade else 0.85)
            if confirme:
                journal.enregistrer_avis(id_prediction, date, "confirme", classe)
            else:
                journal.enregistrer_avis(id_prediction, date, "corrige", (classe + 1) % 3, "Autre")

        # Situation observée (plus tard), pour presque toutes les prédictions. Scénario dégradé : les
        # usagers orientés en classe 1 ne sont presque jamais recontactés (biais des étiquettes)
        if i % 6 != 5 and not (degrade and classe == 1 and alea.random() < 0.75):
            if degrade and classe == 0:
                reelle = 2                       # longue durée orientée « retour rapide » : erreur critique
            elif alea.random() < 0.85:
                reelle = classe
            else:
                reelle = 1                       # écart modéré (jamais 2 → 0 dans le scénario sain)
            source = "référentiel" if i % 4 else "saisie manuelle"
            journal.enregistrer_feedback(id_prediction, date, reelle, f"source : {source}")

    debut = (maintenant - timedelta(days=2)).isoformat(timespec="milliseconds")
    if degrade:
        journal.enregistrer_reentrainement(
            debut, debut, "erreur_activation", 40, "run-factice", "factice-1", "factice-1",
            resultats={"retour_arriere": {"necessaire": True, "pointeur_retabli": True,
                                          "alias_retabli": False, "verifie": True}},
            message_erreur="factice")
    else:
        journal.enregistrer_reentrainement(debut, debut, "mis_en_production", 40, "run-factice",
                                           "factice-1", "factice-2", resultats={"resultats_test": {}})


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sortie", default=str(RACINE / "outputs" / "suivi_demo"))
    args = parser.parse_args()
    # Ancrage sur le début de la semaine en cours : les prédictions remplissent les semaines PASSÉES,
    # toujours découpées de la même façon, quel que soit le jour du lancement (démonstration et tests
    # reproductibles ; la dernière semaine complète est celle qui déclenche les alertes localisées)
    maintenant = datetime.now(timezone.utc)
    debut_de_semaine = (maintenant - timedelta(days=maintenant.weekday())).replace(hour=0, minute=0,
                                                                                    second=0, microsecond=0)
    ancrage = debut_de_semaine - timedelta(seconds=1)
    profils = profils_factices()
    for scenario in ("sain", "degrade"):
        chemin = Path(args.sortie) / f"journal_{scenario}.db"
        chemin.unlink(missing_ok=True)           # toujours repartir d'un journal vide
        remplir(Journal(chemin), scenario, ancrage, profils=profils)
        print(f"Journal factice « {scenario} » : {chemin}")


if __name__ == "__main__":
    for flux in (sys.stdout, sys.stderr):
        flux.reconfigure(encoding="utf-8")
    main()
