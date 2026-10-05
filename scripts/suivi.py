"""Suivi du service : indicateurs depuis le journal de l'API, comparaison aux seuils, alertes (B10).

Usage (depuis la racine du projet) :
    python scripts/suivi.py                       # journal outputs/cisia.db, 30 derniers jours
    python scripts/suivi.py --base outputs/suivi_demo/journal_degrade.db --jours 3650
    python scripts/suivi.py --reference data/raw/dataset_trajectoire_emploi.csv   # + dérive (PSI, KS)

Version EN SERVICE : lue dans models/production/actuelle.json (source qui fait foi, celle de l'API) pour le
journal réel, ou dans le dossier donné par --production ; sinon déduite du dernier événement du journal
(prédiction ou réentraînement), et la source est indiquée dans le rapport.

Avec --reference, la dérive des données est mesurée par rapport à la partie ENTRAÎNEMENT de ce fichier
(même découpage que l'entraînement). En production, le profil de référence serait enregistré avec le
modèle au moment de l'entraînement, pour ne pas relire les données d'origine.

Sorties :
- le rapport à l'écran ;
- outputs/suivi/rapport_<date>.json (rapport complet, non versionné) et dernier_rapport.json (copie du
  dernier, servie par l'API sur /suivi et affichée dans la page « Suivi » de l'interface) ;
- outputs/suivi/derive_par_semaine.csv (avec --reference : PSI par semaine et par métier) ;
- outputs/suivi/alertes.log (une ligne par alerte, horodatée : à surveiller ou à brancher sur un
  outil d'alerte) ;
- code de sortie : 0 = aucune alerte, 1 = alerte « attention », 2 = alerte « critique ». Un
  planificateur (tâche planifiée Windows, cron, job CI) peut ainsi déclencher une notification.
  « Aucune alerte » ne veut pas dire « tout va bien » : le rapport liste aussi les indicateurs NON
  ÉVALUABLES (trop peu de données), avec la raison.

Prototype : lancé à la main. En production, il serait planifié chaque jour sur le serveur qui héberge le
journal (planification non configurée dans ce dépôt : voir les améliorations V2).
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from cisia.preparation import charger_donnees, decouper
from cisia.suivi import analyser, code_de_sortie, version_depuis_production

RACINE = Path(__file__).resolve().parents[1]
NIVEAUX_LOG = {"attention": logging.WARNING, "critique": logging.CRITICAL}
COLONNES_PROFIL = ["age", "niveau_diplome", "anciennete_poste_ans", "code_rome_vise", "synthese_entretien"]


def afficher(rapport):
    periode = rapport["periode"]
    print(f"Suivi du {rapport['date']} — service et dérive : {periode['jours']} derniers jours ; "
          f"situations observées enregistrées sur {periode['etiquettes_jours']} jours ; cohortes mûres : "
          f"prédictions de {periode['cohortes_mures_jours'][0]} à {periode['cohortes_mures_jours'][1]} jours")
    for section, valeurs in rapport["indicateurs"].items():
        print(f"\n[{section}]")
        for nom, valeur in valeurs.items():
            if isinstance(valeur, float):
                valeur = f"{valeur:.3f}"
            print(f"  {nom:42} {valeur}")
    print(f"\n{len(rapport['alertes'])} alerte(s)")
    for alerte in rapport["alertes"]:
        detail = (f" (valeur {alerte['valeur']}, seuil {alerte['seuil']})"
                  if alerte["valeur"] is not None else "")
        print(f"  [{alerte['niveau'].upper()}] {alerte['indicateur']} : {alerte['message']}{detail}")
    non_evaluables = rapport.get("non_evaluables") or []
    print(f"\n{len(non_evaluables)} indicateur(s) non évaluable(s)"
          + (" — aucune alerte ne signifie pas que tout va bien"
             if non_evaluables and not rapport["alertes"] else ""))
    for element in non_evaluables:
        print(f"  [NON ÉVALUABLE] {element['indicateur']} : {element['raison']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=os.getenv("CISIA_BASE", str(RACINE / "outputs" / "cisia.db")),
                        help="journal SQLite de l'API")
    parser.add_argument("--seuils", default=str(RACINE / "config" / "seuils_suivi.json"))
    parser.add_argument("--jours", type=int, default=None,
                        help="fenêtre glissante, en jours (défaut : fichier de seuils)")
    parser.add_argument("--reference", default=None,
                        help="données d'entraînement (CSV) pour mesurer la dérive (PSI, Kolmogorov-Smirnov)")
    parser.add_argument("--production", default=None,
                        help="dossier du modèle en production (actuelle.json) : source qui fait foi pour la "
                             "version en service. Défaut : models/production pour le journal réel ; aucun "
                             "pour un autre journal (version déduite du journal, source indiquée)")
    parser.add_argument("--sortie", default=str(RACINE / "outputs" / "suivi"),
                        help="dossier du rapport JSON et du journal des alertes")
    args = parser.parse_args()

    if not Path(args.base).exists():
        print(f"Journal introuvable : {args.base}")
        return 2
    seuils = json.loads(Path(args.seuils).read_text(encoding="utf-8"))
    reference = None
    if args.reference:
        entrainement, _ = decouper(charger_donnees(args.reference))
        reference = entrainement[COLONNES_PROFIL].to_dict("records")
    journal_reel = Path(args.base).resolve() == (RACINE / "outputs" / "cisia.db").resolve()
    production = args.production or (RACINE / "models" / "production" if journal_reel else None)
    version = version_depuis_production(production) if production else None
    rapport = analyser(args.base, seuils, args.jours, reference=reference, version_en_service=version)
    afficher(rapport)

    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    nom = f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    contenu = json.dumps(rapport, ensure_ascii=False, indent=2)
    (sortie / nom).write_text(contenu, encoding="utf-8")
    (sortie / "dernier_rapport.json").write_text(contenu, encoding="utf-8")   # lu par l'API (/suivi)
    if rapport["derive_par_semaine"]:   # tableau semaine × périmètre × variable (comme le M6)
        pd.DataFrame(rapport["derive_par_semaine"]).to_csv(sortie / "derive_par_semaine.csv", index=False)
    journal_alertes = logging.getLogger("cisia.suivi")
    journal_alertes.propagate = False   # alertes déjà affichées dans le rapport : pas de doublon à l'écran
    gestionnaire = logging.FileHandler(sortie / "alertes.log", encoding="utf-8")
    gestionnaire.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    journal_alertes.addHandler(gestionnaire)
    for alerte in rapport["alertes"]:
        journal_alertes.log(NIVEAUX_LOG[alerte["niveau"]], "%s : %s (base %s)", alerte["indicateur"],
                            alerte["message"], args.base)
    gestionnaire.close()
    print(f"\nRapport : {sortie / nom}")
    return code_de_sortie(rapport)


if __name__ == "__main__":
    for flux in (sys.stdout, sys.stderr):
        flux.reconfigure(encoding="utf-8")
    sys.exit(main())
