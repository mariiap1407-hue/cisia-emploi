"""Suivi du service en production (B10) : indicateurs calculés depuis le journal, comparés à des seuils.

Le journal SQLite de l'API (outputs/cisia.db) est lu en LECTURE SEULE. Indicateurs, sur une fenêtre
glissante (30 jours par défaut) :

1. Service : requêtes acceptées, refusées (saisie invalide), en erreur, sans modèle ; latence (p50, p95).
2. Prédictions : répartition des classes, part de la classe 2 comparée au jeu de test, part des
   recommandations relevées par le garde-fou.
3. Accord conseillers / modèle : part des estimations confirmées (avis du conseiller, jamais réentraîné).
4. Performance réelle : sur les situations OBSERVÉES, mêmes indicateurs et mêmes seuils que le quality
   gate (erreurs critiques ≤ 10 %, rappel de la classe 2 ≥ 60 %, F1 macro ≥ 0,62), dès qu'il y en a assez.
5. Dérive des synthèses : longueur comparée aux synthèses d'entraînement (63 à 77 caractères), négations.
   C'est le signal de déclenchement de la V2 (re-sélection du modèle sur des données réelles).
   Biais des étiquettes : situations observées par source et par classe prédite.
6. Réentraînements : statuts, et retour arrière (pointeur rétabli ? alias du registre rétabli ? vérifié ?).
7. Dérive des données (si une référence est fournie : les données d'ENTRAÎNEMENT) : PSI (Population
   Stability Index) et test de Kolmogorov-Smirnov sur l'âge, l'ancienneté et la longueur de la synthèse ;
   PSI sur le diplôme, le domaine ROME et la répartition des classes prédites.
   Lecture du PSI (convention courante) : < 0,10 stable ; 0,10 à 0,25 dérive modérée (à surveiller) ;
   ≥ 0,25 dérive forte (alerte). Aussi SEMAINE PAR SEMAINE, au niveau global et par domaine métier
   (comme le M6 : parc × semaine), avec un seuil plus haut sur ces petites cellules (0,35).
   Le KS donne la significativité statistique (p-valeur) : avec beaucoup de données, il détecte des
   écarts minimes ; c'est le PSI, qui mesure l'AMPLEUR, qui déclenche l'alerte.

Chaque seuil franchi produit une alerte « attention » ou « critique ». Un indicateur calculé sur trop peu
de données est signalé « insuffisant », sans alerte (pas de fausse alarme sur 3 cas).
"""

import json
import re
import sqlite3
import warnings
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.exceptions import UndefinedMetricWarning

from cisia.evaluation import SEUILS_QUALITE, mesurer, verifier_seuils_qualite

NEGATION = re.compile(r"\b(pas|aucune?|sans|jamais|rien)\b|\bn['’]", re.IGNORECASE)


def lire_journal(chemin):
    """Lignes des tables du journal, en lecture seule (le service continue d'écrire à côté)."""
    uri = Path(chemin).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connexion:
        connexion.row_factory = sqlite3.Row

        def table(nom):
            existe = connexion.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                                       (nom,)).fetchone()
            return [dict(ligne) for ligne in connexion.execute(f"SELECT * FROM {nom}")] if existe else []

        return {nom: table(nom) for nom in ("inferences", "avis_conseillers", "feedbacks", "reentrainements")}


def date_utc(texte):
    try:
        date = datetime.fromisoformat(texte)
    except (TypeError, ValueError):
        return None
    return date if date.tzinfo else date.replace(tzinfo=timezone.utc)


def dans_la_fenetre(lignes, debut, champ="date"):
    return [ligne for ligne in lignes if (date_utc(ligne.get(champ)) or debut) >= debut]


def derniers_par_prediction(lignes, cle):
    """Le plus récent (identifiant le plus grand) pour chaque prédiction : celui qui fait foi."""
    retenus = {}
    for ligne in sorted(lignes, key=lambda ligne: ligne[cle]):
        retenus[ligne["id_prediction"]] = ligne
    return list(retenus.values())


def part(n, total):
    return None if total == 0 else round(n / total, 3)


EPSILON = 1e-4   # évite log(0) quand une case est vide d'un côté


def psi_numerique(reference, actuel, n_cases=10):
    """PSI d'une variable numérique : cases = déciles de la RÉFÉRENCE (valeurs manquantes ignorées)."""
    reference = np.asarray(reference, dtype=float)
    actuel = np.asarray(actuel, dtype=float)
    reference, actuel = reference[~np.isnan(reference)], actuel[~np.isnan(actuel)]
    bornes = np.unique(np.quantile(reference, np.linspace(0, 1, n_cases + 1)[1:-1]))
    cases_ref = np.bincount(np.searchsorted(bornes, reference, side="right"), minlength=len(bornes) + 1)
    cases_act = np.bincount(np.searchsorted(bornes, actuel, side="right"), minlength=len(bornes) + 1)
    return _psi(cases_ref / cases_ref.sum(), cases_act / cases_act.sum())


def psi_categoriel(reference, actuel):
    """PSI d'une variable catégorielle (les modalités absentes d'un côté comptent pour EPSILON)."""
    modalites = sorted(set(reference) | set(actuel), key=str)
    ref = np.array([list(reference).count(m) for m in modalites]) / len(reference)
    act = np.array([list(actuel).count(m) for m in modalites]) / len(actuel)
    return _psi(ref, act)


def psi_proportions(reference, actuel):
    return _psi(np.asarray(reference, dtype=float), np.asarray(actuel, dtype=float))


def _psi(ref, act):
    ref, act = np.clip(ref, EPSILON, None), np.clip(act, EPSILON, None)
    return round(float(np.sum((act - ref) * np.log(act / ref))), 4)


def niveau_psi(valeur, seuils):
    return ("forte" if valeur >= seuils["psi_derive_forte"] else
            "modérée" if valeur >= seuils["psi_derive_moderee"] else "stable")


def profil(lignes):
    """Variables comparées, à partir des entrées BRUTES (journal) ou du CSV de référence."""
    def valeur(ligne, champ):   # valeur manquante (None dans le journal, NaN dans le CSV) → None
        v = ligne.get(champ)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else v

    def nombre(ligne, champ):
        v = valeur(ligne, champ)
        return np.nan if v is None else float(v)

    return {
        "age": [nombre(e, "age") for e in lignes],
        "anciennete_poste_ans": [nombre(e, "anciennete_poste_ans") for e in lignes],
        "longueur_synthese": [float(len(valeur(e, "synthese_entretien") or "")) for e in lignes],
        "niveau_diplome": [valeur(e, "niveau_diplome") or "Non renseigné" for e in lignes],
        "domaine_rome": [str(valeur(e, "code_rome_vise") or "?").strip().upper()[:1] for e in lignes],
    }


def derive_des_donnees(reference, entrees, repartition_predite, seuils):
    """PSI et KS entre la référence (liste de dicts, données d'entraînement) et les entrées reçues."""
    ref, act = profil(reference), profil(entrees)
    resultats = {}
    for variable in ("age", "anciennete_poste_ans", "longueur_synthese"):
        valeurs_act = [v for v in act[variable] if not np.isnan(v)]
        valeurs_ref = [v for v in ref[variable] if not np.isnan(v)]
        if len(valeurs_act) < seuils["min_pour_derive"]:
            resultats[variable] = {"statut": "insuffisant"}
            continue
        psi = psi_numerique(valeurs_ref, valeurs_act)
        ks = ks_2samp(valeurs_ref, valeurs_act)
        resultats[variable] = {"psi": psi, "derive": niveau_psi(psi, seuils),
                               "ks_statistique": round(float(ks.statistic), 4),
                               "ks_p_valeur": float(f"{ks.pvalue:.3g}")}
    for variable in ("niveau_diplome", "domaine_rome"):
        if len(act[variable]) < seuils["min_pour_derive"]:
            resultats[variable] = {"statut": "insuffisant"}
            continue
        psi = psi_categoriel(ref[variable], act[variable])
        resultats[variable] = {"psi": psi, "derive": niveau_psi(psi, seuils)}
    if repartition_predite is not None:
        psi = psi_proportions(seuils["repartition_classes_reference"], repartition_predite)
        resultats["classes_predites"] = {"psi": psi, "derive": niveau_psi(psi, seuils)}
    return resultats


def semaine_iso(texte):
    date = date_utc(texte)
    if date is None:
        return None
    annee, semaine, _ = date.isocalendar()
    return f"{annee}-S{semaine:02d}"


def derive_par_semaine(reference, requetes, seuils):
    """PSI semaine par semaine, au niveau global ET par domaine métier (1re lettre du code ROME).

    Comme le M6 (parc × semaine) : un écart localisé sur un métier peut disparaître dans la moyenne.
    Sur ces petites cellules, le PSI est plus bruité (son biais vaut environ (cases - 1) / effectif) :
    5 cases au lieu de 10, seuil plus haut (psi_derive_cellule) et effectif minimal (min_par_cellule).
    Renvoie une ligne par (semaine, périmètre, variable).
    """
    ref = profil(reference)
    acceptees = [(semaine_iso(r["date"]), json.loads(r["entrees"])) for r in requetes
                 if r["statut"] == "ok" and r["entrees"]]
    lignes = []
    for semaine in sorted({s for s, _ in acceptees if s}):
        entrees = [e for s, e in acceptees if s == semaine]
        perimetres = {"__global__": entrees}
        for e in entrees:
            domaine = str(e.get("code_rome_vise") or "?").strip().upper()[:1]
            perimetres.setdefault(domaine, []).append(e)
        for perimetre, groupe in perimetres.items():
            if len(groupe) < seuils["min_par_cellule"]:
                continue
            act = profil(groupe)
            for variable in ("age", "anciennete_poste_ans", "longueur_synthese", "niveau_diplome"):
                if variable == "niveau_diplome":
                    psi, p_valeur = psi_categoriel(ref[variable], act[variable]), None
                else:
                    valeurs = [v for v in act[variable] if not np.isnan(v)]
                    if len(valeurs) < seuils["min_par_cellule"]:
                        continue
                    references = [v for v in ref[variable] if not np.isnan(v)]
                    psi = psi_numerique(references, valeurs, n_cases=seuils["cases_par_cellule"])
                    p_valeur = float(f"{ks_2samp(references, valeurs).pvalue:.3g}")
                # Dans une petite cellule, l'écart doit être GRAND (PSI) et SIGNIFICATIF (KS, variables
                # numériques) : sur 50 à 80 usagers, le défaut du KS (trop sensible sur de gros volumes)
                # ne joue pas, et il écarte les PSI élevés dus au seul hasard de l'échantillon.
                significatif = p_valeur is None or p_valeur < seuils["ks_p_valeur_cellule"]
                lignes.append({"semaine": semaine, "perimetre": perimetre, "variable": variable,
                               "n": len(groupe), "psi": psi, "ks_p_valeur": p_valeur,
                               "derive": psi >= seuils["psi_derive_cellule"] and significatif})
    return lignes


def analyser(chemin_journal, seuils, jours=None, maintenant=None, reference=None):
    """Rapport de suivi : période, indicateurs et alertes (liste vide = rien à signaler)."""
    jours = jours or seuils["fenetre_jours"]
    maintenant = maintenant or datetime.now(timezone.utc)
    debut = maintenant - timedelta(days=jours)
    journal = lire_journal(chemin_journal)
    alertes = []

    def alerter(niveau, indicateur, message, valeur=None, seuil=None):
        alertes.append({"niveau": niveau, "indicateur": indicateur, "message": message,
                        "valeur": valeur, "seuil": seuil})

    indicateurs = {}

    # 1. Service ------------------------------------------------------------------------------------
    requetes = dans_la_fenetre(journal["inferences"], debut)
    statuts = {s: sum(r["statut"] == s for r in requetes)
               for s in ("ok", "invalide", "erreur", "indisponible")}
    total = len(requetes)
    taux_erreurs = part(statuts["erreur"] + statuts["indisponible"], total)
    taux_refus = part(statuts["invalide"], total)
    durees = [r["duree_ms"] for r in requetes if r["statut"] == "ok" and r["duree_ms"] is not None]
    latence = ({"p50_ms": round(float(np.percentile(durees, 50)), 1),
                "p95_ms": round(float(np.percentile(durees, 95)), 1)} if durees else None)
    indicateurs["service"] = {"requetes": total, **statuts, "taux_erreurs_service": taux_erreurs,
                              "taux_requetes_refusees": taux_refus, "latence": latence,
                              "versions_servies": sorted({r["version_modele"] for r in requetes
                                                          if r["statut"] == "ok" and r["version_modele"]})}
    if taux_erreurs is not None and taux_erreurs > seuils["taux_erreurs_service_max"]:
        alerter("critique", "erreurs du service", "Trop de requêtes en erreur ou sans modèle en service.",
                round(taux_erreurs, 3), seuils["taux_erreurs_service_max"])
    if taux_refus is not None and taux_refus > seuils["taux_requetes_refusees_max"]:
        alerter("attention", "requêtes refusées", "Beaucoup de saisies refusées : formulaire ou intégration "
                "à vérifier.", round(taux_refus, 3), seuils["taux_requetes_refusees_max"])
    if latence and latence["p95_ms"] > seuils["latence_p95_ms_max"]:
        alerter("attention", "latence", "Le 95e centile du temps de prédiction dépasse le seuil.",
                latence["p95_ms"], seuils["latence_p95_ms_max"])

    # 2. Prédictions --------------------------------------------------------------------------------
    predictions = {r["id_prediction"]: json.loads(r["sorties"]) for r in requetes
                   if r["statut"] == "ok" and r["sorties"]}
    classes = [s["classe"] for s in predictions.values()]
    repartition = {str(c): part(classes.count(c), len(classes)) for c in (0, 1, 2)}
    motifs = [((s.get("explication") or {}).get("regle") or {}).get("motif") for s in predictions.values()]
    indicateurs["predictions"] = {"nombre": len(classes), "repartition_classes": repartition,
                                  "part_classe_2_reference": seuils["part_classe_2_reference"],
                                  "part_releves_par_le_garde_fou": part(motifs.count("garde_fou"),
                                                                        len(classes))}
    if len(classes) >= seuils["min_predictions_pour_distribution"]:
        part_2 = repartition["2"]
        if not seuils["part_classe_2_min"] <= part_2 <= seuils["part_classe_2_max"]:
            alerter("attention", "répartition des prédictions",
                    f"Part orientée en accompagnement renforcé inhabituelle (jeu de test : "
                    f"{seuils['part_classe_2_reference']:.1%}) : population ou données différentes ?",
                    round(part_2, 3), [seuils["part_classe_2_min"], seuils["part_classe_2_max"]])
    else:
        indicateurs["predictions"]["statut"] = "insuffisant"

    # 3. Accord conseillers / modèle ----------------------------------------------------------------
    avis = [a for a in derniers_par_prediction(journal["avis_conseillers"], "id_avis")
            if a["id_prediction"] in predictions]
    confirmes = sum(a["avis"] == "confirme" for a in avis)
    corrections = [a for a in avis if a["avis"] == "corrige"]
    accord = part(confirmes, len(avis))
    indicateurs["accord_conseillers"] = {
        "avis": len(avis), "taux_accord": accord,
        "corrections_vers_plus_d_accompagnement": sum(
            a["classe_proposee"] > predictions[a["id_prediction"]]["classe"] for a in corrections),
        "corrections_vers_moins_d_accompagnement": sum(
            a["classe_proposee"] < predictions[a["id_prediction"]]["classe"] for a in corrections),
        "motifs": {m: sum(a["motif"] == m for a in corrections)
                   for m in sorted({a["motif"] for a in corrections})}}
    if len(avis) >= seuils["min_avis"]:
        if accord < seuils["accord_conseillers_min"]:
            alerter("attention", "accord conseillers / modèle", "Les conseillers corrigent souvent "
                    "l'estimation : analyser les motifs de correction.", round(accord, 3),
                    seuils["accord_conseillers_min"])
    else:
        indicateurs["accord_conseillers"]["statut"] = "insuffisant"

    # 4. Performance réelle (situations observées) --------------------------------------------------
    observees = [f for f in derniers_par_prediction(journal["feedbacks"], "id_feedback")
                 if f["id_prediction"] in predictions]
    # Biais des étiquettes (comme le diagnostic de biais du M6) : ce que chaque source apporte, et la part
    # des prédictions qui reçoivent une situation observée, PAR CLASSE PRÉDITE. Si une classe ne revient
    # presque jamais (ex. les usagers orientés « léger » qu'on ne recontacte pas), le réentraînement
    # apprendrait sur un échantillon biaisé.
    par_source = {}
    for f in observees:
        source = (f.get("commentaire") or "non précisée").replace("source : ", "")
        par_source[source] = par_source.get(source, 0) + 1
    observees_par_classe = {c: sum(predictions[f["id_prediction"]]["classe"] == c for f in observees)
                            for c in (0, 1, 2)}
    couverture = {str(c): part(observees_par_classe[c], classes.count(c)) for c in (0, 1, 2)}
    indicateurs["performance_observee"] = {"situations_observees": len(observees), "par_source": par_source,
                                           "couverture_par_classe_predite": couverture}
    couvertures = [v for v in couverture.values() if v is not None]
    if len(observees) >= seuils["min_situations_observees"] and couvertures and max(couvertures) > 0:
        rapport_couverture = min(couvertures) / max(couvertures)
        if rapport_couverture < seuils["couverture_rapport_min"]:
            alerter("attention", "biais des étiquettes", "Les situations observées couvrent très inégalement "
                    "les classes prédites : un réentraînement apprendrait sur un échantillon biaisé.",
                    round(rapport_couverture, 3), seuils["couverture_rapport_min"])
    if len(observees) >= seuils["min_situations_observees"]:
        y_vrai = [f["classe_reelle"] for f in observees]
        y_pred = [predictions[f["id_prediction"]]["classe"] for f in observees]
        with warnings.catch_warnings():   # classe absente des situations observées : indicateur à 0
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            mesures = mesurer(y_vrai, y_pred)
        indicateurs["performance_observee"].update(
            {nom: (None if valeur is None or np.isnan(valeur) else round(float(valeur), 3))
             for nom, valeur in mesures.items() if nom in SEUILS_QUALITE})
        for echec in verifier_seuils_qualite(mesures):
            alerter("critique", "performance réelle", f"Seuil du quality gate non respecté sur les "
                    f"situations observées : {echec}. Envisager un réentraînement (/retrain).")
    else:
        indicateurs["performance_observee"]["statut"] = "insuffisant"

    # 5. Dérive des synthèses -----------------------------------------------------------------------
    syntheses = [json.loads(r["entrees"]).get("synthese_entretien") or "" for r in requetes
                 if r["statut"] == "ok" and r["entrees"]]
    if syntheses:
        hors_format = part(sum(len(s) > seuils["synthese_longueur_entrainement_max"] for s in syntheses),
                           len(syntheses))
        negations = part(sum(bool(NEGATION.search(s)) for s in syntheses), len(syntheses))
        indicateurs["syntheses"] = {"nombre": len(syntheses),
                                    "longueur_moyenne": round(float(np.mean([len(s) for s in syntheses])), 1),
                                    "part_plus_longues_que_l_entrainement": hors_format,
                                    "part_avec_negation": negations}
        if len(syntheses) >= seuils["min_predictions_pour_distribution"]:
            if hors_format > seuils["synthese_part_hors_format_max"]:
                alerter("attention", "dérive des synthèses", "Les synthèses reçues sont plus longues que "
                        "celles de l'entraînement : le modèle lit des textes qu'il n'a jamais vus (V2).",
                        round(hors_format, 3), seuils["synthese_part_hors_format_max"])
            if negations > seuils["synthese_part_negations_max"]:
                alerter("attention", "négations dans les synthèses", "Négations fréquentes : le modèle ne "
                        "les comprend pas (limite connue, V2).", round(negations, 3),
                        seuils["synthese_part_negations_max"])
        else:
            indicateurs["syntheses"]["statut"] = "insuffisant"

    # 6. Réentraînements ----------------------------------------------------------------------------
    reentrainements = sorted(dans_la_fenetre(journal["reentrainements"], debut, "date_debut"),
                             key=lambda r: r["id_reentrainement"])
    indicateurs["reentrainements"] = {
        "nombre": len(reentrainements),
        "statuts": {s: sum(r["statut"] == s for r in reentrainements)
                    for s in sorted({r["statut"] for r in reentrainements})}}
    for r in reentrainements:
        if r["statut"] not in ("erreur", "erreur_activation"):
            continue
        retour = (json.loads(r["resultats"]) if r["resultats"] else {}).get("retour_arriere")
        quand = f"réentraînement n° {r['id_reentrainement']} du {r['date_debut'][:10]}"
        if retour is None:
            alerter("attention", "réentraînement", f"Échec du {quand} ; état du retour arrière non tracé "
                    "(journal antérieur au B10) : vérifier la version en service.")
        elif not retour["verifie"] or retour["pointeur_retabli"] is False:
            alerter("critique", "réentraînement", f"Échec du {quand} ET retour arrière NON confirmé : "
                    "la version en service est incertaine.")
        elif retour["alias_retabli"] is False:
            alerter("critique", "registre MLflow", f"Échec du {quand} : pointeur rétabli, mais l'alias "
                    "« production » du registre n'a PAS été remis : registre et API divergent.")
        else:
            alerter("attention", "réentraînement", f"Échec du {quand} ; retour arrière vérifié, l'ancienne "
                    "version est restée en service.")

    # 7. Dérive des données (PSI, KS) par rapport aux données d'entraînement ----------------------------
    if reference is not None:
        entrees = [json.loads(r["entrees"]) for r in requetes if r["statut"] == "ok" and r["entrees"]]
        repartition_predite = ([classes.count(c) / len(classes) for c in (0, 1, 2)]
                               if len(classes) >= seuils["min_pour_derive"] else None)
        derive = derive_des_donnees(reference, entrees, repartition_predite, seuils)
        indicateurs["derive_des_donnees"] = derive
        for variable, resultat in derive.items():
            if resultat.get("derive") == "forte":
                alerter("attention", "dérive des données", f"Dérive forte de « {variable} » par rapport à la "
                        "référence (données d'entraînement ; jeu de test pour les classes prédites) : le "
                        "modèle travaille hors de son domaine de validité.",
                        resultat["psi"], seuils["psi_derive_forte"])
        # Semaine par semaine, global et par métier : alerte pour une dérive LOCALISÉE la dernière semaine
        par_semaine = derive_par_semaine(reference, requetes, seuils)
        if par_semaine:
            derniere = max(ligne["semaine"] for ligne in par_semaine)
            globales = {ligne["variable"] for ligne in par_semaine if ligne["semaine"] == derniere
                        and ligne["perimetre"] == "__global__" and ligne["derive"]}
            for ligne in par_semaine:
                # Dérive LOCALISÉE : un métier en dérive alors que le niveau global ne l'est pas (sinon,
                # c'est la même dérive globale, déjà signalée)
                if (ligne["semaine"] == derniere and ligne["perimetre"] != "__global__" and ligne["derive"]
                        and ligne["variable"] not in globales):
                    alerter("attention", "dérive localisée",
                            f"Semaine {derniere}, métiers « {ligne['perimetre']} » ({ligne['n']} usagers) : "
                            f"dérive forte de « {ligne['variable']} » (un écart localisé peut disparaître "
                            "dans la moyenne globale).", ligne["psi"], seuils["psi_derive_cellule"])
    else:
        indicateurs["derive_des_donnees"] = {"statut": "non calculée (pas de référence : option --reference)"}
        par_semaine = []

    return {"date": maintenant.isoformat(timespec="seconds"),
            "periode": {"debut": debut.isoformat(timespec="seconds"), "jours": jours},
            "indicateurs": indicateurs, "alertes": alertes, "derive_par_semaine": par_semaine}


def code_de_sortie(rapport):
    """0 : rien à signaler ; 1 : au moins une alerte « attention » ; 2 : au moins une alerte « critique »."""
    niveaux = {a["niveau"] for a in rapport["alertes"]}
    return 2 if "critique" in niveaux else 1 if niveaux else 0
