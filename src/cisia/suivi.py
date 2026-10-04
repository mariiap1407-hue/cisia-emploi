"""Suivi du service en production (B10), aligné sur le module M6 : indicateurs depuis le journal, seuils,
alertes.

Le journal SQLite de l'API (outputs/cisia.db) est lu en LECTURE SEULE. Trois périmètres temporels distincts,
car les informations n'arrivent pas au même rythme :
- FENÊTRE RÉCENTE (30 jours) : santé du service, répartition des prédictions, accord conseillers / modèle,
  synthèses, dérive des données ;
- ÉTIQUETTES RÉCENTES (365 jours) : performance réelle, sur les situations observées ENREGISTRÉES
  récemment, rattachées à leurs prédictions même anciennes (le délai réel n'est connu que 6 à 12 mois après
  la prédiction) ;
- COHORTES MÛRES (prédictions de 13 à 25 mois) : couverture des étiquettes par classe prédite, sur des
  prédictions qui ont toutes eu le temps de recevoir leur issue (sinon, une faible couverture refléterait
  seulement des issues pas encore connues).

Indicateurs :
1. Service : requêtes acceptées, refusées (saisie invalide), en erreur, sans modèle ; latence (p50, p95).
2. Prédictions : répartition des classes, part de la classe 2 comparée au jeu de test, garde-fou.
3. Accord conseillers / modèle (avis du conseiller, jamais utilisé pour réentraîner).
4. Performance réelle : mêmes seuils que le quality gate, chaque indicateur seulement s'il est ÉVALUABLE
   (rappel de la classe 2 et erreurs critiques : assez d'usagers réellement en classe 2 ; F1 macro : assez
   d'usagers dans chaque classe réelle). Un indicateur non évaluable n'est jamais une alerte.
   Risque de biais des étiquettes : provenance, et couverture par classe prédite sur les cohortes mûres.
5. Synthèses : longueur comparée à l'entraînement (63 à 77 caractères), négations (limite connue, V2).
6. Réentraînements : statuts, et retour arrière (pointeur rétabli ? alias du registre rétabli ? vérifié ?).
7. Dérive des données (avec une référence : les données d'ENTRAÎNEMENT) :
   - PSI (ampleur, décide) et test de Kolmogorov-Smirnov (significativité, indicatif) ; < 0,10 stable,
     0,10 à 0,25 modérée, >= 0,25 forte (alerte). Référence constante ou à faible cardinalité : PSI sur les
     valeurs elles-mêmes (des déciles n'auraient pas de sens). Taux de valeurs manquantes suivis à part.
   - Semaine par semaine, au niveau global ET par domaine métier ROME (comme le M6 : parc × semaine). Chaque
     métier est comparé à SA propre référence (le même métier dans l'entraînement) : sinon on mesurerait la
     différence entre métiers, pas une évolution dans le temps. Petites cellules : seuil 0,35 (comme le M6),
     5 cases, effectif minimal 50 des deux côtés, et KS significatif (p < 0,01) pour les variables
     numériques. Politique de confirmation face aux nombreux tests (semaine × métier × variable) : seule la
     DERNIÈRE semaine déclenche des alertes ; les cellules passées restent un historique à lire.

Prudence d'interprétation : une dérive est un SIGNAL À INVESTIGUER, pas une preuve de perte de performance.
Le KS suppose des distributions continues : avec des ex æquo (âges entiers, longueurs de texte), ses
p-valeurs sont approximatives. Les seuils sont une politique de départ, pas une garantie statistique.
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
from sklearn.metrics import f1_score

from cisia.evaluation import SEUILS_QUALITE, taux_erreurs_critiques

NEGATION = re.compile(r"\b(pas|aucune?|sans|jamais|rien)\b|\bn['’]", re.IGNORECASE)
EPSILON = 1e-4                     # évite log(0) quand une case est vide d'un côté
VARIABLES_NUMERIQUES = ("age", "anciennete_poste_ans", "longueur_synthese")
VARIABLES_MANQUANTES = {"age": np.nan, "anciennete_poste_ans": np.nan, "niveau_diplome": "Non renseigné"}


# --- Lecture du journal -----------------------------------------------------------------------------

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


def entre(lignes, debut, fin=None, champ="date"):
    """Lignes datées dans [debut, fin] (une date illisible est gardée : pas de perte silencieuse)."""
    retenues = []
    for ligne in lignes:
        date = date_utc(ligne.get(champ))
        if date is None or (date >= debut and (fin is None or date <= fin)):
            retenues.append(ligne)
    return retenues


def derniers_par_prediction(lignes, cle):
    """Le plus récent (identifiant le plus grand) pour chaque prédiction : celui qui fait foi."""
    retenus = {}
    for ligne in sorted(lignes, key=lambda ligne: ligne[cle]):
        retenus[ligne["id_prediction"]] = ligne
    return list(retenus.values())


def part(n, total):
    return None if total == 0 else round(n / total, 3)


# --- PSI et KS --------------------------------------------------------------------------------------

def _psi(ref, act):
    ref, act = np.clip(ref, EPSILON, None), np.clip(act, EPSILON, None)
    return round(float(np.sum((act - ref) * np.log(act / ref))), 4)


def psi_categoriel(reference, actuel):
    """PSI d'une variable catégorielle (une modalité absente d'un côté compte pour EPSILON)."""
    reference, actuel = list(reference), list(actuel)
    modalites = sorted(set(reference) | set(actuel), key=str)
    ref = np.array([reference.count(m) for m in modalites]) / len(reference)
    act = np.array([actuel.count(m) for m in modalites]) / len(actuel)
    return _psi(ref, act)


def psi_numerique(reference, actuel, n_cases=10, cardinalite_max=10):
    """PSI d'une variable numérique : cases = quantiles de la RÉFÉRENCE, bornes extrêmes infinies.

    Référence constante ou à faible cardinalité (<= cardinalite_max valeurs distinctes) : les quantiles se
    confondent et une valeur nouvelle tomberait dans la même case qu'une ancienne ; on compare alors les
    valeurs elles-mêmes (PSI catégoriel). Valeurs manquantes ignorées (leur taux est suivi à part).
    """
    reference = np.asarray(reference, dtype=float)
    actuel = np.asarray(actuel, dtype=float)
    reference, actuel = reference[~np.isnan(reference)], actuel[~np.isnan(actuel)]
    if len(np.unique(reference)) <= cardinalite_max:
        return psi_categoriel(np.round(reference, 3), np.round(actuel, 3))
    bornes = np.unique(np.quantile(reference, np.linspace(0, 1, n_cases + 1)[1:-1]))
    cases_ref = np.bincount(np.searchsorted(bornes, reference, side="right"), minlength=len(bornes) + 1)
    cases_act = np.bincount(np.searchsorted(bornes, actuel, side="right"), minlength=len(bornes) + 1)
    return _psi(cases_ref / cases_ref.sum(), cases_act / cases_act.sum())


def psi_proportions(reference, actuel):
    return _psi(np.asarray(reference, dtype=float), np.asarray(actuel, dtype=float))


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


def sans_manquants(valeurs):
    return [v for v in valeurs if not (isinstance(v, float) and np.isnan(v))]


def taux_manquant(valeurs, marqueur):
    if not valeurs:
        return None
    manquants = sum((isinstance(v, float) and np.isnan(v)) if marqueur is np.nan else v == marqueur
                    for v in valeurs)
    return round(manquants / len(valeurs), 3)


def comparer_variable(variable, ref_valeurs, act_valeurs, n_cases, cardinalite_max):
    """PSI (et KS pour une variable numérique) entre deux listes de valeurs, manquants exclus."""
    if variable in VARIABLES_NUMERIQUES:
        ref, act = sans_manquants(ref_valeurs), sans_manquants(act_valeurs)
        ks = ks_2samp(ref, act)
        return {"psi": psi_numerique(ref, act, n_cases, cardinalite_max), "n_reference": len(ref),
                "n": len(act), "ks_statistique": round(float(ks.statistic), 4),
                "ks_p_valeur": float(f"{ks.pvalue:.3g}")}
    return {"psi": psi_categoriel(ref_valeurs, act_valeurs), "n_reference": len(ref_valeurs),
            "n": len(act_valeurs), "ks_p_valeur": None}


def derive_des_donnees(reference, entrees, repartition_predite, seuils):
    """Fenêtre récente : PSI et KS par variable, et taux de valeurs manquantes, contre la référence."""
    ref, act = profil(reference), profil(entrees)
    resultats = {}
    for variable in (*VARIABLES_NUMERIQUES, "niveau_diplome", "domaine_rome"):
        if len(sans_manquants(act[variable])) < seuils["min_pour_derive"]:
            resultats[variable] = {"statut": "insuffisant", "n": len(sans_manquants(act[variable]))}
            continue
        resultat = comparer_variable(variable, ref[variable], act[variable], 10, seuils["cardinalite_max"])
        resultat["derive"] = niveau_psi(resultat["psi"], seuils)
        resultats[variable] = resultat
    for variable, marqueur in VARIABLES_MANQUANTES.items():
        if len(act[variable]) >= seuils["min_pour_derive"]:
            resultats.setdefault(variable, {}).update(
                taux_manquant_reference=taux_manquant(ref[variable], marqueur),
                taux_manquant=taux_manquant(act[variable], marqueur))
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

    Chaque métier est comparé au MÊME métier dans la référence (dérive dans le temps, pas écart entre
    métiers). Cellules : effectif minimal des deux côtés (après exclusion des manquants), 5 cases, seuil
    0,35, KS significatif pour les variables numériques. Une ligne par (semaine, périmètre, variable).
    """
    references = {"__global__": reference}
    for ligne in reference:
        domaine = str(ligne.get("code_rome_vise") or "?").strip().upper()[:1]
        references.setdefault(domaine, []).append(ligne)
    profils_reference = {perimetre: profil(lignes) for perimetre, lignes in references.items()}

    acceptees = [(semaine_iso(r["date"]), json.loads(r["entrees"])) for r in requetes
                 if r["statut"] == "ok" and r["entrees"]]
    lignes = []
    for semaine in sorted({s for s, _ in acceptees if s}):
        entrees = [e for s, e in acceptees if s == semaine]
        perimetres = {"__global__": entrees}
        for e in entrees:
            perimetres.setdefault(str(e.get("code_rome_vise") or "?").strip().upper()[:1], []).append(e)
        for perimetre, groupe in perimetres.items():
            if perimetre not in profils_reference:
                continue                          # métier absent de l'entraînement : pas de référence
            ref, act = profils_reference[perimetre], profil(groupe)
            for variable in (*VARIABLES_NUMERIQUES, "niveau_diplome"):
                n_ref, n_act = len(sans_manquants(ref[variable])), len(sans_manquants(act[variable]))
                if min(n_ref, n_act) < seuils["min_par_cellule"]:
                    continue
                resultat = comparer_variable(variable, ref[variable], act[variable],
                                             seuils["cases_par_cellule"], seuils["cardinalite_max"])
                significatif = resultat["ks_p_valeur"] is None or \
                    resultat["ks_p_valeur"] < seuils["ks_p_valeur_cellule"]
                lignes.append({"semaine": semaine, "perimetre": perimetre, "variable": variable,
                               "n": resultat["n"], "n_reference": resultat["n_reference"],
                               "psi": resultat["psi"], "ks_p_valeur": resultat["ks_p_valeur"],
                               "derive": resultat["psi"] >= seuils["psi_derive_cellule"] and significatif})
    return lignes


# --- Performance réelle ----------------------------------------------------------------------------

def performance_observee(y_vrai, y_pred, seuils):
    """Indicateurs du quality gate, chacun seulement s'il est évaluable (sinon None et une raison)."""
    y_vrai, y_pred = np.asarray(y_vrai), np.asarray(y_pred)
    n_par_classe = {c: int((y_vrai == c).sum()) for c in (0, 1, 2)}
    resultats, non_evaluables = {"n_par_classe_reelle": {str(c): n for c, n in n_par_classe.items()}}, []
    if n_par_classe[2] >= seuils["min_classe_2_reelle"]:
        classe_2 = y_vrai == 2
        resultats["Erreurs critiques (2→0)"] = round(float(taux_erreurs_critiques(y_vrai, y_pred)), 3)
        resultats["Rappel classe 2"] = round(float((y_pred[classe_2] == 2).mean()), 3)
    else:
        resultats["Erreurs critiques (2→0)"] = resultats["Rappel classe 2"] = None
        non_evaluables.append(f"erreurs critiques et rappel de la classe 2 : {n_par_classe[2]} usager(s) "
                              f"réellement en classe 2 (minimum {seuils['min_classe_2_reelle']})")
    if min(n_par_classe.values()) >= seuils["min_par_classe_reelle"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            # Périmètre explicite : les trois classes, toutes présentes en nombre suffisant
            f1 = f1_score(y_vrai, y_pred, labels=[0, 1, 2], average="macro")
            resultats["F1 macro"] = round(float(f1), 3)
    else:
        resultats["F1 macro"] = None
        non_evaluables.append(f"F1 macro : moins de {seuils['min_par_classe_reelle']} usagers dans au moins "
                              f"une classe réelle ({n_par_classe})")
    echecs = []
    for indicateur, (sens, seuil) in SEUILS_QUALITE.items():
        valeur = resultats[indicateur]
        if valeur is not None and ((sens == "max" and valeur > seuil) or (sens == "min" and valeur < seuil)):
            echecs.append(f"{indicateur} = {valeur:.3f} (seuil {sens} : {seuil})")
    return resultats, echecs, non_evaluables


# --- Rapport ----------------------------------------------------------------------------------------

def analyser(chemin_journal, seuils, jours=None, maintenant=None, reference=None):
    """Rapport de suivi : périodes, indicateurs, alertes, et indicateurs NON ÉVALUABLES (avec la raison)."""
    jours = jours or seuils["fenetre_jours"]
    maintenant = maintenant or datetime.now(timezone.utc)
    debut = maintenant - timedelta(days=jours)
    journal = lire_journal(chemin_journal)
    alertes, non_evaluables, indicateurs = [], [], {}

    def alerter(niveau, indicateur, message, valeur=None, seuil=None):
        alertes.append({"niveau": niveau, "indicateur": indicateur, "message": message,
                        "valeur": valeur, "seuil": seuil})

    def non_evaluable(indicateur, raison):
        non_evaluables.append({"indicateur": indicateur, "raison": raison})

    # 1. Service (fenêtre récente) --------------------------------------------------------------------
    requetes = entre(journal["inferences"], debut)
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
    if total >= seuils["min_requetes_service"]:
        if taux_erreurs > seuils["taux_erreurs_service_max"]:
            alerter("critique", "erreurs du service", "Trop de requêtes en erreur ou sans modèle en service.",
                    taux_erreurs, seuils["taux_erreurs_service_max"])
        if taux_refus > seuils["taux_requetes_refusees_max"]:
            alerter("attention", "requêtes refusées", "Beaucoup de saisies refusées : formulaire ou "
                    "intégration à vérifier.", taux_refus, seuils["taux_requetes_refusees_max"])
        if latence and latence["p95_ms"] > seuils["latence_p95_ms_max"]:
            alerter("attention", "latence", "Le 95e centile du temps de prédiction dépasse le seuil.",
                    latence["p95_ms"], seuils["latence_p95_ms_max"])
    else:
        non_evaluable("service (erreurs, refus, latence)",
                      f"{total} requête(s) (minimum {seuils['min_requetes_service']})")

    # 2. Prédictions (fenêtre récente) ------------------------------------------------------------------
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
        if not seuils["part_classe_2_min"] <= repartition["2"] <= seuils["part_classe_2_max"]:
            alerter("attention", "répartition des prédictions",
                    f"Part orientée en accompagnement renforcé inhabituelle (jeu de test : "
                    f"{seuils['part_classe_2_reference']:.1%}) : population ou données différentes ? "
                    "À vérifier.",
                    repartition["2"], [seuils["part_classe_2_min"], seuils["part_classe_2_max"]])
    else:
        non_evaluable("répartition des prédictions",
                      f"{len(classes)} prédiction(s) (minimum {seuils['min_predictions_pour_distribution']})")

    # 3. Accord conseillers / modèle (avis donnés sur les prédictions récentes) -------------------------
    avis = [a for a in derniers_par_prediction(journal["avis_conseillers"], "id_avis")
            if a["id_prediction"] in predictions]
    corrections = [a for a in avis if a["avis"] == "corrige"]
    accord = part(sum(a["avis"] == "confirme" for a in avis), len(avis))
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
                    "l'estimation : analyser les motifs de correction.", accord,
                    seuils["accord_conseillers_min"])
    else:
        non_evaluable("accord conseillers / modèle", f"{len(avis)} avis (minimum {seuils['min_avis']})")

    # 4. Performance réelle : étiquettes ENREGISTRÉES récemment, prédictions de toute date ---------------
    toutes = {r["id_prediction"]: r for r in journal["inferences"] if r["statut"] == "ok" and r["sorties"]}
    debut_etiquettes = maintenant - timedelta(days=seuils["fenetre_etiquettes_jours"])
    etiquettes = [f for f in derniers_par_prediction(journal["feedbacks"], "id_feedback")
                  if f["id_prediction"] in toutes]
    recentes = entre(etiquettes, debut_etiquettes)
    par_source = {}
    for f in recentes:
        source = (f.get("commentaire") or "non précisée").replace("source : ", "")
        par_source[source] = par_source.get(source, 0) + 1
    delais = [(date_utc(f["date"]) - date_utc(toutes[f["id_prediction"]]["date"])).days for f in recentes
              if date_utc(f["date"]) and date_utc(toutes[f["id_prediction"]]["date"])]
    performance = {"situations_observees": len(recentes), "par_source": par_source,
                   "delai_median_jours": int(np.median(delais)) if delais else None}
    if len(recentes) >= seuils["min_situations_observees"]:
        y_vrai = [f["classe_reelle"] for f in recentes]
        y_pred = [json.loads(toutes[f["id_prediction"]]["sorties"])["classe"] for f in recentes]
        mesures, echecs, raisons = performance_observee(y_vrai, y_pred, seuils)
        performance.update(mesures)
        for raison in raisons:
            non_evaluable("performance réelle", raison)
        for echec in echecs:
            alerter("critique", "performance réelle", "Seuil du quality gate non respecté sur les "
                    f"situations observées : {echec}. Analyser les cas, puis envisager un réentraînement "
                    "(/retrain).")
    else:
        non_evaluable("performance réelle", f"{len(recentes)} situation(s) observée(s) enregistrée(s) "
                      f"sur {seuils['fenetre_etiquettes_jours']} jours "
                      f"(minimum {seuils['min_situations_observees']})")

    # Risque de biais des étiquettes : couverture par classe prédite, sur des COHORTES MÛRES seulement
    fin_cohorte = maintenant - timedelta(days=seuils["maturite_jours"])
    debut_cohorte = maintenant - timedelta(days=seuils["cohorte_max_jours"])
    cohorte = entre(list(toutes.values()), debut_cohorte, fin_cohorte)
    etiquetees = {f["id_prediction"] for f in etiquettes}
    couverture, effectifs = {}, {}
    for c in (0, 1, 2):
        ids = [r["id_prediction"] for r in cohorte if json.loads(r["sorties"])["classe"] == c]
        effectifs[str(c)] = len(ids)
        couverture[str(c)] = (part(sum(i in etiquetees for i in ids), len(ids))
                              if len(ids) >= seuils["min_par_classe_couverture"] else None)
    performance["cohortes_mures"] = {"predictions": len(cohorte), "par_classe_predite": effectifs,
                                     "couverture_par_classe_predite": couverture}
    evaluables = [v for v in couverture.values() if v is not None]
    if len(evaluables) >= 2 and max(evaluables) > 0:
        rapport_couverture = round(min(evaluables) / max(evaluables), 3)
        if rapport_couverture < seuils["couverture_rapport_min"]:
            alerter("attention", "risque de biais des étiquettes", "Sur les cohortes mûres, les "
                    "situations observées couvrent très inégalement les classes prédites : risque de biais "
                    "à investiguer avant tout réentraînement.", rapport_couverture,
                    seuils["couverture_rapport_min"])
    else:
        non_evaluable("risque de biais des étiquettes",
                      f"cohortes mûres (prédictions de {seuils['maturite_jours']} à "
                      f"{seuils['cohorte_max_jours']} jours) : effectifs par classe {effectifs} "
                      f"(minimum {seuils['min_par_classe_couverture']})")
    indicateurs["performance_observee"] = performance

    # 5. Synthèses (fenêtre récente) --------------------------------------------------------------------
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
                        "celles de l'entraînement : le modèle lit des textes d'un format qu'il n'a pas "
                        "appris (V2).",
                        hors_format, seuils["synthese_part_hors_format_max"])
            if negations > seuils["synthese_part_negations_max"]:
                alerter("attention", "négations dans les synthèses", "Négations fréquentes : le modèle ne "
                        "les comprend pas (limite connue, V2).", negations,
                        seuils["synthese_part_negations_max"])
        else:
            non_evaluable("synthèses", f"{len(syntheses)} synthèse(s) "
                                       f"(minimum {seuils['min_predictions_pour_distribution']})")

    # 6. Réentraînements (fenêtre récente) --------------------------------------------------------------
    reentrainements = sorted(entre(journal["reentrainements"], debut, champ="date_debut"),
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

    # 7. Dérive des données (fenêtre récente, puis semaine par semaine) ---------------------------------
    par_semaine = []
    if reference is not None:
        entrees = [json.loads(r["entrees"]) for r in requetes if r["statut"] == "ok" and r["entrees"]]
        repartition_predite = ([classes.count(c) / len(classes) for c in (0, 1, 2)]
                               if len(classes) >= seuils["min_pour_derive"] else None)
        derive = derive_des_donnees(reference, entrees, repartition_predite, seuils)
        indicateurs["derive_des_donnees"] = derive
        deja_signalees = set()
        for variable, resultat in derive.items():
            if resultat.get("derive") == "forte":
                deja_signalees.add(variable)
                alerter("attention", "dérive des données", f"Dérive forte de « {variable} » sur {jours} "
                        "jours par rapport à la référence (données d'entraînement ; jeu de test pour les "
                        "classes prédites) : signal à investiguer, pas une preuve de perte de performance.",
                        resultat["psi"], seuils["psi_derive_forte"])
            taux, taux_reference = resultat.get("taux_manquant"), resultat.get("taux_manquant_reference")
            if taux is not None and taux_reference is not None:
                ecart = round(taux - taux_reference, 3)
                if ecart > seuils["ecart_taux_manquants_max"]:
                    alerter("attention", "valeurs manquantes", f"« {variable} » est beaucoup plus souvent "
                            f"non renseigné qu'à l'entraînement ({resultat['taux_manquant_reference']:.0%} → "
                            f"{resultat['taux_manquant']:.0%}) : saisie ou intégration à vérifier.",
                            ecart, seuils["ecart_taux_manquants_max"])
            if resultat.get("statut") == "insuffisant":
                non_evaluable("dérive des données", f"« {variable} » : {resultat['n']} valeur(s) "
                                                    f"(minimum {seuils['min_pour_derive']})")

        par_semaine = derive_par_semaine(reference, requetes, seuils)
        if par_semaine:
            derniere = max(ligne["semaine"] for ligne in par_semaine)
            # a) Dérive GLOBALE de la dernière semaine : alerte explicite (elle peut passer inaperçue sur
            #    30 jours), sauf si la même variable est déjà signalée sur une période qui la contient
            globales = set()
            for ligne in par_semaine:
                if ligne["semaine"] == derniere and ligne["perimetre"] == "__global__" and ligne["derive"]:
                    globales.add(ligne["variable"])
                    if ligne["variable"] not in deja_signalees:
                        alerter("attention", "dérive de la dernière semaine",
                                f"Semaine {derniere} ({ligne['n']} usagers) : dérive forte de "
                                f"« {ligne['variable']} » au niveau global, invisible sur la fenêtre "
                                "complète.", ligne["psi"],
                                seuils["psi_derive_cellule"])
            # b) Dérive LOCALISÉE : un métier dérive (par rapport à SA référence) sans dérive globale
            for ligne in par_semaine:
                if (ligne["semaine"] == derniere and ligne["perimetre"] != "__global__" and ligne["derive"]
                        and ligne["variable"] not in globales):
                    alerter("attention", "dérive localisée",
                            f"Semaine {derniere}, métiers « {ligne['perimetre']} » ({ligne['n']} usagers) : "
                            f"dérive forte de « {ligne['variable']} » par rapport à ce même métier dans "
                            "l'entraînement (un écart localisé peut disparaître dans la moyenne globale).",
                            ligne["psi"],
                            seuils["psi_derive_cellule"])
    else:
        indicateurs["derive_des_donnees"] = {"statut": "non calculée (pas de référence : option --reference)"}

    return {"date": maintenant.isoformat(timespec="seconds"),
            "periode": {"debut": debut.isoformat(timespec="seconds"), "jours": jours,
                        "etiquettes_jours": seuils["fenetre_etiquettes_jours"],
                        "cohortes_mures_jours": [seuils["maturite_jours"], seuils["cohorte_max_jours"]]},
            "indicateurs": indicateurs, "alertes": alertes, "non_evaluables": non_evaluables,
            "derive_par_semaine": par_semaine}


def code_de_sortie(rapport):
    """0 : aucune alerte ; 1 : au moins une alerte « attention » ; 2 : au moins une alerte « critique »."""
    niveaux = {a["niveau"] for a in rapport["alertes"]}
    return 2 if "critique" in niveaux else 1 if niveaux else 0
