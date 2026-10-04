"""Textes et règles d'affichage de l'interface PHARE (sans Streamlit : testable seul).

Les libellés suivent l'ordre validé à l'étape 6 : délai estimé (classe) → recommandation →
niveau d'alerte → risque en %, au second plan.
"""

import calendar
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

FUSEAU = ZoneInfo("Europe/Paris")

# Classe prédite : titre, délai, libellé court (badges), ton de couleur
CLASSES = {
    0: {"titre": "Retour rapide", "delai": "Moins de 6 mois", "court": "Rapide · < 6 mois", "ton": "vert",
        "phrase": "Un accompagnement léger devrait suffire ; à confirmer avec l'usager."},
    1: {"titre": "Délai moyen", "delai": "6 à 12 mois", "court": "Moyen · 6–12 mois", "ton": "bleu",
        "phrase": "Un accompagnement standard est à construire avec l'usager."},
    2: {"titre": "Risque de longue durée", "delai": "Plus de 12 mois", "court": "Longue durée · > 12 mois",
        "ton": "ambre", "phrase": "Un besoin d'accompagnement renforcé est à examiner avec l'usager."},
}

DIPLOMES = ["Non renseigné", "Sans diplôme", "Bac", "Bac+2", "Bac+5"]
AGES = ["Non renseigné", *range(16, 71)]
ANCIENNETES = ["Non renseignée", *[i / 2 for i in range(0, 81)]]      # 0 à 40 ans, par demi-année

# Métiers fréquents (ROME v3, sous-ensemble) : liste fermée pour la saisie. En production : le
# référentiel ROME complet (France Travail). Le modèle n'utilise que le grand domaine (1re lettre).
METIERS_ROME = {
    "A1203": "Aménagement et entretien des espaces verts", "D1106": "Vente en alimentation",
    "D1214": "Vente en habillement et accessoires de la personne", "D1507": "Mise en rayon libre-service",
    "F1602": "Électricité bâtiment", "F1703": "Maçonnerie", "G1602": "Personnel de cuisine",
    "G1803": "Service en restauration", "I1604": "Mécanique automobile",
    "J1501": "Soins d'hygiène, de confort du patient", "K1302": "Assistance auprès d'adultes",
    "K1304": "Services domestiques", "K2204": "Nettoyage de locaux", "M1203": "Comptabilité",
    "M1601": "Accueil et renseignements", "M1607": "Secrétariat",
    "M1805": "Études et développement informatique", "N1101": "Conduite d'engins de déplacement des charges",
    "N1103": "Magasinage et préparation de commandes",
    "N4101": "Conduite de transport de marchandises sur longue distance",
}
REFERENTIEL = Path(__file__).resolve().parent / "referentiel_fictif.json"

MOTIFS = ["Informations complémentaires issues de l'entretien", "Situation personnelle non prise en compte",
          "Projet professionnel en évolution", "Autre"]

# Grands domaines du Répertoire opérationnel des métiers et des emplois (1re lettre du code ROME)
DOMAINES_ROME = {
    "A": "Agriculture et pêche, espaces naturels et verts, soins aux animaux",
    "B": "Arts et façonnage d'ouvrages d'art",
    "C": "Banque, assurance, immobilier",
    "D": "Commerce, vente et grande distribution",
    "E": "Communication, média et multimédia",
    "F": "Construction, bâtiment et travaux publics",
    "G": "Hôtellerie-restauration, tourisme, loisirs et animation",
    "H": "Industrie",
    "I": "Installation et maintenance",
    "J": "Santé",
    "K": "Services à la personne et à la collectivité",
    "L": "Spectacle",
    "M": "Support à l'entreprise",
    "N": "Transport et logistique",
}

PERIODES = ["Tout", "Aujourd'hui", "7 derniers jours", "Ce mois-ci"]
STATUTS_RETOUR = ["Tous les statuts", "À examiner", "Confirmée", "Corrigée"]


# Démonstration : les dates du référentiel fictif sont RELATIVES au jour de lancement (inscription il y a
# quelques jours = premier entretien), et la situation observée est lue à une date SIMULÉE, 400 jours plus
# tard : la chronologie reste cohérente (prédiction au premier entretien, issue connue ensuite).
DECALAGE_OBSERVATION_JOURS = 400
FENETRE_PREMIER_ENTRETIEN_JOURS = 31


def date_observation_demo(jour=None):
    return (jour or date.today()) + timedelta(days=DECALAGE_OBSERVATION_JOURS)


def charger_referentiel(chemin=REFERENTIEL, jour=None):
    """Dossiers du référentiel FICTIF (démonstration), indexés par identifiant, avec des dates réelles.

    Dans le fichier : inscription_il_y_a_jours et reprise_apres_jours (null = pas de reprise connue).
    """
    jour = jour or date.today()
    dossiers = {}
    for brut in json.loads(Path(chemin).read_text(encoding="utf-8"))["dossiers"]:
        inscription = jour - timedelta(days=brut["inscription_il_y_a_jours"])
        reprise = None if brut["reprise_apres_jours"] is None else \
            inscription + timedelta(days=brut["reprise_apres_jours"])
        dossiers[brut["id_usager"]] = {
            "id_usager": brut["id_usager"], "date_naissance": brut["date_naissance"],
            "niveau_diplome": brut["niveau_diplome"], "anciennete_poste_ans": brut["anciennete_poste_ans"],
            "date_inscription": inscription.isoformat(),
            "date_reprise_emploi": reprise.isoformat() if reprise else None}
    return dossiers


def age_au(date_naissance, jour=None):
    """Âge en années révolues. Seul l'âge est envoyé au modèle, jamais la date de naissance."""
    naissance, jour = date.fromisoformat(date_naissance), jour or date.today()
    return jour.year - naissance.year - ((jour.month, jour.day) < (naissance.month, naissance.day))


def ajouter_mois(jour, mois):
    """Même jour, `mois` mois plus tard (fin de mois si le jour n'existe pas : 31/01 + 1 mois = 28/02)."""
    total = jour.month - 1 + mois
    annee, mois_cible = jour.year + total // 12, total % 12 + 1
    return date(annee, mois_cible, min(jour.day, calendar.monthrange(annee, mois_cible)[1]))


def situation_depuis_referentiel(dossier, jour):
    """Classe OBSERVÉE au jour `jour`, en mois calendaires ; None si elle n'est pas encore connue.

    Reprise avant 6 mois → 0 ; de 6 à 12 mois inclus → 1 ; après 12 mois → 2.
    Sans reprise connue à ce jour (ou reprise future) : toujours en recherche après 12 mois → 2 ;
    sinon, pas encore connue. Des dates incohérentes (reprise avant l'inscription) lèvent ValueError.
    """
    inscription = date.fromisoformat(dossier["date_inscription"])
    reprise = dossier.get("date_reprise_emploi")
    reprise = date.fromisoformat(reprise) if reprise else None
    if reprise is not None and reprise < inscription:
        raise ValueError("Dates incohérentes : reprise d'emploi avant l'inscription.")
    six_mois, douze_mois = ajouter_mois(inscription, 6), ajouter_mois(inscription, 12)
    if reprise is not None and reprise <= jour:                  # reprise déjà constatée
        return 0 if reprise < six_mois else 1 if reprise <= douze_mois else 2
    return 2 if jour > douze_mois else None


def premier_entretien_valide(dossier, date_prediction):
    """La prédiction a-t-elle été faite au premier entretien, avant toute reprise d'emploi ?

    Sinon, la situation observée ne doit pas servir d'étiquette : l'issue était peut-être déjà connue.
    """
    inscription = date.fromisoformat(dossier["date_inscription"])
    jour = datetime.fromisoformat(date_prediction).astimezone(FUSEAU).date()
    reprise = dossier.get("date_reprise_emploi")
    avant_reprise = reprise is None or jour < date.fromisoformat(reprise)
    dans_la_fenetre = inscription <= jour <= inscription + timedelta(days=FENETRE_PREMIER_ENTRETIEN_JOURS)
    return dans_la_fenetre and avant_reprise


def libelle_metier(code):
    code = (code or "").strip().upper()
    return f"{code} · {METIERS_ROME[code]}" if code in METIERS_ROME else libelle_rome(code)


def libelle_rome(code):
    """« M1607 » → « M1607 · Support à l'entreprise ».

    Seul le grand domaine est affiché : pas de libellé de métier inventé sans le référentiel ROME complet.
    """
    code = (code or "").strip().upper()
    domaine = DOMAINES_ROME.get(code[:1])
    return f"{code} · {domaine}" if domaine else code


def date_locale(date_iso, avec_a=False):
    """Date UTC de l'API → « 04/10/2026 · 14:32 » (heure de Paris)."""
    try:
        date = datetime.fromisoformat(date_iso)
    except (TypeError, ValueError):
        return date_iso or ""
    if date.tzinfo is None:   # anciennes lignes sans fuseau
        date = date.replace(tzinfo=FUSEAU)
    date = date.astimezone(FUSEAU)
    return date.strftime("%d/%m/%Y à %H:%M" if avec_a else "%d/%m/%Y · %H:%M")


def retour_conseiller(element):
    """Statut de l'avis du conseiller : (libellé, ton)."""
    avis = element.get("avis_conseiller")
    if not avis:
        return "À examiner", "gris"
    if avis["avis"] == "confirme":
        return "Confirmée", "vert"
    return f"Corrigée : {CLASSES[avis['classe_proposee']]['court'].lower()}", "bleu"


def situation_observee(element):
    """Situation réelle connue plus tard (feedback) : (libellé, ton)."""
    observee = element.get("situation_observee")
    if not observee:
        return "Non connue", "gris"
    return CLASSES[observee["classe_reelle"]]["court"], "vert"


def texte_regle(regle):
    """Phrase qui dit comment la règle de décision a choisi l'estimation (ce que SHAP n'explique pas)."""
    if not regle:
        return None
    risque = f"{regle['risque_recalibre']:.0%}"
    if regle["motif"] == "seuil_classe_2":
        return (f"Le risque estimé de longue durée ({risque}) atteint le seuil de "
                f"{regle['seuil_classe_2']:.0%} : accompagnement renforcé.")
    if regle["motif"] == "garde_fou":
        return (f"Le modèle penchait pour un retour rapide, mais le risque estimé de longue durée ({risque}) "
                f"atteint le garde-fou de {regle['seuil_garde_fou']:.0%} : l'estimation est relevée à "
                "« délai moyen » par prudence.")
    if regle["motif"] == "plus_probable_0_1":
        texte = (f"Le risque estimé de longue durée ({risque}) reste sous le seuil de "
                 f"{regle['seuil_classe_2']:.0%} : le modèle retient la plus probable entre retour rapide "
                 "et délai moyen.")
        if regle["classe_la_plus_probable"] == 2:
            texte += " (Avant recalibration, « longue durée » était l'issue la plus probable.)"
        return texte
    return "Le modèle retient l'estimation la plus probable."


def lecture_explication(explication, nombre_mots=4):
    """Explication SHAP de l'API → barres affichables (largeur relative au facteur le plus fort) et mots.

    sens : « pour » (rapproche de l'estimation), « contre » (en éloigne), « neutre » (effet négligeable).
    None si la prédiction n'a pas d'explication (modèle antérieur à l'ajout de SHAP).
    """
    if not explication:
        return None
    facteurs = explication["facteurs"]
    maximum = max((abs(f["contribution"]) for f in facteurs), default=0) or 1
    lignes = []
    for facteur in facteurs:
        largeur = round(100 * abs(facteur["contribution"]) / maximum)
        sens = "neutre" if largeur < 2 else "pour" if facteur["contribution"] > 0 else "contre"
        lignes.append({"facteur": facteur["facteur"], "sens": sens, "largeur": largeur})
    mots = explication.get("mots") or []
    return {"lignes": lignes,
            "mots_pour": [m["mot"] for m in mots if m["contribution"] > 0][:nombre_mots],
            "mots_contre": [m["mot"] for m in mots if m["contribution"] < 0][:nombre_mots]}


def usager_pour_api(age, diplome, anciennete, code_rome, synthese):
    """Valeurs du formulaire → champ « usager » de /predict (vide = information manquante)."""
    return {"age": age, "niveau_diplome": None if diplome == "Non renseigné" else diplome,
            "anciennete_poste_ans": anciennete, "code_rome_vise": (code_rome or "").strip().upper(),
            "synthese_entretien": (synthese or "").strip()}


def filtrer(historique, identifiant="", periode="Tout", classe=None, statut="Tous les statuts",
            maintenant=None):
    """Prédictions acceptées de l'historique, filtrées comme dans l'écran « Historique »."""
    maintenant = (maintenant or datetime.now(FUSEAU)).astimezone(FUSEAU)
    debut = {"Aujourd'hui": maintenant.replace(hour=0, minute=0, second=0, microsecond=0),
             "7 derniers jours": maintenant - timedelta(days=7),
             "Ce mois-ci": maintenant.replace(day=1, hour=0, minute=0, second=0, microsecond=0)}.get(periode)
    resultat = []
    for element in historique:
        if element.get("statut") != "ok" or not element.get("sorties"):
            continue
        if identifiant and identifiant.strip().lower() not in (element.get("id_usager") or "").lower():
            continue
        if classe is not None and element["sorties"]["classe"] != classe:
            continue
        if statut != "Tous les statuts" and not retour_conseiller(element)[0].startswith(statut):
            continue
        if debut is not None:
            try:
                date = datetime.fromisoformat(element["date"])
            except (TypeError, ValueError):
                continue
            if date.tzinfo is None:
                date = date.replace(tzinfo=FUSEAU)
            if date < debut:
                continue
        resultat.append(element)
    return resultat
