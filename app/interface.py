"""PHARE · interface conseiller (Streamlit). Elle n'appelle que l'API : aucun modèle, aucune donnée ici.

Écrans : nouvelle analyse (formulaire) → résultat → appréciation du conseiller ; historique ; aide.
Lancement (depuis la racine du projet, l'API tournant à côté) :
    streamlit run app/interface.py
"""

import base64
import time
from datetime import date, datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

import client
from presentation import (
    AGES,
    ANCIENNETES,
    CLASSES,
    DIPLOMES,
    METIERS_ROME,
    MOTIFS,
    NOMS_VARIABLES,
    PERIODES,
    STATUTS_RETOUR,
    age_au,
    alertes_triees,
    carte_par_metier,
    charger_referentiel,
    date_locale,
    episode_de_la_prediction,
    filtrer,
    lecture_explication,
    libelle_metier,
    premier_entretien_valide,
    retour_conseiller,
    semaines_en_derive,
    situation_depuis_referentiel,
    situation_observee,
    texte_regle,
    usager_pour_api,
)

DOSSIER = Path(__file__).resolve().parent
LOGO = DOSSIER / "assets" / "logo_phare.png"
INTRO = DOSSIER / "assets" / "intro_phare.gif"
DUREE_INTRO_S = 7.2          # durée de l'animation (96 images × 75 ms) : jouée une seule fois
LIGNES_PAR_PAGE = 10
LIMITE_HISTORIQUE = 500       # l'historique filtre côté interface : seulement ces dernières analyses
MAX_SYNTHESE = 250           # données d'entraînement : 63 à 77 caractères ; au-delà de ~3×, hors distribution
REFERENTIEL = charger_referentiel()
LIBELLES_CHAMPS = {"id_usager": "Identifiant usager", "age": "Âge", "niveau_diplome": "Niveau de diplôme",
                   "anciennete_poste_ans": "Ancienneté", "code_rome_vise": "Métier visé (code ROME)",
                   "synthese_entretien": "Synthèse de l'entretien"}

st.set_page_config(page_title="PHARE · Aide à l'orientation", page_icon=str(LOGO), layout="wide")
st.markdown(f"<style>{(DOSSIER / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

etat = st.session_state
etat.setdefault("page", "analyse")
etat.setdefault("saisie", {})
etat.setdefault("courant", None)
etat.setdefault("message", None)
etat.setdefault("page_historique", 0)
etat.setdefault("dossier", None)       # dossier choisi : conservé hors widget (Streamlit efface
                                        # l'état d'un widget quand sa page n'est plus affichée)


# --- Outils d'affichage -----------------------------------------------------------------------

def html(contenu):
    st.markdown(contenu, unsafe_allow_html=True)


def aller(page):
    etat.page = page


def badge(texte, ton):
    return f'<span class="badge {ton}">{escape(texte)}</span>'


def info(texte):
    html(f'<div class="info"><span class="i">i</span><span>{texte}</span></div>')


def prototype():
    html('<div style="text-align:center"><span class="prototype">Prototype · Données fictives</span></div>')


def afficher_message():
    if etat.message:
        genre, texte = etat.message
        (st.success if genre == "ok" else st.error)(texte)
        etat.message = None


# --- Écran d'accueil : animation jouée une seule fois par session --------------------------------

if not etat.get("intro_vue"):
    gif = base64.b64encode(INTRO.read_bytes()).decode()
    html(f'<div class="intro"><img src="data:image/gif;base64,{gif}" alt="PHARE"></div>')
    time.sleep(DUREE_INTRO_S)
    etat.intro_vue = True
    st.rerun()


# --- En-tête ------------------------------------------------------------------------------------

def entete():
    with st.container(key="entete"):
        colonnes = st.columns([2.3, 1.5, 1.1, 0.8, 0.8, 2.2, 1.8], vertical_alignment="center")
        colonnes[0].image(str(LOGO), width=220)
        for colonne, (page, libelle) in zip(colonnes[1:5], [("analyse", "Nouvelle analyse"),
                                                           ("historique", "Historique"), ("suivi", "Suivi"),
                                                           ("aide", "Aide")]):
            actif = etat.page == page or (page == "analyse" and etat.page in ("resultat", "appreciation"))
            with colonne.container(key=f"nav_{page}{'_actif' if actif else ''}"):
                st.button(libelle, on_click=aller, args=(page,), key=f"bouton_nav_{page}")
        colonnes[6].markdown('<div class="espace">👤 Espace conseiller</div>', unsafe_allow_html=True)


def pied():
    html('<div class="pied"><span><b>PHARE</b> — Prototype pédagogique</span>'
         '<span>Confidentialité &nbsp;|&nbsp; Accessibilité</span></div>')


# --- Nouvelle analyse ---------------------------------------------------------------------------

SAISIE_MANUELLE = "Saisie manuelle (dossier hors référentiel)"
AUTRE_METIER = "Autre code ROME…"


def valeurs_initiales():
    """Valeurs de départ du formulaire : dossier du référentiel choisi, ou saisie précédente."""
    if etat.saisie.get("dossier") == etat.dossier:     # retour depuis le résultat : saisie conservée
        return etat.saisie
    dossier = REFERENTIEL.get(etat.dossier)
    if dossier:
        return {"id_usager": dossier["id_usager"], "age": age_au(dossier["date_naissance"]),
                "diplome": dossier["niveau_diplome"] or "Non renseigné",
                "anciennete": dossier["anciennete_poste_ans"], "code_rome": "", "synthese": ""}
    return etat.saisie


def reinitialiser_formulaire():
    etat.saisie, etat.dossier = {}, None
    etat.pop("dossier_choisi", None)
    etat.version_formulaire = etat.get("version_formulaire", 0) + 1   # nouvelles clés : champs vidés


def page_analyse():
    html('<div class="fil">Accueil &nbsp;›&nbsp; <b>Nouvelle analyse</b></div>')
    st.title("Nouvelle analyse")
    html('<div class="sous-titre">Recherchez le dossier de l\'usager, vérifiez ses informations et '
         'rédigez la synthèse de l\'entretien.</div>')
    info("La décision d'orientation reste celle du conseiller.")
    afficher_message()

    st.subheader("1. Dossier de l'usager")
    choix = [None, *REFERENTIEL, SAISIE_MANUELLE]
    st.selectbox("Rechercher un dossier", choix, key="dossier_choisi", placeholder="Identifiant du dossier…",
                 index=choix.index(etat.dossier) if etat.dossier in choix else 0,
                 on_change=lambda: setattr(etat, "dossier", etat.dossier_choisi),
                 format_func=lambda c: "—" if c is None else c if c in (SAISIE_MANUELLE,) else
                 f"{c} · inscrit le {date_locale(REFERENTIEL[c]['date_inscription']).split(' ·')[0]}")
    html('<p class="aide-saisie">Démonstration : référentiel <b>fictif</b> de 20 dossiers. En service, les '
         'informations administratives viendraient du référentiel national des usagers.</p>')
    if etat.dossier is None:
        return

    depart = valeurs_initiales()
    depuis_referentiel = etat.dossier in REFERENTIEL
    version = f"{etat.get('version_formulaire', 0)}_{etat.dossier}"
    with st.form(f"analyse_{version}", border=False):
        if depuis_referentiel:
            html('<p class="aide-saisie">Âge, diplôme et ancienneté viennent du référentiel : modifiez-les '
                 'seulement s\'ils sont inexacts. L\'âge est calculé à partir de la date de naissance, qui '
                 'n\'est jamais transmise au modèle. Le métier visé se décide avec l\'usager pendant '
                 'l\'entretien.</p>')
        gauche, droite = st.columns(2)
        id_usager = gauche.text_input("Identifiant usager *", value=depart.get("id_usager", ""),
                                      placeholder="ex. DE-0248", disabled=depuis_referentiel,
                                      help="Identifiant pseudonyme du dossier : jamais de nom.")
        age = droite.selectbox("Âge", AGES,
                               index=AGES.index(depart.get("age") or "Non renseigné"))
        diplome = gauche.selectbox("Niveau de diplôme", DIPLOMES,
                                   index=DIPLOMES.index(depart.get("diplome") or "Non renseigné"))
        anciennete_depart = depart.get("anciennete")
        anciennete = droite.selectbox(
            "Ancienneté dans le dernier poste", ANCIENNETES,
            index=ANCIENNETES.index(anciennete_depart if anciennete_depart is not None else "Non renseignée"),
            format_func=lambda v: v if isinstance(v, str)
            else f"{v:g} an{'s' if v > 1 else ''}".replace(".", ","))
        metiers = [*METIERS_ROME, AUTRE_METIER]
        code_depart = depart.get("code_rome", "")
        metier = gauche.selectbox("Métier visé (code ROME) *", metiers,
                                  index=metiers.index(code_depart) if code_depart in METIERS_ROME else None,
                                  placeholder="Choisissez un métier…",
                                  format_func=lambda c: c if c == AUTRE_METIER else libelle_metier(c))
        autre_code = droite.text_input("Autre code ROME (si non listé)",
                                       placeholder="Lettre + 4 chiffres, ex. H2909",
                                       value="" if code_depart in METIERS_ROME else code_depart)

        st.subheader("2. Synthèse de l'entretien")
        synthese = st.text_area(
            "Synthèse de l'entretien *", value=depart.get("synthese", ""), height=130, max_chars=MAX_SYNTHESE,
            placeholder="Projet : …\nExpériences : …\nFreins éventuels (mobilité, garde d'enfants, santé, "
                        "langue…) : …")
        html(f'<p class="aide-saisie">2 à 3 phrases courtes, sur le modèle ci-dessus '
             f'({MAX_SYNTHESE} caractères au plus). Ne saisissez ni nom ni coordonnées.</p>')
        html('<p class="aide-saisie">Commune, nationalité et statut d\'allocataire ne sont pas demandés : le '
             'modèle ne les utilise pas (minimisation des données). * Champs obligatoires.</p>')
        _, bouton_analyse, bouton_reinit, _ = st.columns([2, 1.6, 1.1, 2])
        analyser = bouton_analyse.form_submit_button("Analyser la situation →", type="primary",
                                                     width="stretch")
        reinitialiser = bouton_reinit.form_submit_button("Réinitialiser", width="stretch",
                                                         on_click=reinitialiser_formulaire)

    if reinitialiser or not analyser:
        return
    code_rome = autre_code.strip().upper() if metier in (None, AUTRE_METIER) else metier
    age = None if age == "Non renseigné" else age
    anciennete = None if anciennete == "Non renseignée" else anciennete
    etat.saisie = {"dossier": etat.dossier, "id_usager": id_usager, "age": age, "diplome": diplome,
                   "anciennete": anciennete, "code_rome": code_rome, "synthese": synthese}
    manquants = [nom for nom, valeur in [("Identifiant usager", id_usager), ("Métier visé", code_rome),
                                         ("Synthèse de l'entretien", synthese)] if not (valeur or "").strip()]
    if manquants:
        st.error("Champs obligatoires manquants : " + ", ".join(manquants) + ".")
        return
    usager = usager_pour_api(age, diplome, anciennete, code_rome, synthese)
    try:
        with st.spinner("Analyse en cours…"):
            prediction = client.predire(usager, id_usager.strip())
    except client.ErreurAPI as erreur:
        st.error(erreur.message)
        for detail in erreur.details:
            champ = LIBELLES_CHAMPS.get((detail.get("champ") or [""])[-1], "Saisie")
            st.markdown(f"- **{champ}** : {detail.get('message')}")
        return
    etat.courant = {"id_prediction": prediction["id_prediction"], "id_usager": id_usager.strip(),
                    "date": datetime.now(timezone.utc).isoformat(), "entrees": usager, "sorties": prediction,
                    "avis": None, "observee": None, "source": "analyse"}
    aller("resultat")
    st.rerun()


# --- Résultat -----------------------------------------------------------------------------------

def carte_resultat(sorties):
    """La recommandation d'abord (ce que le conseiller utilise), le délai et le risque en explication."""
    classe = CLASSES[sorties["classe"]]
    details = (f"Délai de retour à l'emploi estimé : <b>{classe['delai'].lower()}</b> &nbsp;·&nbsp; "
               f"Niveau d'alerte : <b>{escape(sorties['niveau_alerte'])}</b> &nbsp;·&nbsp; "
               f"Risque estimé de chômage de longue durée : <b>{escape(sorties['risque_affiche'])}</b>")
    html(f'<div class="carte {classe["ton"]}"><div class="etiquette">Recommandation</div>'
         f'<div class="titre">{escape(sorties["recommandation"])}</div>'
         f'<div class="phrase">{classe["phrase"]}</div><div class="details">{details}</div></div>')


def explication_resultat(sorties):
    """Pourquoi cette recommandation ? 1. la règle de décision ; 2. les facteurs du modèle (SHAP)."""
    st.subheader("Pourquoi cette recommandation ?")
    explication = sorties.get("explication")
    lecture = lecture_explication(explication)
    if lecture is None:
        html('<p class="aide-saisie">Explication indisponible : prédiction faite avant l\'ajout de '
             'l\'explicabilité.</p>')
        return
    regle = texte_regle(explication.get("regle"))
    if regle:
        html(f'<p class="shap-regle"><b>Règle de décision</b> — {escape(regle)}</p>')

    titre = CLASSES[explication["classe_expliquee"]]["titre"]
    html(f'<p class="shap-titre"><b>Ce qui a pesé dans le score du modèle pour « {escape(titre)} »</b></p>')
    lignes = "".join(
        f'<div class="shap-ligne"><span class="shap-nom">{escape(ligne["facteur"])}</span>'
        f'<span class="shap-gauche">{barre(ligne, "contre")}</span>'
        f'<span class="shap-droite">{barre(ligne, "pour")}</span></div>' for ligne in lecture["lignes"])
    html('<div class="shap"><div class="shap-ligne shap-legende"><span></span>'
         '<span class="shap-gauche">← diminue ce score</span>'
         f'<span class="shap-droite">augmente ce score →</span></div>{lignes}</div>')
    mots = []
    if lecture["mots_pour"]:
        mots.append("augmentent : " + " ".join(badge(m, "bleu") for m in lecture["mots_pour"]))
    if lecture["mots_contre"]:
        mots.append("diminuent : " + " ".join(badge(m, "gris") for m in lecture["mots_contre"]))
    if mots:
        html('<p class="shap-mots">Repères dans la synthèse (indicatifs) — ' + " &nbsp;·&nbsp; ".join(mots)
             + "</p>")
    html('<p class="shap-note">Méthode SHAP : part de chaque information dans le score brut du modèle, '
         'pour cet usager, avant recalibration et règle de décision. Les repères de la synthèse sont '
         'approximatifs (fragments de caractères regroupés par mot) : seule la ligne « Synthèse » fait foi. '
         'Cela décrit le calcul du modèle, pas la cause de la situation de l\'usager.</p>')


def barre(ligne, sens):
    if ligne["sens"] != sens:
        return ""
    return f'<span class="barre {sens}" style="width:{ligne["largeur"]}%"></span>'


def informations_analysees(entrees):
    valeurs = [f"{entrees['age']:g} ans" if entrees.get("age") is not None else "Âge non renseigné",
               entrees.get("niveau_diplome") or "Diplôme non renseigné",
               (f"{entrees['anciennete_poste_ans']:g} an(s) d'ancienneté"
                if entrees.get("anciennete_poste_ans") is not None else "Ancienneté non renseignée"),
               libelle_metier(entrees.get("code_rome_vise"))]
    html('<div class="analysees">' + "".join(f"<span>{escape(v)}</span>" for v in valeurs) + "</div>")


def page_resultat():
    courant = etat.courant
    if not courant:
        aller("analyse")
        st.rerun()
    sorties, entrees = courant["sorties"], courant["entrees"]
    if courant["source"] == "analyse":
        with st.container(key="lien_retour"):
            st.button("← Modifier les informations", on_click=aller, args=("analyse",))
    else:
        with st.container(key="lien_retour"):
            st.button("← Retour à l'historique", on_click=aller, args=("historique",))
    prototype()
    st.title("Résultat de l'analyse")
    html(f'<div class="sous-titre">Usager {escape(courant["id_usager"] or "—")} · '
         f'{date_locale(courant["date"], avec_a=True)}</div>')
    afficher_message()
    carte_resultat(sorties)
    info("Cette estimation est une aide à la décision. Vous conservez la responsabilité de l'orientation.")
    explication_resultat(sorties)

    st.subheader("Informations analysées")
    informations_analysees(entrees)
    with st.expander("Consulter la synthèse de l'entretien"):
        st.write(entrees.get("synthese_entretien") or "—")

    avis = courant.get("avis")
    if avis:
        libelle, ton = retour_conseiller({"avis_conseiller": avis})
        quand = f" — le {date_locale(avis['date'], avec_a=True)}" if avis.get("date") else ""
        html(f'<p class="discret">Appréciation du conseiller : {badge(libelle, ton)}{escape(quand)}</p>')
        if avis.get("avis") == "corrige":
            html(f'<p class="discret">Motif : {escape(avis.get("motif") or "—")}'
                 + (f' · Précisions : {escape(avis["precisions"])}' if avis.get("precisions") else "")
                 + "</p>")
    html('<div class="question">Cette estimation correspond-elle à votre appréciation ?</div>')
    _, confirmer, corriger, _ = st.columns([2, 1.5, 1.5, 2])
    if confirmer.button("Confirmer l'estimation", type="primary", width="stretch"):
        try:
            reponse = client.envoyer_avis(courant["id_prediction"], "confirme")
            courant["avis"] = {"avis": "confirme", "classe_proposee": reponse["classe_proposee"],
                               "date": datetime.now(timezone.utc).isoformat()}
            etat.message = ("ok", "Votre confirmation est enregistrée.")
        except client.ErreurAPI as erreur:
            etat.message = ("erreur", erreur.message)
        st.rerun()
    corriger.button("Proposer une correction", on_click=aller, args=("appreciation",), width="stretch")

    if courant["source"] == "historique":
        situation_observee_formulaire(courant)

    _, liens, _ = st.columns([2, 2, 2])
    with liens:
        gauche, droite = st.columns(2)
        with gauche.container(key="lien_nouvelle"):
            st.button("Nouvelle analyse", on_click=nouvelle_analyse)
        with droite.container(key="lien_historique"):
            st.button("Voir l'historique", on_click=aller, args=("historique",))
    html(f'<p class="discret">Modèle : version {escape(str(sorties.get("version_modele", "")))}</p>')


def enregistrer_observee(courant, classe, source):
    try:
        client.envoyer_situation_observee(courant["id_prediction"], classe, source)
        courant["observee"] = {"classe_reelle": classe}
        etat.message = ("ok", "La situation observée est enregistrée.")
    except client.ErreurAPI as erreur:
        etat.message = ("erreur", erreur.message)
    st.rerun()


def situation_observee_formulaire(courant):
    st.subheader("Situation observée")
    info("À renseigner plus tard, quand le délai réel de retour à l'emploi est connu. C'est la seule "
         "information utilisée pour réentraîner le modèle (l'appréciation ci-dessus ne l'est jamais).")
    observee = courant.get("observee")
    if observee:
        html(f'<p>Situation enregistrée : {badge(CLASSES[observee["classe_reelle"]]["court"], "vert")}</p>')
    cle = courant["id_prediction"]

    # 1. Voie normale : rapprochement avec le référentiel des usagers (dates d'inscription et de reprise)
    # Épisode ancré sur la date de la prédiction : identique quel que soit le jour de consultation
    episode = episode_de_la_prediction(courant.get("id_usager"), courant["date"])
    dossier, observation = episode if episode else (None, None)
    if dossier and not premier_entretien_valide(dossier, courant["date"]):
        html('<p class="aide-saisie">Cette analyse n\'a pas été faite au premier entretien (ou l\'a été '
             'après une reprise d\'emploi) : la situation du référentiel ne peut pas lui être rattachée.</p>')
    elif dossier:
        html(f'<p class="aide-saisie">Démonstration : le référentiel fictif est lu à une date d\'observation '
             f'<b>simulée</b>, le {observation.strftime("%d/%m/%Y")}.</p>')
        if st.button("Interroger le référentiel", key=f"referentiel_{cle}"):
            try:
                etat[f"referentiel_{cle}_resultat"] = situation_depuis_referentiel(dossier, observation)
            except ValueError as erreur:
                etat[f"referentiel_{cle}_resultat"] = str(erreur)
            etat[f"referentiel_{cle}_interroge"] = True
        if etat.get(f"referentiel_{cle}_interroge"):
            connue = etat.get(f"referentiel_{cle}_resultat")
            reprise = dossier.get("date_reprise_emploi")
            if isinstance(connue, str):
                st.error(f"Référentiel : {connue} Rien n'est enregistré.")
            elif connue is None:
                info("Référentiel : pas encore de reprise d'emploi et moins de 12 mois depuis l'inscription. "
                     "La situation n'est pas encore connue : rien à enregistrer pour l'instant.")
            else:
                constatee = reprise and date.fromisoformat(reprise) <= observation
                detail = (f"reprise d'emploi le {date.fromisoformat(reprise).strftime('%d/%m/%Y')}"
                          if constatee else "toujours en recherche plus de 12 mois après l'inscription")
                html(f'<p>Référentiel : {escape(detail)} → {badge(CLASSES[connue]["court"], "vert")}</p>')
                if st.button("Enregistrer la situation du référentiel", type="primary",
                             key=f"enreg_ref_{cle}"):
                    enregistrer_observee(courant, connue, "référentiel")
    else:
        html('<p class="aide-saisie">Dossier absent du référentiel : saisie manuelle uniquement.</p>')

    # 2. Repli : saisie manuelle par le conseiller
    with st.expander("Saisie manuelle", expanded=not dossier):
        choix = st.radio("Délai réellement constaté", list(CLASSES), horizontal=True, index=None,
                         format_func=lambda c: CLASSES[c]["court"], key=f"observee_choix_{cle}")
        if st.button("Enregistrer la situation observée", disabled=choix is None, key=f"enreg_manuel_{cle}"):
            enregistrer_observee(courant, choix, "saisie manuelle")


def nouvelle_analyse():
    etat.saisie, etat.courant, etat.dossier = {}, None, None
    etat.pop("dossier_choisi", None)
    aller("analyse")


# --- Appréciation du conseiller -----------------------------------------------------------------

def page_appreciation():
    courant = etat.courant
    if not courant:
        aller("analyse")
        st.rerun()
    classe_predite = courant["sorties"]["classe"]
    with st.container(key="lien_retour"):
        st.button("← Retour au résultat", on_click=aller, args=("resultat",))
    prototype()
    st.title("Votre appréciation professionnelle")
    html(f'<div class="sous-titre">Usager {escape(courant["id_usager"] or "—")}</div>')
    classe = CLASSES[classe_predite]
    html(f'<div class="carte" style="display:flex;justify-content:space-between;align-items:center">'
         f'<div><div class="etiquette">Estimation initiale du modèle</div>'
         f'<div style="font-size:1.3rem;font-weight:700">{classe["titre"]}</div></div>'
         f'{badge(classe["court"], classe["ton"])}</div>')

    st.subheader("Quel délai estimez-vous plus adapté ?")
    proposee = st.radio("Délai estimé", list(CLASSES), index=classe_predite, label_visibility="collapsed",
                        format_func=lambda c: f"{CLASSES[c]['titre']} · {CLASSES[c]['delai'].lower()}")
    motif = st.selectbox("Motif de la correction *", MOTIFS, index=None, placeholder="Choisissez un motif")
    precisions = st.text_area("Précisions", max_chars=1000, height=90,
                              placeholder="Décrivez les éléments qui motivent votre appréciation.")
    html('<p class="aide-saisie">Sans ajouter de données personnelles inutiles.</p>')
    info("Votre appréciation est conservée séparément de la prédiction initiale. Elle ne modifie pas le "
         "modèle et ne sert pas à le réentraîner.")
    _, enregistrer, annuler, _ = st.columns([2, 1.6, 1, 2])
    if enregistrer.button("Enregistrer mon retour →", type="primary", width="stretch"):
        if proposee != classe_predite and motif is None:
            st.error("Choisissez un motif pour proposer une correction.")
            return
        try:
            if proposee == classe_predite:
                client.envoyer_avis(courant["id_prediction"], "confirme")
                courant["avis"] = {"avis": "confirme", "classe_proposee": classe_predite,
                                   "date": datetime.now(timezone.utc).isoformat()}
            else:
                client.envoyer_avis(courant["id_prediction"], "corrige", proposee, motif, precisions)
                courant["avis"] = {"avis": "corrige", "classe_proposee": proposee, "motif": motif,
                                   "precisions": (precisions or "").strip() or None,
                                   "date": datetime.now(timezone.utc).isoformat()}
            etat.message = ("ok", "Votre appréciation est enregistrée.")
        except client.ErreurAPI as erreur:
            etat.message = ("erreur", erreur.message)
        aller("resultat")
        st.rerun()
    annuler.button("Annuler", on_click=aller, args=("resultat",), width="stretch")


# --- Historique ---------------------------------------------------------------------------------

def consulter(element):
    etat.courant = {"id_prediction": element["id_prediction"], "id_usager": element.get("id_usager"),
                    "date": element["date"], "entrees": element["entrees"], "sorties": element["sorties"],
                    "avis": element.get("avis_conseiller"), "observee": element.get("situation_observee"),
                    "source": "historique"}
    aller("resultat")


def reinitialiser_filtres():
    for cle in ("filtre_id", "filtre_periode", "filtre_classe", "filtre_statut"):
        etat.pop(cle, None)
    etat.page_historique = 0


def page_historique():
    prototype()
    titre, bouton = st.columns([4, 1.3], vertical_alignment="bottom")
    with titre:
        st.title("Historique des analyses")
        html('<div class="sous-titre">Retrouvez les estimations et les retours des conseillers.</div>')
    bouton.button("＋ Nouvelle analyse", type="primary", on_click=nouvelle_analyse, width="stretch")
    afficher_message()
    try:
        historique = client.historique(LIMITE_HISTORIQUE)
    except client.ErreurAPI as erreur:
        st.error(erreur.message)
        return

    f1, f2, f3, f4 = st.columns([2, 1.2, 1.3, 1.3])
    identifiant = f1.text_input("Identifiant usager", placeholder="Rechercher un identifiant…",
                                key="filtre_id")
    periode = f2.selectbox("Période", PERIODES, key="filtre_periode")
    choix_classes = [None, *CLASSES]
    classe = f3.selectbox("Estimation", choix_classes, key="filtre_classe",
                          format_func=lambda c: "Toutes les classes" if c is None else CLASSES[c]["court"])
    statut = f4.selectbox("Retour conseiller", STATUTS_RETOUR, key="filtre_statut")
    lignes = filtrer(historique, identifiant, periode, classe, statut)

    gauche, droite = st.columns([4, 1])
    gauche.markdown(f"**{len(lignes)} analyse{'s' if len(lignes) > 1 else ''}** "
                    f"<span class='aide-saisie'>(recherche parmi les {LIMITE_HISTORIQUE} dernières)</span>",
                    unsafe_allow_html=True)
    with droite.container(key="lien_filtres"):
        st.button("Réinitialiser les filtres", on_click=reinitialiser_filtres)

    largeurs = [1.1, 1.5, 1.9, 2.2, 1.7, 1.0]
    for colonne, titre_colonne in zip(st.columns(largeurs), ["Usager", "Date et heure", "Estimation initiale",
                                                             "Retour conseiller", "Situation observée",
                                                             "Action"]):
        colonne.markdown(f'<div class="ligne-titre">{titre_colonne}</div>', unsafe_allow_html=True)

    nombre_pages = max(1, -(-len(lignes) // LIGNES_PAR_PAGE))
    etat.page_historique = min(etat.page_historique, nombre_pages - 1)
    debut = etat.page_historique * LIGNES_PAR_PAGE
    for element in lignes[debut:debut + LIGNES_PAR_PAGE]:
        classe_element = CLASSES[element["sorties"]["classe"]]
        retour, ton_retour = retour_conseiller(element)
        observee, ton_observee = situation_observee(element)
        cellules = st.columns(largeurs, vertical_alignment="center")
        cellules[0].markdown(f'<div class="cellule">{escape(element.get("id_usager") or "—")}</div>',
                             unsafe_allow_html=True)
        cellules[1].markdown(f'<div class="cellule">{date_locale(element["date"])}</div>',
                             unsafe_allow_html=True)
        cellules[2].markdown(badge(classe_element["court"], classe_element["ton"]), unsafe_allow_html=True)
        cellules[3].markdown(badge(retour, ton_retour), unsafe_allow_html=True)
        cellules[4].markdown(badge(observee, ton_observee), unsafe_allow_html=True)
        with cellules[5].container(key=f"lien_{element['id_prediction']}"):
            st.button("Consulter →", on_click=consulter, args=(element,),
                      key=f"consulter_{element['id_prediction']}")
        html('<hr class="fin">')

    note, pagination = st.columns([3, 1.4], vertical_alignment="center")
    note.markdown('<p class="aide-saisie" style="margin-top:0.6rem">La prédiction initiale, '
                  'l\'appréciation du conseiller et la situation observée restent consultables '
                  'séparément.</p>',
                  unsafe_allow_html=True)
    precedent, numero, suivant = pagination.columns([1, 1.6, 1], vertical_alignment="center")
    if precedent.button("‹", disabled=etat.page_historique == 0):
        etat.page_historique -= 1
        st.rerun()
    fin = min(debut + LIGNES_PAR_PAGE, len(lignes))
    numero.markdown(f'<p class="discret">{debut + 1 if lignes else 0}–{fin} sur {len(lignes)}</p>',
                    unsafe_allow_html=True)
    if suivant.button("›", disabled=etat.page_historique >= nombre_pages - 1):
        etat.page_historique += 1
        st.rerun()


# --- Suivi du modèle (équipe data) --------------------------------------------------------------

def pourcentage(valeur):
    return "—" if valeur is None else f"{valeur:.0%}"


def page_suivi():
    prototype()
    st.title("Suivi du modèle")
    html('<div class="sous-titre">Le modèle vieillit-il ? Service, dérive des données, performance réelle, '
         'réentraînements.</div>')
    info("Réservé à l'équipe data (en production : accès par rôle). Rapport produit chaque jour par "
         "<b>scripts/suivi.py</b>, à côté du journal et des données d'entraînement.")
    try:
        rapport = client.suivi()
    except client.ErreurAPI as erreur:
        st.error(erreur.message)
        return
    if rapport is None:
        st.warning("Aucun rapport de suivi pour l'instant : lancer « python scripts/suivi.py --reference "
                   "data/raw/dataset_trajectoire_emploi.csv ».")
        return
    indicateurs = rapport["indicateurs"]
    html(f'<p class="aide-saisie">Rapport du {date_locale(rapport["date"], avec_a=True)} · '
         f'{rapport["periode"]["jours"]} derniers jours</p>')

    # Indicateurs clés
    service, accord = indicateurs["service"], indicateurs["accord_conseillers"]
    performance = indicateurs["performance_observee"]
    tuiles = st.columns(4)
    tuiles[0].metric("Prédictions", indicateurs["predictions"]["nombre"])
    tuiles[1].metric("Latence (95e centile)",
                     f"{service['latence']['p95_ms']:.0f} ms" if service.get("latence") else "—")
    tuiles[2].metric("Accord conseillers / modèle", pourcentage(accord.get("taux_accord")))
    tuiles[3].metric("Erreurs critiques observées", pourcentage(performance.get("Erreurs critiques (2→0)")))

    # Alertes
    alertes = alertes_triees(rapport)
    st.subheader(f"Alertes ({len(alertes)})")
    if not alertes:
        st.success("Rien à signaler sur la période.")
    for alerte in alertes:
        ton = "ambre" if alerte["niveau"] == "critique" else "bleu"
        html(f'<p>{badge(alerte["niveau"].capitalize(), ton)} <b>{escape(alerte["indicateur"])}</b> — '
             f'{escape(alerte["message"])}</p>')

    # Dérive des données (PSI, KS), comme le tableau de bord du module M6
    lignes = rapport.get("derive_par_semaine") or []
    st.subheader("Dérive des données par rapport à l'entraînement")
    derive = indicateurs.get("derive_des_donnees", {})
    if "statut" in derive:
        st.info(f"Dérive {derive['statut']}.")
    else:
        st.dataframe(pd.DataFrame([{"Variable": NOMS_VARIABLES.get(v, v), "PSI": r.get("psi"),
                                    "Dérive": r.get("derive", r.get("statut")),
                                    "KS (p-valeur)": r.get("ks_p_valeur")} for v, r in derive.items()]),
                     hide_index=True, width="stretch")
        html('<p class="aide-saisie">PSI : &lt; 0,10 stable ; 0,10 à 0,25 dérive modérée ; ≥ 0,25 dérive '
             'forte (alerte). Le KS indique la significativité ; c\'est le PSI (ampleur de l\'écart) qui '
             'décide.</p>')
    if lignes:
        st.markdown("**Variables en dérive par semaine (niveau global)**")
        st.bar_chart(semaines_en_derive(lignes), color="#000091")
        st.markdown("**Carte de chaleur des écarts par métier (PSI, semaine par semaine)**")
        variables = sorted({ligne["variable"] for ligne in lignes}, key=lambda v: NOMS_VARIABLES.get(v, v))
        variable = st.selectbox("Variable", variables, format_func=lambda v: NOMS_VARIABLES.get(v, v),
                                key="suivi_variable")
        carte, styles = carte_par_metier(lignes, variable)
        if carte.empty:
            st.info("Pas assez d'usagers par métier et par semaine pour cette variable.")
        else:
            st.dataframe(carte.style.apply(lambda _: styles, axis=None).format("{:.2f}", na_rep="—"),
                         width="stretch")
            html('<p class="aide-saisie">Rouge : dérive retenue (PSI ≥ 0,35 ET KS significatif) ; orangé : '
                 'PSI ≥ 0,10, à surveiller ; « — » : moins de 50 usagers dans la cellule. Un écart localisé '
                 'sur un métier peut disparaître dans la moyenne globale.</p>')

    # Étiquettes et réentraînements
    st.subheader("Situations observées et réentraînements")
    gauche, droite = st.columns(2)
    with gauche:
        st.markdown(f"**{performance['situations_observees']} situations observées**")
        for source, nombre in (performance.get("par_source") or {}).items():
            st.markdown(f"- {escape(source)} : {nombre}")
        couverture = performance.get("couverture_par_classe_predite") or {}
        st.markdown("Part des prédictions ayant reçu une situation observée : " + " · ".join(
            f"{CLASSES[int(c)]['court']} {pourcentage(v)}" for c, v in couverture.items()))
    with droite:
        reentrainements = indicateurs["reentrainements"]
        st.markdown(f"**{reentrainements['nombre']} réentraînement(s)**")
        for statut, nombre in reentrainements["statuts"].items():
            st.markdown(f"- {escape(statut.replace('_', ' '))} : {nombre}")


# --- Aide ---------------------------------------------------------------------------------------

def page_aide():
    prototype()
    st.title("Aide")
    st.markdown("""
**À quoi sert PHARE ?** À estimer, au premier entretien, le délai probable de retour à l'emploi d'un usager,
pour aider le conseiller à choisir le bon niveau d'accompagnement. **La décision reste celle du conseiller.**

**Les trois estimations**
- **Retour rapide** : moins de 6 mois.
- **Délai moyen** : 6 à 12 mois.
- **Risque de longue durée** : plus de 12 mois → accompagnement renforcé à examiner.

**Le risque en %** est la probabilité estimée de chômage de longue durée, recalibrée pour être lisible.
C'est une estimation, pas une certitude.

**« Pourquoi cette recommandation ? »** montre, pour cet usager, quelles informations ont rapproché ou
éloigné le modèle de sa recommandation (méthode SHAP), et les mots de la synthèse qui ont le plus pesé.
Cela décrit le calcul du modèle, pas la cause de la situation de l'usager : c'est un appui pour
discuter de l'estimation, et pour la corriger si un élément important n'a pas été pris en compte.

**Informations demandées** : âge, diplôme, ancienneté, métier visé et synthèse de l'entretien.
La nationalité, la commune et le statut d'allocataire ne sont **pas** demandés : le modèle ne les utilise pas
(minimisation des données, non-discrimination).

**Deux retours différents**
- **Votre appréciation** (« confirmer » ou « proposer une correction ») : donnée au moment de l'entretien,
  conservée pour suivre l'accord entre conseillers et modèle. Elle ne sert **pas** à réentraîner le modèle.
- **La situation observée** : le délai réellement constaté, renseigné plus tard depuis l'historique,
  de préférence en interrogeant le référentiel des usagers (dates d'inscription et de reprise d'emploi).
  C'est la seule information utilisée pour réentraîner le modèle.

**Suivi du modèle** (équipe data) : alertes du service, dérive des données par rapport à l'entraînement
(PSI et test de Kolmogorov-Smirnov, semaine par semaine et par métier), performance réelle sur les
situations observées, réentraînements. Rapport produit chaque jour par `scripts/suivi.py`.

**Prototype** : les données affichées sont fictives ; ne saisissez jamais de nom ni de coordonnées.
""")


# --- Routage ------------------------------------------------------------------------------------

entete()
{"analyse": page_analyse, "resultat": page_resultat, "appreciation": page_appreciation,
 "historique": page_historique, "suivi": page_suivi, "aide": page_aide}[etat.page]()
pied()
