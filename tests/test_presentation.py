"""Règles d'affichage de l'interface PHARE (sans lancer Streamlit)."""

from datetime import date, datetime, timezone

import pytest

from app.presentation import (
    AGES,
    ANCIENNETES,
    DIPLOMES,
    age_au,
    ajouter_mois,
    charger_referentiel,
    date_locale,
    date_observation_demo,
    filtrer,
    lecture_explication,
    libelle_metier,
    libelle_rome,
    premier_entretien_valide,
    retour_conseiller,
    situation_depuis_referentiel,
    situation_observee,
    texte_regle,
    usager_pour_api,
)


def element(id_prediction, date, classe, id_usager=None, avis=None, observee=None, statut="ok"):
    return {"id_prediction": id_prediction, "date": date, "statut": statut, "id_usager": id_usager,
            "sorties": {"classe": classe} if statut == "ok" else None,
            "avis_conseiller": avis, "situation_observee": observee}


def test_libelles():
    assert libelle_rome(" m1607 ") == "M1607 · Support à l'entreprise"
    assert date_locale("2026-10-04T12:32:00.000+00:00") == "04/10/2026 · 14:32"   # heure de Paris
    assert date_locale("2026-10-04T12:32:00+00:00", avec_a=True) == "04/10/2026 à 14:32"


def test_retour_du_conseiller_et_situation_observee():
    assert retour_conseiller(element("a", "d", 2)) == ("À examiner", "gris")
    confirme = retour_conseiller(element("a", "d", 2, avis={"avis": "confirme", "classe_proposee": 2}))
    assert confirme[0] == "Confirmée"
    corrige = retour_conseiller(element("a", "d", 2, avis={"avis": "corrige", "classe_proposee": 1}))
    assert corrige[0].startswith("Corrigée") and "6–12 mois" in corrige[0]
    assert situation_observee(element("a", "d", 2))[0] == "Non connue"
    assert situation_observee(element("a", "d", 2, observee={"classe_reelle": 0}))[0] == "Rapide · < 6 mois"


def test_saisie_vers_api():
    usager = usager_pour_api(None, "Non renseigné", 4.5, " m1607 ", "  Projet clair. ")
    assert usager == {"age": None, "niveau_diplome": None, "anciennete_poste_ans": 4.5,
                      "code_rome_vise": "M1607", "synthese_entretien": "Projet clair."}


def test_filtres_de_l_historique():
    maintenant = datetime(2026, 10, 4, 18, 0, tzinfo=timezone.utc)
    historique = [
        element("p1", "2026-10-04T10:00:00+00:00", 2, "DE-0248",
                avis={"avis": "corrige", "classe_proposee": 1}),
        element("p2", "2026-10-02T10:00:00+00:00", 1, "DE-0247",
                avis={"avis": "confirme", "classe_proposee": 1}),
        element("p3", "2026-09-20T10:00:00+00:00", 0, "DE-0246"),
        element("p4", "2026-10-04T11:00:00+00:00", None, statut="invalide"),   # jamais affichée
    ]

    def ids(**filtres):
        return [e["id_prediction"] for e in filtrer(historique, maintenant=maintenant, **filtres)]

    assert ids() == ["p1", "p2", "p3"]
    assert ids(identifiant="0247") == ["p2"]
    assert ids(periode="Aujourd'hui") == ["p1"]
    assert ids(periode="Ce mois-ci") == ["p1", "p2"]
    assert ids(classe=0) == ["p3"]
    assert ids(statut="Corrigée") == ["p1"] and ids(statut="À examiner") == ["p3"]


def test_libelle_metier():
    assert libelle_metier(" m1607 ") == "M1607 · Secrétariat"            # métier de la liste fermée
    assert libelle_metier("H2909") == "H2909 · Industrie"                # hors liste : grand domaine seul


def test_age_calcule_depuis_la_date_de_naissance():
    assert age_au("1990-10-05", date(2026, 10, 4)) == 35                 # veille de l'anniversaire
    assert age_au("1990-10-04", date(2026, 10, 4)) == 36


def test_ajouter_mois():
    assert ajouter_mois(date(2026, 1, 1), 6) == date(2026, 7, 1)
    assert ajouter_mois(date(2026, 8, 31), 6) == date(2027, 2, 28)       # fin de mois
    assert ajouter_mois(date(2026, 1, 15), 12) == date(2027, 1, 15)


def test_situation_depuis_referentiel_en_mois_calendaires():
    jour = date(2027, 6, 1)

    def classe(reprise, inscription="2026-01-01", jour=jour):
        return situation_depuis_referentiel({"date_inscription": inscription, "date_reprise_emploi": reprise},
                                            jour)

    # Bornes exactes (relecture B9) : 6 mois pile → classe 1 ; 12 mois pile → classe 1 ; au-delà → 2
    assert classe("2026-06-30") == 0
    assert classe("2026-07-01") == 1
    assert classe("2027-01-01") == 1
    assert classe("2027-01-02") == 2
    # Sans reprise : en recherche depuis plus de 12 mois → 2 ; sinon pas encore connue
    assert classe(None) == 2
    assert classe(None, jour=date(2026, 12, 31)) is None
    # Reprise FUTURE à la date d'observation : pas encore constatée
    assert classe("2027-02-01", jour=date(2026, 10, 4)) is None
    # Dates incohérentes refusées
    with pytest.raises(ValueError):
        classe("2025-12-01")


def test_premier_entretien_et_chronologie():
    dossier = {"date_inscription": "2026-10-01", "date_reprise_emploi": "2026-12-15"}
    assert premier_entretien_valide(dossier, "2026-10-04T09:00:00+00:00")
    assert not premier_entretien_valide(dossier, "2026-12-20T09:00:00+00:00")     # après la reprise
    assert not premier_entretien_valide(dossier, "2026-09-20T09:00:00+00:00")     # avant l'inscription
    assert not premier_entretien_valide({**dossier, "date_reprise_emploi": None},
                                        "2026-11-15T09:00:00+00:00")             # hors premier entretien


def test_referentiel_fictif_compatible_avec_le_formulaire():
    jour = date(2026, 10, 4)
    referentiel = charger_referentiel(jour=jour)
    assert len(referentiel) == 20
    observees = set()
    for identifiant, dossier in referentiel.items():
        assert dossier["id_usager"] == identifiant
        assert age_au(dossier["date_naissance"], jour) in AGES
        assert (dossier["niveau_diplome"] or "Non renseigné") in DIPLOMES
        assert dossier["anciennete_poste_ans"] is None or dossier["anciennete_poste_ans"] in ANCIENNETES
        assert "code_rome_vise" not in dossier          # le métier visé se décide pendant l'entretien
        # Chronologie : tous les dossiers sont au premier entretien aujourd'hui, sans reprise passée
        assert premier_entretien_valide(dossier, f"{jour.isoformat()}T09:00:00+00:00")
        observees.add(situation_depuis_referentiel(dossier, date_observation_demo(jour)))
    assert observees == {0, 1, 2}                       # la démonstration couvre les trois issues


def test_texte_de_la_regle():
    regle = {"motif": "garde_fou", "classe_la_plus_probable": 0, "risque_recalibre": 0.1016,
             "seuil_classe_2": 0.25, "seuil_garde_fou": 0.08}
    assert "garde-fou de 8%" in texte_regle(regle) and "10%" in texte_regle(regle)
    assert "seuil de 25%" in texte_regle({**regle, "motif": "seuil_classe_2", "risque_recalibre": 0.4})
    assert "Avant recalibration" in texte_regle({**regle, "motif": "plus_probable_0_1",
                                                 "classe_la_plus_probable": 2})
    assert texte_regle(None) is None


def test_lecture_de_l_explication():
    explication = {"classe_expliquee": 2, "valeur_de_base": -1.1,
                   "facteurs": [{"facteur": "Synthèse de l'entretien", "contribution": 0.8},
                                {"facteur": "Âge", "contribution": -0.4},
                                {"facteur": "Niveau de diplôme", "contribution": 0.001}],
                   "mots": [{"mot": "permis", "contribution": 0.3},
                            {"mot": "mobilité", "contribution": -0.2}]}
    lecture = lecture_explication(explication)
    assert [(ligne["sens"], ligne["largeur"]) for ligne in lecture["lignes"]] == \
        [("pour", 100), ("contre", 50), ("neutre", 0)]
    assert lecture["mots_pour"] == ["permis"] and lecture["mots_contre"] == ["mobilité"]
    assert lecture_explication(None) is None                 # prédiction d'avant l'ajout de SHAP
