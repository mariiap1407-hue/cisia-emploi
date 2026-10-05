"""Parcours principal de l'interface PHARE, de bout en bout (relecture B9).

L'interface Streamlit tourne avec AppTest (sans navigateur) et parle à la VRAIE application FastAPI
(TestClient), avec un modèle de contrôle et un journal temporaires :
analyse d'un dossier → résultat expliqué → correction du conseiller → historique → réouverture avec le
motif et les précisions → situation observée lue dans le référentiel fictif.
Ce test vérifie le fonctionnement, pas l'apparence (pas de rendu dans un navigateur).
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import creer_application
from tests.test_api import CLE, production  # noqa: F401  (fixture : modèle de contrôle en production)

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_APP = RACINE / "app"


def texte(at):
    """Tout le texte affiché (markdown et messages), pour chercher une phrase."""
    elements = [*at.markdown, *at.success, *at.error, *at.info, *at.subheader, *at.title]
    return "\n".join(str(e.value) for e in elements)


def bouton(at, libelle):
    return next(b for b in at.button if b.label == libelle)


@pytest.fixture
def interface(production, tmp_path, monkeypatch):  # noqa: F811
    """AppTest branché sur l'API réelle : client.appeler() passe par TestClient au lieu du réseau."""
    monkeypatch.syspath_prepend(str(DOSSIER_APP))      # l'interface importe « client » et « presentation »
    import client as module_client

    application = creer_application(production, tmp_path / "journal.db", cle_api=CLE,
                                    dossier_suivi=tmp_path / "suivi")
    with TestClient(application) as api:
        def appeler(methode, chemin, corps=None, delai=30):
            reponse = api.request(methode, chemin, json=corps, headers={"X-API-Key": CLE})
            if reponse.status_code >= 400:
                detail = reponse.json().get("detail")
                raise module_client.ErreurAPI(reponse.status_code, str(detail))
            return reponse.json()

        monkeypatch.setattr(module_client, "appeler", appeler)
        at = AppTest.from_file(str(DOSSIER_APP / "interface.py"), default_timeout=60)
        at.session_state["intro_vue"] = True             # pas d'animation d'accueil dans le test
        yield at, api


def test_parcours_complet_du_conseiller(interface):
    at, api = interface
    at.run()
    assert not at.exception

    # 1. Analyse : dossier du référentiel, métier et synthèse saisis pendant l'entretien
    at.selectbox(key="dossier_choisi").set_value("DE-0003").run()
    next(s for s in at.selectbox if s.label.startswith("Métier visé")).set_value("M1607")
    next(t for t in at.text_area if t.label.startswith("Synthèse")).input(
        "Projet : secrétariat. Expériences : accueil. Freins : mobilité réduite.")
    bouton(at, "Analyser la situation →").click().run()
    assert not at.exception
    assert at.session_state["page"] == "resultat"
    courant = at.session_state["courant"]
    assert courant["id_usager"] == "DE-0003"
    affiche = texte(at)
    assert "Pourquoi cette recommandation" in affiche and "Règle de décision" in affiche

    # 2. Correction du conseiller, avec motif et précisions
    bouton(at, "Proposer une correction").click().run()
    assert at.session_state["page"] == "appreciation"
    autre_classe = (courant["sorties"]["classe"] + 1) % 3
    at.radio[0].set_value(autre_classe)
    next(s for s in at.selectbox if s.label.startswith("Motif")).set_value("Autre")
    next(t for t in at.text_area if t.label == "Précisions").input("Formation qualifiante en cours")
    bouton(at, "Enregistrer mon retour →").click().run()
    assert not at.exception
    assert at.session_state["page"] == "resultat"
    assert "Précisions : Formation qualifiante en cours" in texte(at)

    # 3. Historique, puis réouverture : motif et précisions relus depuis l'API
    at.button(key="bouton_nav_historique").click().run()
    assert "DE-0003" in texte(at)
    at.button(key=f"consulter_{courant['id_prediction']}").click().run()
    assert not at.exception
    affiche = texte(at)
    assert "Motif : Autre" in affiche and "Précisions : Formation qualifiante en cours" in affiche

    # 4. Situation observée lue dans le référentiel fictif (DE-0003 : délai moyen), puis enregistrée
    at.button(key=f"referentiel_{courant['id_prediction']}").click().run()
    assert "Moyen" in texte(at)
    at.button(key=f"enreg_ref_{courant['id_prediction']}").click().run()
    assert not at.exception
    element = api.get("/history?limite=1", headers={"X-API-Key": CLE}).json()[0]
    assert element["id_usager"] == "DE-0003"
    assert element["situation_observee"]["classe_reelle"] == 1
    assert element["avis_conseiller"]["precisions"] == "Formation qualifiante en cours"


def test_page_suivi_du_modele(interface, tmp_path, donnees_factices):
    """Page « Suivi » : rapport RÉEL du suivi (journal factice dégradé, référence factice), via /suivi."""
    import importlib.util
    import json
    from datetime import datetime, timezone

    from api.journal import Journal
    from cisia.preparation import charger_donnees, decouper
    from cisia.suivi import analyser

    at, _ = interface
    at.run()
    at.button(key="bouton_nav_suivi").click().run()
    assert not at.exception
    assert any("Aucun rapport" in str(w.value) for w in at.warning)          # avant le premier rapport

    script = RACINE / "scripts" / "generer_journal_factice.py"
    spec = importlib.util.spec_from_file_location("generateur", script)
    generateur = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generateur)
    journal = tmp_path / "journal_degrade.db"
    generateur.remplir(Journal(journal), "degrade", generateur.ancrage_semaine(datetime.now(timezone.utc)),
                       profils=generateur.profils_factices(), nombre=2100, jours=21, nombre_mures=400)
    entrainement, _ = decouper(charger_donnees(donnees_factices / "entrainement.csv"))
    seuils = json.loads((RACINE / "config" / "seuils_suivi.json").read_text(encoding="utf-8"))
    rapport = analyser(journal, seuils, reference=entrainement[generateur.COLONNES].to_dict("records"))
    (tmp_path / "suivi").mkdir()
    (tmp_path / "suivi" / "dernier_rapport.json").write_text(json.dumps(rapport), encoding="utf-8")

    at.run()
    assert not at.exception
    affiche = texte(at)
    assert "Alertes" in affiche and "registre MLflow" in affiche
    assert "risque de biais des étiquettes" in affiche
    assert "Cohortes mûres" in affiche and "None" not in affiche
    assert "Performance réelle par version (en service" in affiche              # attribution par version
    assert len(at.dataframe) >= 2                                            # tableau PSI + carte par métier
    # La carte s'ouvre sur la variable de l'alerte « dérive localisée » (l'âge dans ce scénario)
    assert at.selectbox(key="suivi_variable").value == "age"
    at.selectbox(key="suivi_variable").set_value("anciennete_poste_ans").run()
    assert not at.exception
