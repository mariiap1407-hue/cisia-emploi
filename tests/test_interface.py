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

    with TestClient(creer_application(production, tmp_path / "journal.db", cle_api=CLE)) as api:
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
