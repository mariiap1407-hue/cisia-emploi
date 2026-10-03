"""Éléments partagés par les tests."""

import subprocess
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def donnees_factices():
    """Génère les jeux factices s'ils n'existent pas encore, et renvoie leur dossier."""
    dossier = RACINE / "data" / "factice"
    if not (dossier / "entrainement.csv").exists():
        subprocess.run([sys.executable, "scripts/generer_donnees_factices.py"], cwd=RACINE, check=True)
    return dossier
