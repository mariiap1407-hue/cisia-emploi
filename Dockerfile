# Image de l'API CISIA : code, règle de décision et modèle EN SERVICE, rien d'autre.
# Pas de données (ni brutes, ni journaux), pas de clé : la clé est donnée au lancement
# (variable d'environnement CISIA_CLE_API), jamais écrite dans l'image.
#
# Construction (depuis la racine du projet) : docker compose build
# Le modèle copié est celui de models/production/ AU MOMENT DE LA CONSTRUCTION (actuelle.json
# désigne la version en service) : reconstruire l'image après chaque mise en production.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# LightGBM a besoin de la bibliothèque OpenMP (libgomp), absente de l'image « slim »
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 1. Dépendances (couche réutilisée tant que requirements.txt ne change pas : constructions plus rapides)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2. Code partagé (module cisia), API, règle de décision, modèle en service
COPY pyproject.toml .
COPY src/ src/
# (le dossier build/ et les métadonnées créés par l'installation sont supprimés : inutiles à l'exécution)
RUN pip install --no-deps . \
    && rm -rf build src/*.egg-info
COPY api/ api/
COPY config/ config/
COPY models/production/ models/production/

# 3. Utilisateur sans droits d'administration ; seul le dossier du journal est modifiable
RUN useradd --create-home --uid 1000 cisia \
    && mkdir -p /app/outputs \
    && chown cisia:cisia /app/outputs
USER cisia

EXPOSE 8000

# Disponibilité : /health répond 200 seulement si le modèle est chargé ET une clé configurée
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" \
    || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
