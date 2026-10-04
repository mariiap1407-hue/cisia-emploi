# Serveur de suivi MLflow de la pile CISIA.
# MLflow est installé À LA CONSTRUCTION de l'image (et non au démarrage du conteneur) :
# démarrer ne demande ni réseau ni téléchargement, et la version installée reste la même.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install mlflow==3.16.1

EXPOSE 5000

# Base (runs, registre) et artefacts (matrices, modèles) dans /mlflow-data, monté depuis le poste
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--artifacts-destination", "/mlflow-data/artifacts"]
