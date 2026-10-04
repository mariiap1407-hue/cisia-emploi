# Interface conseiller PHARE (Streamlit). Elle n'appelle que l'API : ni modèle, ni données, ni clé dans
# l'image. Adresse de l'API (CISIA_API_URL) et clé (CISIA_CLE_API) données au lancement.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements-ui.txt .
RUN pip install -r requirements-ui.txt

COPY .streamlit/ .streamlit/
COPY app/ app/

RUN useradd --create-home --uid 1000 phare
USER phare

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)" \
    || exit 1

CMD ["streamlit", "run", "app/interface.py", "--server.address", "0.0.0.0", "--server.port", "8501", \
     "--server.headless", "true"]
