# CISIA — Aide à l'orientation des demandeurs d'emploi

Prototype réalisé pour l'examen de certification Simplon. Au premier entretien, le service estime le
délai probable de retour à l'emploi d'un usager et recommande un niveau d'accompagnement :

| Classe | Délai estimé | Recommandation |
|---|---|---|
| 0 | moins de 6 mois | accompagnement léger |
| 1 | 6 à 12 mois | accompagnement standard |
| 2 | plus de 12 mois | accompagnement renforcé à examiner |

**La décision reste celle du conseiller.** Le modèle (LightGBM + TF-IDF sur la synthèse d'entretien,
probabilités recalibrées, règle de décision explicite) n'utilise que l'âge, le diplôme, l'ancienneté, le
métier visé et la synthèse de l'entretien : ni nationalité, ni commune, ni statut d'allocataire.

L'étude complète (données, choix du modèle, évaluation, éthique, architecture) est dans le **notebook**,
transmis à part. Ce dépôt contient le code d'industrialisation : entraînement, API, interface, Docker,
CI/CD et suivi en production.

> Les vraies données (personnelles) ne sont **jamais** versionnées. La CI travaille sur des données
> **factices** générées par un script ; le modèle réel est entraîné sur le poste, à côté des données.

---

## Démarrage rapide

### Prérequis
- Python 3.12 (environnement conda `exam`) ;
- Docker Desktop (facultatif, pour la pile complète).

### Installation
```bash
pip install -r requirements-dev.txt
pip install -e .
```
Créer un fichier `.env` à la racine (jamais versionné) :
```
CISIA_CLE_API=<une clé de votre choix>
```

### Entraîner le modèle
```bash
python scripts/entrainer.py --promouvoir        # vraies données (data/raw/), mise en production si le quality gate passe
python scripts/entrainer.py --donnees data/factice/entrainement.csv --sans-recherche   # données factices (contrôle)
mlflow ui                                        # suivi des entraînements : http://127.0.0.1:5000
```

### Lancer l'application
```bash
uvicorn api.main:app --reload                    # API : http://127.0.0.1:8000/docs
streamlit run app/interface.py                   # interface conseiller PHARE : http://127.0.0.1:8501
```
Ou toute la pile avec Docker (API, interface, serveur MLflow) :
```bash
docker compose up -d --build --wait
```

### Suivi du modèle
```bash
python scripts/suivi.py --reference data/raw/dataset_trajectoire_emploi.csv
```
Démonstration sur des journaux factices (service sain / service dégradé) :
```bash
python scripts/generer_journal_factice.py
python scripts/suivi.py --base outputs/suivi_demo/journal_degrade.db --reference data/factice/entrainement.csv
```
Code de sortie : 0 = aucune alerte, 1 = alerte « attention », 2 = alerte « critique ». Le rapport est aussi
affiché dans la page « Suivi » de l'interface.

### Tests
```bash
ruff check .
pytest
```

---

## Routes de l'API

Toutes les routes sauf `/health` exigent l'en-tête `X-API-Key`. Documentation interactive : `/docs`.

| Route | Rôle |
|---|---|
| `GET /health` | Disponibilité : 200 seulement si un modèle est chargé et une clé configurée |
| `POST /predict` | Prédiction pour un usager : classe, recommandation, risque, explication |
| `GET /history` | Historique des prédictions, avec l'avis du conseiller et la situation observée |
| `POST /avis` | Avis du conseiller à l'entretien (confirme / corrige) : suivi seulement, jamais utilisé pour réentraîner |
| `POST /feedback` | Situation **observée** (délai réel) : seule information utilisée pour réentraîner |
| `POST /retrain` | Réentraînement avec les feedbacks, quality gate, puis champion / challenger |
| `GET /suivi` | Dernier rapport de suivi produit par `scripts/suivi.py` |

Variables d'environnement : `CISIA_CLE_API`, `CISIA_PRODUCTION`, `CISIA_BASE`, `CISIA_DONNEES`,
`CISIA_MIN_FEEDBACKS`, `CISIA_SUIVI` (détail dans `api/main.py`), `MLFLOW_TRACKING_URI`, `USE_REGISTRY`.

---

## Structure du projet

```
├── src/cisia/            code partagé : préparation, modèle, règle de décision, évaluation,
│                         explication (SHAP), modèle MLflow, suivi
├── api/                  API FastAPI : routes, schémas, journal SQLite, réentraînement
├── app/                  interface Streamlit PHARE (référentiel FICTIF de 20 dossiers)
├── scripts/              entraîner, suivre, expliquer le modèle, générer les données factices,
│                         préparer et vérifier l'image de la CI
├── config/               règle de décision (seuils) et seuils du suivi : versionnés
├── tests/                tests unitaires, API, interface, quality gate, cohérence avec le notebook
├── docker/, Dockerfile   images de l'API, de l'interface et du serveur MLflow
├── docker-compose.yml    pile locale (ports ouverts sur 127.0.0.1 seulement)
├── .github/workflows/    CI/CD
├── data/                 non versionné : vraies données dans data/raw/ ; données factices dans
│                         data/factice/ (python scripts/generer_donnees_factices.py)
├── models/               non versionné : candidats et modèle en production (actuelle.json)
└── outputs/              non versionné : journal des prédictions, rapports de suivi
```

---

## CI/CD (GitHub Actions)

À chaque push, trois jobs :
1. **test** : ruff, données factices, entraînement d'un modèle de contrôle, **quality gate**
   (erreurs critiques ≤ 10 %, rappel de la classe 2 ≥ 60 %, F1 macro ≥ 0,62), tests ;
2. **build-test-push** : images construites, **testées avant publication**, publiées sur `ghcr.io` ;
3. **deploy-demo** : déploiement de l'API de démonstration sur Render (région Francfort), puis vérification.

Démonstration d'échec du quality gate : *Actions > CI/CD > Run workflow* avec
`reference = data/factice/reference_derivee.csv` → le gate échoue, rien n'est construit ni déployé.

Une modification qui ne touche que la documentation peut être poussée sans relancer la CI, avec
`[skip ci]` dans le message du commit.

---

## Limites connues

- **Données d'entraînement peu variées** : les synthèses font toutes 63 à 77 caractères ; le modèle
  reconnaît des formulations plus qu'il ne comprend un texte libre (la négation n'est pas comprise).
- **Démonstration** : l'API déployée sur Render utilise un modèle **factice**, avec un journal éphémère ;
  `/retrain` y répond 503 (pas de données dans l'image). Le modèle réel reste sur le poste.
- **Sécurité** : une clé d'API unique et partagée ; en production, authentification par l'annuaire
  (SSO) et droits par rôle.
- **Suivi** : rapport lancé à la main (planification prévue en production) ; seuils de départ à
  recalibrer sur du trafic réel ; profil de référence recalculé depuis les données d'entraînement.
- **Outillage** : actions GitHub épinglées par version majeure ; dépendances indirectes de l'image
  MLflow non figées ; avertissement MLflow sur les annotations de type de `predict` (non traité :
  l'ajout activerait la validation des entrées par MLflow et modifierait le modèle sérialisé).
- **Prototype** : interface sans charte de l'État (DSFR) ni audit d'accessibilité ; référentiel des
  usagers fictif.

Les améliorations envisagées (V2) sont détaillées dans le notebook.
