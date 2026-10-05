# CISIA — Aide à l'orientation des demandeurs d'emploi

Prototype réalisé pour l'examen de certification Simplon. Au premier entretien, le service estime le
délai probable de retour à l'emploi d'un usager et recommande un niveau d'accompagnement :

| Classe | Délai estimé | Recommandation |
|---|---|---|
| 0 | moins de 6 mois | accompagnement léger |
| 1 | 6 à 12 mois | accompagnement standard |
| 2 | plus de 12 mois | accompagnement renforcé à examiner |

**La décision reste celle du conseiller.** Le modèle (LightGBM + TF-IDF sur la synthèse d'entretien,
probabilité de la classe 2 recalibrée, règle de décision explicite) n'utilise que l'âge, le diplôme, l'ancienneté, le
métier visé et la synthèse de l'entretien : ni nationalité, ni commune, ni statut d'allocataire.

L'étude complète (données, choix du modèle, évaluation, éthique, architecture) est dans le **notebook**,
transmis à part. Ce dépôt contient le code d'industrialisation : entraînement, API, interface, Docker,
CI/CD et suivi en production.

> Les vraies données (personnelles) ne sont **jamais** versionnées. La CI travaille sur des données
> **factices** générées par un script ; le modèle réel est entraîné sur le poste, à côté des données.

## Démonstration en ligne

| | Adresse |
|---|---|
| Interface conseiller PHARE | https://phare-demo.onrender.com (accès par mot de passe, communiqué à part) |
| API (documentation) | https://cisia-api.onrender.com/docs |

Démonstration sur Render (offre gratuite, région Francfort) : **modèle de contrôle entraîné sur des
données factices**, journal éphémère, données fictives uniquement. Les services s'endorment après
15 minutes sans visite : le premier accès peut prendre environ une minute, et la première analyse peut
échouer le temps que l'API se réveille (il suffit de recommencer).

---

## Démarrage rapide

### Prérequis et installation
Python 3.12 (environnement conda `exam`) ; Docker Desktop pour la pile complète (facultatif).
```bash
pip install -r requirements-dev.txt
pip install -e .
```
Créer un fichier `.env` à la racine (modèle : `.env.exemple` ; jamais versionné) :
```
CISIA_CLE_API=<une clé longue et aléatoire>
```

### Parcours A — démonstration sur données factices (depuis un dépôt vierge)
Aucune donnée réelle n'est nécessaire. Le modèle obtenu ne vaut rien en conditions réelles : il sert à
faire tourner la chaîne (c'est ce que fait la CI).
```bash
python scripts/generer_donnees_factices.py                       # crée data/factice/*.csv
python scripts/entrainer.py --donnees data/factice/entrainement.csv --sans-recherche
python scripts/preparer_image_ci.py models/candidats/<run_id> --sortie models/ci   # <run_id> affiché par l'entraînement
```
Puis lancer l'API sur ce modèle de contrôle, en ajoutant au `.env` : `CISIA_PRODUCTION=models/ci`.
Un modèle entraîné sur des données factices ne peut jamais être mis dans `models/production` (refusé
par les scripts).

### Parcours B — données réelles (poste autorisé uniquement)
Le fichier de données (`data/raw/dataset_trajectoire_emploi.csv`) n'est pas dans le dépôt.
```bash
python scripts/entrainer.py --promouvoir      # mise en production dans models/production si le quality gate passe
```

### Lancer l'application sur le poste
```bash
uvicorn api.main:app --reload                 # API : http://127.0.0.1:8000/docs
streamlit run app/interface.py                # interface conseiller PHARE : http://127.0.0.1:8501
mlflow ui                                     # historique des entraînements (base mlflow.db du poste)
```
Le journal des prédictions est `outputs/cisia.db`.

### Lancer la pile Docker (API, interface, serveur MLflow)
```bash
docker compose up -d --build --wait
```
Arrêter d'abord `uvicorn` et `streamlit` lancés sur le poste : ils utilisent les mêmes ports (8000, 8501),
et le conteneur concerné resterait à l'état « Created ». L'image de l'API embarque le modèle de `models/production` (parcours B). Différences avec le poste :
- le journal est dans le **volume Docker `journal`** (pas dans `outputs/cisia.db` du poste) ;
- le serveur MLflow de la pile (http://127.0.0.1:5000) a **sa propre base** (`./mlflow-data`), distincte
  de `mlflow.db` du poste. Pour y enregistrer un entraînement : `MLFLOW_TRACKING_URI=http://localhost:5000`.

### Suivi du modèle
Sur le poste (journal `outputs/cisia.db`, version en service lue dans `models/production`) :
```bash
python scripts/suivi.py --reference data/raw/dataset_trajectoire_emploi.csv
```
Le rapport (`outputs/suivi/dernier_rapport.json`) est servi par `/suivi` et affiché dans la page « Suivi ».
Code de sortie : 0 = aucune alerte, 1 = alerte « attention », 2 = alerte « critique ».

Sous Docker, le script n'est pas dans l'image : copier le journal sur le poste, produire le rapport, puis
le remettre dans le volume (procédure manuelle du prototype, vérifiée le 05/10) :
```bash
docker cp cisia-api:/app/outputs/cisia.db outputs/journal_docker.db
python scripts/suivi.py --base outputs/journal_docker.db --sortie outputs/suivi_docker --production models/production
docker cp outputs/suivi_docker/. cisia-api:/app/outputs/suivi/
```

Démonstration sur des journaux factices (service sain / service dégradé) :
```bash
python scripts/generer_journal_factice.py
python scripts/suivi.py --base outputs/suivi_demo/journal_degrade.db --reference data/factice/entrainement.csv
```

### Tests
```bash
ruff check .
pytest            # génère les données factices si besoin
```

---

## Routes de l'API

Les routes métier exigent l'en-tête `X-API-Key` ; `/health` et la documentation (`/docs`,
`/openapi.json`) restent accessibles sans clé.

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
2. **build-test-push** : images de l'API, de l'interface et de MLflow construites ; l'API est **testée avant
   publication** (santé, prédiction, sécurité) et l'interface testée au démarrage ; images de l'API et de
   l'interface publiées sur `ghcr.io` (l'image MLflow est seulement construite) ;
3. **deploy-demo** : déploiement de l'API puis de l'interface de démonstration sur Render (région
   Francfort), avec vérification.

Démonstration d'échec du quality gate : *Actions > CI/CD > Run workflow* avec
`reference = data/factice/reference_derivee.csv` → le gate échoue, rien n'est construit ni déployé.

Une modification qui ne touche que la documentation peut être poussée sans relancer la CI, avec
`[skip ci]` dans le message du commit.

---

## Limites connues

- **Données d'entraînement peu variées** : les synthèses font toutes 63 à 77 caractères ; le modèle
  reconnaît des formulations plus qu'il ne comprend un texte libre (la négation n'est pas comprise).
- **Démonstration** : l'API déployée sur Render utilise un modèle **factice**, avec un journal éphémère ;
  `/retrain` y répond 503 (pas de données dans l'image). Le modèle réel reste sur le poste. La CI vérifie
  que l'interface répond après son déploiement, mais pas quelle version est servie (l'interface
  n'expose pas de numéro de version, contrairement à l'API).
- **Sécurité** : une clé d'API unique et partagée ; en production, authentification par l'annuaire
  (SSO) et droits par rôle.
- **Suivi** : rapport lancé à la main (planification prévue en production) ; seuils de départ à
  recalibrer sur du trafic réel ; profil de référence recalculé depuis les données d'entraînement.
- **Historique MLflow propre au poste** : les runs enregistrés dans `mlflow.db` référencent des chemins
  d'artefacts absolus (Windows) ; copier la base sur une autre machine ne suffit pas pour retrouver les
  modèles. Sur une autre machine, on recrée de nouveaux runs (parcours A ou B) ; la restauration de
  l'historique existant n'est pas prise en charge (V2 : un serveur MLflow partagé).
- **Outillage** : actions GitHub épinglées par version majeure ; dépendances indirectes de l'image
  MLflow non figées ; avertissement MLflow sur les annotations de type de `predict` (non traité :
  l'ajout activerait la validation des entrées par MLflow et modifierait le modèle sérialisé).
- **Prototype** : interface sans charte de l'État (DSFR) ni audit d'accessibilité ; référentiel des
  usagers fictif.

Les améliorations envisagées (V2) sont détaillées dans le notebook.
