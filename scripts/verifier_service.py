"""Vérifie qu'une API CISIA en marche répond correctement (CI : image de test, puis service déployé).

Bibliothèque standard seulement : s'exécute sur n'importe quelle machine avec Python, sans installation.

Contrôles :
1. attente active sur /health (délai maximal par requête, nombre d'essais limité) jusqu'au code 200 ;
2. si --version est donnée : la version servie doit être exactement celle-là (bon modèle déployé) ;
3. /predict avec la clé : code 200, champs attendus, même version que --version, classe dans {0, 1, 2},
   risque fini entre 0 et 1 ;
4. /predict sans clé : code 401 ;
5. si --retrain est donné : code attendu de /retrain (503 dans une image sans données).

Usage : python scripts/verifier_service.py http://127.0.0.1:8000 --cle "$CLE" [--version controle-1a2b3c4d]
        [--essais 30 --pause 2] [--retrain 503]
"""

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request

USAGER = {"age": 35, "niveau_diplome": "Bac", "anciennete_poste_ans": 4.5, "code_rome_vise": "M1607",
          "synthese_entretien": "Projet clair, mobilité possible."}
CHAMPS_PREDICTION = {"id_prediction", "classe", "recommandation", "niveau_alerte", "risque_longue_duree",
                     "risque_affiche", "version_modele"}


def appeler(methode, url, cle=None, corps=None, delai=10):
    """Renvoie (code HTTP, contenu JSON ou None). Une absence de réponse donne le code 0."""
    entetes = {"Content-Type": "application/json"}
    if cle:
        entetes["X-API-Key"] = cle
    donnees = json.dumps(corps).encode("utf-8") if corps is not None else None
    requete = urllib.request.Request(url, data=donnees, headers=entetes, method=methode)
    try:
        with urllib.request.urlopen(requete, timeout=delai) as reponse:
            return reponse.status, json.loads(reponse.read() or b"null")
    except urllib.error.HTTPError as erreur:
        return erreur.code, None
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return 0, None


def echec(message):
    print(f"ÉCHEC : {message}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", help="adresse de l'API, ex. http://127.0.0.1:8000")
    parser.add_argument("--cle", required=True, help="clé d'API (en-tête X-API-Key)")
    parser.add_argument("--version", help="version de modèle attendue sur /health")
    parser.add_argument("--essais", type=int, default=30, help="nombre d'essais sur /health")
    parser.add_argument("--pause", type=float, default=2, help="secondes entre deux essais")
    parser.add_argument("--duree-max", type=float, default=None,
                        help="durée maximale d'attente de /health, en secondes (en plus du nombre d'essais)")
    parser.add_argument("--retrain", type=int, help="code HTTP attendu pour /retrain")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    # 1-2. Disponibilité (et bonne version), avec attente active bornée : nombre d'essais ET durée
    # totale. La durée est stricte : délai de chaque requête et pause plafonnés au temps restant.
    attendue = f" avec la version {args.version}" if args.version else ""
    fin = None if args.duree_max is None else time.monotonic() + args.duree_max

    def temps_restant():
        return float("inf") if fin is None else fin - time.monotonic()

    for essai in range(1, args.essais + 1):
        if temps_restant() <= 0:
            echec(f"/health : délai maximal de {args.duree_max:g} s dépassé sans réponse 200{attendue}")
        code, sante = appeler("GET", f"{url}/health", delai=min(10, temps_restant()))
        if code == 200 and (args.version is None or (sante or {}).get("version_modele") == args.version):
            print(f"Santé 200 après {essai} essai(s) : {sante}")
            break
        print(f"Essai {essai} : code {code}, version {(sante or {}).get('version_modele')}")
        time.sleep(max(0, min(args.pause, temps_restant())))
    else:
        echec(f"/health n'a pas répondu 200{attendue} en {args.essais} essai(s)")

    # 3. Prédiction avec la clé : contenu vérifié
    code, prediction = appeler("POST", f"{url}/predict", args.cle, {"usager": USAGER, "id_session": "ci"})
    if code != 200:
        echec(f"/predict avec clé : code {code} (attendu 200)")
    manquants = CHAMPS_PREDICTION - set(prediction or {})
    if manquants:
        echec(f"/predict : champs manquants {sorted(manquants)}")
    if args.version is not None and prediction["version_modele"] != args.version:
        echec(f"/predict : version {prediction['version_modele']} (attendue {args.version})")
    risque = prediction["risque_longue_duree"]
    if prediction["classe"] not in (0, 1, 2):
        echec(f"/predict : classe {prediction['classe']} hors de {{0, 1, 2}}")
    if not (isinstance(risque, (int, float)) and math.isfinite(risque) and 0 <= risque <= 1):
        echec(f"/predict : risque {risque} hors de [0, 1]")
    print(f"Prédiction 200 : classe {prediction['classe']}, risque {risque}, "
          f"version {prediction['version_modele']}")

    # 4. Sans clé : refusé
    code, _ = appeler("POST", f"{url}/predict", corps={"usager": USAGER})
    if code != 401:
        echec(f"/predict sans clé : code {code} (attendu 401)")
    print("Sans clé : 401")

    # 5. Réentraînement (facultatif)
    if args.retrain is not None:
        code, _ = appeler("POST", f"{url}/retrain", args.cle, {}, delai=30)
        if code != args.retrain:
            echec(f"/retrain : code {code} (attendu {args.retrain})")
        print(f"Réentraînement : {code}")

    print("Service vérifié.")


if __name__ == "__main__":
    for flux in (sys.stdout, sys.stderr):
        flux.reconfigure(encoding="utf-8")
    main()
