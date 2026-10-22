"""cadastre-ia — numérisation et contrôle des titres fonciers sénégalais.

Cinq modules, et une règle de dépendance unique : `api` → modules métier → `core`,
jamais l'inverse. Aucun module métier n'importe `api` ni `storage` ; le dépôt de
données leur est injecté, pour que remplacer JSON par PostGIS ne touche rien d'autre.

    core        le contrat partagé : schéma d'extraction, réglages, normalisation
    document    acquisition : image, PDF, prétraitement, qualité      (module 1)
    extraction  les deux lecteurs, leur fusion, l'orchestration       (module 2)
    validation  les règles métier et le verdict                       (module 3)
    geospatial  polygones, superficies, chevauchements                (module 4)
    review      vérité terrain, décisions humaines, historique        (module 5)
    storage     persistance, derrière une interface
"""
