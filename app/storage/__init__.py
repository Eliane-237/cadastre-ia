"""Persistance des parcelles validées, derrière une interface.

Aujourd'hui un fichier JSON par document ; demain PostGIS. Les modules métier ne
connaissent que `ParcelRepository` : changer d'implémentation ne doit toucher ni la
géométrie, ni la validation, ni l'extraction.
"""
