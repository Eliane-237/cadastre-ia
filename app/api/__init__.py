"""La couche HTTP. Elle ne décide rien : elle expose ce que les modules métier décident.

Aucun module de `app/` en dehors d'ici n'importe FastAPI, et aucun routeur ne contient
de règle métier. C'est ce qui permet d'avoir, aujourd'hui, un CLI et une API qui font
exactement la même chose — et qui ne peuvent pas diverger.
"""
