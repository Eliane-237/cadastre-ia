"""Modèle d'extraction des titres fonciers et plans de bornage (UTM 28N).

PaddleOCR-VL-1.6 lit la page et le tableau de coordonnées, Qwen3-VL-8B remplit le JSON de
l'`Extraction` ; la fusion et la validation géométrique départagent leurs lectures.
"""
