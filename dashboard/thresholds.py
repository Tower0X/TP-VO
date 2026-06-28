"""
dashboard/thresholds.py — Gestion des Seuils de Congestion
===========================================================
Groupe 8 : Estimation de la Densité de Trafic Urbain
"""

from config import inference_cfg

def get_congestion_info(count: float) -> dict:
    """
    Détermine le statut de congestion et les éléments d'interface associés
    (couleur, icône, message).
    
    Args:
        count: Nombre de véhicules estimés.
        
    Returns:
        dict: Informations de style et de statut pour l'UI.
    """
    if count < inference_cfg.threshold_fluid:
        return {
            "status": "Fluide",
            "color": "#00FF00", # Vert
            "hex": "green",
            "icon": "🟢",
            "message": "Circulation normale, aucun ralentissement détecté."
        }
    elif count < inference_cfg.threshold_moderate:
        return {
            "status": "Modéré",
            "color": "#FFA500", # Orange
            "hex": "orange",
            "icon": "🟡",
            "message": "Densité moyenne, léger ralentissement possible."
        }
    else:
        return {
            "status": "Congestion",
            "color": "#FF0000", # Rouge
            "hex": "red",
            "icon": "🔴",
            "message": "Trafic dense ! Risque de bouchons imminent."
        }

def get_threshold_config():
    """Retourne la configuration actuelle des seuils."""
    return {
        "fluid_max": inference_cfg.threshold_fluid,
        "moderate_max": inference_cfg.threshold_moderate
    }
