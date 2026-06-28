"""
dashboard/heatmap.py — Rendu des Heatmaps Transparents
=======================================================
Groupe 8 : Estimation de la Densité de Trafic Urbain
"""

import cv2
import numpy as np
import matplotlib.cm as cm

from config import inference_cfg

def render_heatmap_overlay(
    image_bgr: np.ndarray,
    density_map: np.ndarray,
    alpha: float = None,
    colormap_name: str = None
) -> np.ndarray:
    """
    Rend une heatmap superposée à l'image avec transparence (alpha blending).
    
    Args:
        image_bgr: Image originale en format BGR (cv2)
        density_map: Tenseur 2D (H, W) des prédictions (taille == image_bgr.shape[:2])
        alpha: Facteur de transparence de la heatmap (0.0 = invisible, 1.0 = opaque)
        colormap_name: Nom de la colormap matplotlib (jet, turbo, hot, inferno, viridis)
        
    Returns:
        Image overlay en format BGR (cv2)
    """
    if alpha is None:
        alpha = inference_cfg.heatmap_alpha
        
    if colormap_name is None:
        colormap_name = inference_cfg.heatmap_colormap
        
    H, W = image_bgr.shape[:2]
    
    # Redimensionnement de sécurité si les tailles diffèrent
    if density_map.shape != (H, W):
        density_map = cv2.resize(density_map, (W, H), interpolation=cv2.INTER_CUBIC)
        
    # Normalisation adaptative
    # On évite que des pics aberrants n'écrasent la colormap en utilisant un percentile 99% 
    # au lieu du maximum absolu, mais si la carte est vide, on gère la division par zéro.
    max_val = np.percentile(density_map, 99)
    if max_val > 0:
        norm_map = np.clip(density_map / max_val, 0.0, 1.0)
    else:
        norm_map = density_map
        
    # Appliquer la colormap
    import matplotlib
    try:
        # Nouvelle API matplotlib (>= 3.7)
        if hasattr(matplotlib, 'colormaps'):
            cmap = matplotlib.colormaps.get_cmap(colormap_name)
        else:
            cmap = cm.get_cmap(colormap_name)
    except Exception:
        if hasattr(matplotlib, 'colormaps'):
            cmap = matplotlib.colormaps.get_cmap("jet")
        else:
            cmap = cm.get_cmap("jet") # Fallback
    # cmap retourne RGBA (H, W, 4) dans [0, 1]
    heatmap_rgba = cmap(norm_map)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    
    # Conversion RGB -> BGR pour cv2
    heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
    
    # Créer un masque d'alpha dynamique : les zones vides (densité ~ 0) sont plus transparentes
    # Cela permet de garder l'image originale nette là où il n'y a pas de trafic
    dynamic_alpha = norm_map[..., np.newaxis] * alpha
    
    # Blend: result = heatmap * dynamic_alpha + image * (1 - dynamic_alpha)
    overlay = (heatmap_bgr * dynamic_alpha + image_bgr * (1 - dynamic_alpha)).astype(np.uint8)
    
    return overlay
