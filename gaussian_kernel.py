"""
data/gaussian_kernel.py — Génération de Density Maps par Noyau Gaussien Adaptatif
====================================================================================
Groupe 8 : Estimation de la Densité de Trafic Urbain

Principe :
    Pour chaque véhicule annoté (point px, py), on calcule un σ adaptatif
    basé sur la distance moyenne aux k plus proches voisins.
    Cette adaptation capture la perspective de la caméra : les véhicules
    lointains (plus petits) reçoivent un σ réduit, les proches un σ plus large.

    σᵢ = β × (1/k) × Σⱼ d(pᵢ, pⱼ)    pour les k plus proches voisins j

    La density map finale est la somme des gaussiennes normalisées :
        D(x,y) = Σᵢ N(x,y ; pᵢ, σᵢ²I)

    Propriété de conservation : ∫∫ D(x,y) dx dy = N_véhicules

Référence :
    Li et al., "CSRNet: Dilated Convolutional Neural Networks for
    Understanding the Highly Congested Scenes", CVPR 2018.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from typing import Tuple, Optional

from config import density_cfg


# ─────────────────────────────────────────────────────────────
# NOYAU GAUSSIEN FIXE (fallback scènes peu peuplées)
# ─────────────────────────────────────────────────────────────

def gaussian_kernel_fixed(
    points: np.ndarray,
    map_shape: Tuple[int, int],
    sigma: float = 4.0,
) -> np.ndarray:
    """
    Génère une density map avec σ fixe pour toutes les annotations.

    Utilisé quand le nombre de points est insuffisant pour calculer
    des voisins (< k_nearest + 1 véhicules dans la scène).

    Args:
        points    : (N, 2) array de coordonnées (col, row) = (x, y)
        map_shape : (H, W) dimensions de la carte de densité
        sigma     : écart-type gaussien fixe en pixels

    Returns:
        density_map : (H, W) float32, normalisée → somme = N véhicules
    """
    H, W = map_shape
    density = np.zeros((H, W), dtype=np.float32)

    if len(points) == 0:
        return density

    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        # Clamp pour rester dans les limites de l'image
        xi = np.clip(xi, 0, W - 1)
        yi = np.clip(yi, 0, H - 1)
        density[yi, xi] += 1.0

    density = gaussian_filter(density, sigma=sigma)
    return density.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# NOYAU GAUSSIEN ADAPTATIF (méthode principale)
# ─────────────────────────────────────────────────────────────

def gaussian_kernel_adaptive(
    points: np.ndarray,
    map_shape: Tuple[int, int],
    k: int           = density_cfg.k_nearest,
    beta: float      = density_cfg.beta,
    sigma_min: float = density_cfg.sigma_min,
    sigma_max: float = density_cfg.sigma_max,
) -> np.ndarray:
    """
    Génère une density map avec σ adaptatif par point selon ses k voisins.

    Algorithme :
        1. Construire un KD-Tree sur les N points annotés
        2. Pour chaque point i, trouver ses k plus proches voisins
        3. Calculer σᵢ = β × mean(distances aux k voisins)
        4. Clipper σᵢ dans [sigma_min, sigma_max]
        5. Placer une gaussienne 2D de paramètre σᵢ centrée en pᵢ
        6. Sommer toutes les gaussiennes → density map

    Args:
        points    : (N, 2) array [x, y] (coordonnées pixel)
        map_shape : (H, W)
        k         : nombre de voisins pour estimation σ
        beta      : facteur de proportionnalité σ
        sigma_min : seuil bas σ (pixels)
        sigma_max : seuil haut σ (pixels)

    Returns:
        density_map : (H, W) float32
                      Propriété : density_map.sum() ≈ len(points)
    """
    H, W = map_shape
    density = np.zeros((H, W), dtype=np.float32)

    if len(points) == 0:
        return density

    # Fallback si pas assez de points pour les voisins
    if len(points) <= k:
        return gaussian_kernel_fixed(points, map_shape, sigma=sigma_min * 2)

    # KD-Tree pour recherche efficace des voisins (O(N log N))
    tree = KDTree(points)

    for i, (x, y) in enumerate(points):
        # k+1 car le point lui-même est inclus dans la recherche
        distances, _ = tree.query([x, y], k=k + 1)
        # Exclure distance nulle (le point lui-même)
        neighbor_distances = distances[1:]

        # σ adaptatif selon la géométrie locale
        sigma = beta * np.mean(neighbor_distances)
        sigma = float(np.clip(sigma, sigma_min, sigma_max))

        # Placement du point sur une carte temporaire
        point_map = np.zeros((H, W), dtype=np.float32)
        xi = int(np.clip(round(x), 0, W - 1))
        yi = int(np.clip(round(y), 0, H - 1))
        point_map[yi, xi] = 1.0

        # Convolution gaussienne avec σᵢ
        point_density = gaussian_filter(point_map, sigma=sigma)
        density += point_density

    return density.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# DENSITY MAP POUR UNE IMAGE (interface haut niveau)
# ─────────────────────────────────────────────────────────────

def generate_density_map(
    points: np.ndarray,
    image_shape: Tuple[int, int],
    adaptive: bool = True,
    downscale: int = 8,
    sigma_fixed: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interface principale de génération des density maps.

    Génère deux cartes :
        - density_full  : même résolution que l'image (H, W)
        - density_small : résolution réduite (H//8, W//8) = target CSRNet

    Args:
        points      : (N, 2) annotations [x, y] en pixels
        image_shape : (H, W) taille originale de l'image
        adaptive    : utiliser le noyau adaptatif (True) ou fixe (False)
        downscale   : facteur de réduction (8 pour CSRNet)
        sigma_fixed : σ fixe si adaptive=False

    Returns:
        density_full  : (H, W)   float32, somme ≈ N
        density_small : (H//8, W//8) float32, somme ≈ N
    """
    H, W = image_shape

    if adaptive:
        density_full = gaussian_kernel_adaptive(points, (H, W))
    else:
        sig = sigma_fixed or density_cfg.sigma_min * 2
        density_full = gaussian_kernel_fixed(points, (H, W), sigma=sig)

    # Réduction résolution → target CSRNet (1/8)
    # On redimensionne la density map et on compense la somme
    H_s, W_s = H // downscale, W // downscale
    density_small = _resize_density_map(density_full, (H_s, W_s))

    return density_full, density_small


def _resize_density_map(density: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Redimensionne une density map en conservant le compte de véhicules.

    Simple moyenne de bloc (bin) : la somme est préservée par facteur
    de normalisation après interpolation.

    Args:
        density      : (H, W) carte source
        target_shape : (H', W') taille cible

    Returns:
        (H', W') density map normalisée
    """
    import cv2

    H_t, W_t = target_shape
    original_sum = density.sum()

    # Bicubique pour préserver les gradients fins
    resized = cv2.resize(density, (W_t, H_t), interpolation=cv2.INTER_CUBIC)
    resized = np.maximum(resized, 0.0)  # Clip valeurs négatives (artefacts bicubique)

    # Renormalisation pour conserver le compte exact
    if resized.sum() > 1e-8:
        resized *= original_sum / resized.sum()

    return resized.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# VISUALISATION RAPIDE (debug)
# ─────────────────────────────────────────────────────────────

def visualize_density_map(
    image: np.ndarray,
    density: np.ndarray,
    points: Optional[np.ndarray] = None,
    title: str = "Density Map",
) -> np.ndarray:
    """
    Superpose la density map sur l'image originale pour visualisation.

    Args:
        image   : (H, W, 3) uint8 image BGR
        density : (H, W) float32 density map
        points  : (N, 2) annotations optionnelles
        title   : titre de la figure

    Returns:
        overlay : (H, W, 3) uint8 image avec heatmap superposée
    """
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    H, W = image.shape[:2]

    # Redimensionner density map si nécessaire
    if density.shape != (H, W):
        density_vis = cv2.resize(density, (W, H), interpolation=cv2.INTER_CUBIC)
    else:
        density_vis = density.copy()

    # Normaliser pour colormap
    if density_vis.max() > 0:
        density_norm = density_vis / density_vis.max()
    else:
        density_norm = density_vis

    # Appliquer colormap jet
    colormap = cm.get_cmap("jet")
    heatmap_rgba = colormap(density_norm)
    heatmap_rgb  = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_bgr  = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)

    # Blend
    overlay = cv2.addWeighted(image, 0.5, heatmap_bgr, 0.5, 0)

    # Dessiner les points d'annotation
    if points is not None:
        for x, y in points:
            cv2.circle(overlay, (int(x), int(y)), 3, (0, 255, 0), -1)

    count = density.sum()
    cv2.putText(
        overlay,
        f"Count: {count:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    return overlay
