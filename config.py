"""
config.py — Configuration Centralisée du Système CSRNet
=========================================================
Groupe 8 : Estimation de la Densité de Trafic Urbain
Domaine  : Mobilité Intelligente

Toutes les hyperparamètres, chemins et constantes sont
centralisés ici pour garantir la reproductibilité et
faciliter les expériences.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import torch


# ─────────────────────────────────────────────────────────────
# CHEMINS
# ─────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent
DATA_DIR     = ROOT_DIR / "data_root"        # Racine dataset TRANCOS
CKPT_DIR     = ROOT_DIR / "checkpoints"      # Sauvegarde modèles
LOG_DIR      = ROOT_DIR / "logs"             # TensorBoard / W&B
OUTPUT_DIR   = ROOT_DIR / "outputs"          # Résultats inférence

for d in [DATA_DIR, CKPT_DIR, LOG_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# ARCHITECTURE CSRNET
# ─────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """Paramètres de l'architecture CSRNet."""

    # Front-end VGG-16 (sans couches FC)
    vgg_pretrained: bool = True          # Charger poids ImageNet
    frontend_out_channels: int = 512     # Canaux de sortie du front-end

    # Back-end : convolutions dilatées
    # Dilation [2,2,2,2,2] → grand champ réceptif sans perte résolution
    backend_dilation: List[int] = field(
        default_factory=lambda: [2, 2, 2, 2, 2]
    )
    backend_channels: List[int] = field(
        default_factory=lambda: [512, 256, 128, 64, 64]
    )

    # Sortie : density map à 1/8 de la taille d'entrée
    output_scale_factor: int = 8


# ─────────────────────────────────────────────────────────────
# DENSITY MAP — NOYAU GAUSSIEN ADAPTATIF
# ─────────────────────────────────────────────────────────────
@dataclass
class DensityConfig:
    """Paramètres de génération des density maps ground truth."""

    # Nombre de voisins pour σ adaptatif (géométrie de la scène)
    k_nearest: int = 3

    # Facteur de proportionnalité pour σ adaptatif
    beta: float = 0.3

    # σ minimum/maximum (pixels) pour éviter les dégénérescences
    sigma_min: float = 1.0
    sigma_max: float = 15.0

    # Taille du noyau gaussien (doit être impair)
    kernel_size: int = 15


# ─────────────────────────────────────────────────────────────
# DATASET & PREPROCESSING
# ─────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    """Paramètres dataset TRANCOS et augmentations."""

    # Répertoire TRANCOS 
    trancos_root: Path = ROOT_DIR
    train_data_dir: Path = ROOT_DIR / "train_data"
    test_data_dir: Path = ROOT_DIR / "test_data"
    
    # Cache
    cache_density_maps: bool = True

    # Taille de crop pour entraînement
    crop_size: Tuple[int, int] = (400, 400)

    # Normalisation ImageNet (cohérent avec VGG-16 pré-entraîné)
    img_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    img_std:  Tuple[float, ...] = (0.229, 0.224, 0.225)

    # Augmentations actives
    random_flip:        bool = True
    random_crop:        bool = True
    color_jitter:       bool = True   # luminosité, contraste, saturation
    gaussian_noise:     bool = True   # bruit additif σ~0.02
    random_gamma:       bool = True   # correction gamma aléatoire


# ─────────────────────────────────────────────────────────────
# ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    """Hyperparamètres d'entraînement."""

    epochs:        int   = 400
    batch_size:    int   = 8
    num_workers:   int   = 4
    pin_memory:    bool  = torch.cuda.is_available()

    # Optimiseur Adam
    lr:            float = 1e-5
    weight_decay:  float = 1e-4

    # Scheduler : ReduceLROnPlateau
    lr_patience:   int   = 15
    lr_factor:     float = 0.5
    min_lr:        float = 1e-7

    # Loss : MSE spatialement pondérée
    # λ_high = facteur de pénalité zones haute densité
    lambda_high:   float = 5.0
    # Seuil (véhicules/pixel²) pour activer la pondération haute densité
    density_threshold: float = 0.2

    # Sauvegarde
    save_every:    int = 10      # checkpoints toutes les N epochs
    best_metric:   str = "mae"   # critère de sélection du meilleur modèle


# ─────────────────────────────────────────────────────────────
# INFÉRENCE & TEMPOREL
# ─────────────────────────────────────────────────────────────
@dataclass
class InferenceConfig:
    """Paramètres d'inférence et de traitement temporel."""

    # Agrégation temporelle glissante
    fps:                int   = 25         # FPS caméra
    window_minutes:     int   = 5          # Fenêtre glissante (minutes)
    window_frames:      int   = 0          # Sera calculé dans __post_init__

    def __post_init__(self):
        self.window_frames = self.fps * 60 * self.window_minutes


    # Seuils de congestion (véhicules estimés par image)
    threshold_fluid:    float = 20.0       # < 20 véhicules → Fluide
    threshold_moderate: float = 40.0       # 20–40 → Modéré
    # > 40 → Congestion

    # Heatmap overlay
    heatmap_alpha:     float = 0.6        # Transparence overlay
    heatmap_colormap:  str   = "jet"      # Colormap (jet, hot, turbo)


# ─────────────────────────────────────────────────────────────
# INSTANCES GLOBALES (import direct dans les modules)
# ─────────────────────────────────────────────────────────────
model_cfg     = ModelConfig()
density_cfg   = DensityConfig()
data_cfg      = DataConfig()
train_cfg     = TrainConfig()
inference_cfg = InferenceConfig()
