"""
inference/predictor.py — Moteur d'Inférence CSRNet
===================================================
Groupe 8 : Estimation de la Densité de Trafic Urbain

Classe principale pour l'inférence en production. Gère le chargement
du modèle, le pré-traitement des images et le redimensionnement des
cartes de densité en sortie.
"""

from pathlib import Path
from typing import Union, Tuple
import cv2
import numpy as np
import torch
import torchvision.transforms as T

from config import DEVICE, model_cfg
from models.csrnet import CSRNet

class TrafficPredictor:
    def __init__(self, model_path: Union[str, Path] = None):
        """
        Initialise le prédicteur avec un modèle pré-entraîné.
        
        Args:
            model_path: Chemin vers le checkpoint du modèle (.pth)
        """
        self.device = DEVICE
        self.model = CSRNet().to(self.device)
        self.model.eval()
        
        if model_path is not None:
            self.load_model(model_path)
            
        # Pipeline de pré-traitement (identique à validation)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path: Union[str, Path]):
        """Charge les poids du modèle."""
        print(f"Chargement du modèle depuis {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # --- ADAPTATION DYNAMIQUE POUR LE MODÈLE DE LA CAMARADE ---
        if "best_csrnet.pth" in str(model_path):
            import torch.nn as nn
            # Le modèle de la camarade utilise l'architecture CSRNet "Configuration B" complète
            # (6 convolutions dilatées au lieu de 5). On recrée le backend à la volée.
            classmate_channels = [512, 512, 512, 256, 128, 64]
            classmate_dilations = [2, 2, 2, 2, 2, 2]
            
            layers = []
            in_channels = 512
            for out_channels, dilation in zip(classmate_channels, classmate_dilations):
                layers.append(nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation
                ))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                
            self.model.backend = nn.Sequential(*layers).to(self.device)
        # ----------------------------------------------------------
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        print("Modèle chargé avec succès.")

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Prédit la carte de densité et le compte pour une image BGR.
        
        Args:
            image: Image BGR (typiquement issue de cv2.imread)
            
        Returns:
            density_map: Array 2D float32 (taille originale de l'image)
            count: Nombre estimé de véhicules (float)
        """
        H, W = image.shape[:2]
        
        # S'assurer que les dimensions sont divisibles par 8
        # (pour éviter les erreurs de taille après la série de convolutions/poolings)
        pad_h = (8 - (H % 8)) % 8
        pad_w = (8 - (W % 8)) % 8
        
        if pad_h > 0 or pad_w > 0:
            import torch.nn.functional as F
            # cv2 pad: top, bottom, left, right
            image_padded = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w, 
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        else:
            image_padded = image
            
        # Conversion RGB -> PIL ou directement via torchvision
        img_rgb = cv2.cvtColor(image_padded, cv2.COLOR_BGR2RGB)
        
        # Pré-traitement -> (1, 3, H', W')
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        # Inférence
        pred = self.model(img_tensor)
        
        # Récupération de la density map (1, 1, H'/8, W'/8) -> (H'/8, W'/8)
        density_small = pred.squeeze().cpu().numpy()
        density_small = np.maximum(density_small, 0.0)  # ReLU effectif
        
        # Upscale vers la taille originale de l'image (H, W)
        # On utilise cv2.resize qui conserve mieux la structure que l'interpolation PyTorch par défaut
        # Note: on compense la somme pour conserver le compte total (x 64 car H*W / (H/8)*(W/8) = 64)
        count = float(density_small.sum())
        
        density_full = cv2.resize(density_small, (W + pad_w, H + pad_h), interpolation=cv2.INTER_CUBIC)
        density_full = np.maximum(density_full, 0.0)
        
        # Retirer le padding
        if pad_h > 0 or pad_w > 0:
            density_full = density_full[:H, :W]
            
        # Renormaliser pour garantir le compte exact après upscale/crop
        if density_full.sum() > 1e-8 and count > 0:
            density_full = density_full * (count / density_full.sum())
            
        return density_full, count

    @torch.no_grad()
    def predict_batch(self, images: list) -> Tuple[list, list]:
        """
        Prédit sur une liste d'images (pourrait être optimisé avec un vrai DataLoader).
        """
        maps = []
        counts = []
        for img in images:
            d_map, count = self.predict(img)
            maps.append(d_map)
            counts.append(count)
        return maps, counts
