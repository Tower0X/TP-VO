"""
models/loss.py — Fonction de Perte (Loss) Spatiale Pondérée
===========================================================
Groupe 8 : Estimation de la Densité de Trafic Urbain

Cette fonction de perte est une innovation spécifique du projet.
Elle pénalise plus sévèrement les erreurs dans les zones de haute densité
tout en assurant une transition douce et dérivable grâce à une fonction sigmoïde.
"""

import torch
import torch.nn as nn

from config import train_cfg

class SpatiallyWeightedMSELoss(nn.Module):
    """
    Mean Squared Error (MSE) avec pondération spatiale adaptative.
    
    L = (1/N) * Σ_i w_i * (ŷ_i - y_i)²
    w_i = 1 + (λ_high - 1) * sigmoid(α * (y_i - τ))
    """
    
    def __init__(self, 
                 lambda_high: float = train_cfg.lambda_high, 
                 threshold: float = train_cfg.density_threshold,
                 alpha: float = 100.0):
        """
        Args:
            lambda_high : Facteur de pénalité maximal pour les zones denses.
            threshold   : Seuil de densité (τ) à partir duquel on considère la zone comme "dense".
            alpha       : Pente de la sigmoïde (contrôle la netteté de la transition).
        """
        super(SpatiallyWeightedMSELoss, self).__init__()
        self.lambda_high = lambda_high
        self.threshold = threshold
        self.alpha = alpha
        
        # Loss de base sans réduction pour pouvoir appliquer les poids par pixel
        self.base_loss = nn.MSELoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calcule la perte pondérée.
        
        Args:
            pred   : Tenseur prédit (B, 1, H, W)
            target : Tenseur de vérité terrain (B, 1, H, W)
            
        Returns:
            Scalar tensor de la perte moyenne.
        """
        # 1. Calcul de la perte non pondérée (pixel par pixel)
        # shape: (B, 1, H, W)
        mse = self.base_loss(pred, target)
        
        # 2. Calcul de la pondération spatiale
        # w_i = 1 + (λ - 1) * σ(α * (y_i - τ))
        # Utilise target (la vérité terrain) pour définir les poids
        # shape: (B, 1, H, W)
        transition = torch.sigmoid(self.alpha * (target - self.threshold))
        weights = 1.0 + (self.lambda_high - 1.0) * transition
        
        # 3. Application des poids
        weighted_mse = mse * weights
        
        # 4. Moyenne sur l'ensemble du batch (réduction)
        return torch.mean(weighted_mse)
