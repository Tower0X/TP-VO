"""
inference/temporal.py — Agrégation Temporelle Glissante
=========================================================
Groupe 8 : Estimation de la Densité de Trafic Urbain

Lisse les prédictions individuelles sur une fenêtre temporelle
pour éliminer le bruit et détecter les tendances macroscopiques 
de congestion (augmentation / diminution du trafic).
"""

from collections import deque
import numpy as np

from config import inference_cfg

class TemporalAggregator:
    def __init__(self, window_frames: int = None):
        """
        Args:
            window_frames: Taille de la fenêtre glissante en nombre de frames.
                           Par défaut, utilise la valeur configurée (ex: 5 min à 25 fps).
        """
        # Utilise __post_init__ fallback de la config si nécessaire
        self.window_frames = window_frames or inference_cfg.window_frames
        if self.window_frames == 0:
             self.window_frames = inference_cfg.fps * 60 * inference_cfg.window_minutes
             
        self.history = deque(maxlen=self.window_frames)
        
        # Moyenne Mobile Exponentielle (EMA) pour la tendance fluide
        self.ema = 0.0
        self.alpha = 2.0 / (min(self.window_frames, 100) + 1.0) # smoothing factor

    def update(self, count: float) -> dict:
        """
        Ajoute une nouvelle observation et retourne les statistiques mises à jour.
        
        Args:
            count: Véhicules estimés sur la frame courante.
            
        Returns:
            dict: {
                'current': float,
                'mean': float,
                'ema': float,
                'trend': str ("hausse", "baisse", "stable"),
                'min': float,
                'max': float,
                'status': str ("Fluide", "Modéré", "Congestion")
            }
        """
        self.history.append(count)
        
        # Mise à jour EMA
        if len(self.history) == 1:
            self.ema = count
        else:
            self.ema = (count * self.alpha) + (self.ema * (1 - self.alpha))
            
        # Calcul des stats
        hist_array = np.array(self.history)
        mean_val = np.mean(hist_array)
        min_val = np.min(hist_array)
        max_val = np.max(hist_array)
        
        # Détection de tendance
        trend = "stable"
        if len(self.history) > 10:
            # Pente simple sur les 10 dernières valeurs
            recent = hist_array[-10:]
            x = np.arange(10)
            slope, _ = np.polyfit(x, recent, 1)
            if slope > 0.5:
                trend = "hausse"
            elif slope < -0.5:
                trend = "baisse"
                
        # Status de congestion (basé sur la moyenne lissée)
        status = self._get_congestion_status(self.ema)
        
        return {
            'current': count,
            'mean': mean_val,
            'ema': self.ema,
            'trend': trend,
            'min': min_val,
            'max': max_val,
            'status': status
        }

    def _get_congestion_status(self, count: float) -> str:
        """Détermine l'état du trafic selon les seuils configurés."""
        if count < inference_cfg.threshold_fluid:
            return "Fluide"
        elif count < inference_cfg.threshold_moderate:
            return "Modéré"
        else:
            return "Congestion"

    def reset(self):
        """Réinitialise l'historique."""
        self.history.clear()
        self.ema = 0.0
