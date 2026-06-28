"""
models/csrnet.py — Architecture CSRNet
======================================
Groupe 8 : Estimation de la Densité de Trafic Urbain

Architecture basée sur "CSRNet: Dilated Convolutional Neural Networks 
for Understanding the Highly Congested Scenes" (Li et al., CVPR 2018).
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG16_Weights

from config import model_cfg

class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        
        # ─────────────────────────────────────────────────────────────
        # FRONT-END (VGG-16 tronqué)
        # ─────────────────────────────────────────────────────────────
        # CSRNet utilise les 10 premières couches de convolution de VGG-16
        # avec seulement 3 couches de MaxPool, réduisant la résolution d'un facteur 8.
        # Structure VGG16: 
        # (conv1_1, conv1_2, pool1) -> 2 convs (facteur 1/2)
        # (conv2_1, conv2_2, pool2) -> 2 convs (facteur 1/4)
        # (conv3_1, conv3_2, conv3_3, pool3) -> 3 convs (facteur 1/8)
        # (conv4_1, conv4_2, conv4_3) -> 3 convs (sans pool4)
        
        self.frontend = self._make_frontend()
        
        # ─────────────────────────────────────────────────────────────
        # BACK-END (Convolutions dilatées)
        # ─────────────────────────────────────────────────────────────
        # Canaux de sortie du front-end : 512
        # Taux de dilatation : [2, 2, 2, 2, 2]
        # Canaux : 512 -> 512 -> 256 -> 128 -> 64 -> 64
        self.backend = self._make_backend()
        
        # Couche finale de prédiction (1x1 conv)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        # ─────────────────────────────────────────────────────────────
        # INITIALISATION
        # ─────────────────────────────────────────────────────────────
        if not model_cfg.vgg_pretrained:
            self._initialize_weights(self.frontend)
        self._initialize_weights(self.backend)
        self._initialize_weights(self.output_layer)

    def _make_frontend(self):
        # Charger VGG-16 depuis torchvision
        if model_cfg.vgg_pretrained:
            vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
        else:
            vgg16 = models.vgg16()
            
        # Extraire les features jusqu'à conv4_3 (index 22 dans vgg16.features)
        # 0-4: conv1 + pool1
        # 5-9: conv2 + pool2
        # 10-16: conv3 + pool3
        # 17-22: conv4_1, conv4_2, conv4_3 (pas de pool4 qui est à l'index 23)
        features = list(vgg16.features.children())[:23]
        
        return nn.Sequential(*features)

    def _make_backend(self):
        layers = []
        in_channels = model_cfg.frontend_out_channels  # 512
        
        dilations = model_cfg.backend_dilation
        out_channels_list = model_cfg.backend_channels
        
        for out_channels, dilation in zip(out_channels_list, dilations):
            # Pour conserver la même taille spatiale : padding = dilation
            layers.append(nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                padding=dilation, 
                dilation=dilation
            ))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
            
        return nn.Sequential(*layers)

    def _initialize_weights(self, module):
        """
        Initialisation gaussienne (N(0, 0.01)) selon le papier original,
        pour les couches ajoutées (backend + output).
        """
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Tenseur d'entrée de taille (B, 3, H, W)
        Returns:
            Density map de taille (B, 1, H/8, W/8)
        """
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        
        # ReLU final : les densités négatives sont physiquement impossibles.
        # Sans cette contrainte, les valeurs négatives annulent les positives
        # lors du comptage par sommation et gaspillent du signal de gradient.
        x = torch.relu(x)
        
        return x
