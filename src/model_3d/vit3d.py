import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNETRClassifier(nn.Module):
    def __init__(self, in_channels=2, num_classes=4):
        super().__init__()

        # Backbone SwinUNETR com pesos pré-treinados (task 09 spleen)
        self.backbone = SwinUNETR(
            img_size=(64, 64, 64),
            in_channels=in_channels,
            out_channels=768,
            feature_size=96,
            use_checkpoint=True
        )

        # Cabeça de classificação substituindo a de segmentação
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x