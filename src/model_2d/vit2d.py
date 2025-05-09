import torch.nn as nn
from torchvision.models import vit_b_16
from timm import create_model


class ViT2DClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.vit = vit_b_16(pretrained=True)

        # Substituir a cabeça: acessar corretamente o in_features
        in_features = self.vit.heads[0].in_features
        self.vit.heads = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vit(x)
    

class TimmViT2DClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Cria o modelo com dropout regular e na atenção
        self.vit = create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=num_classes,  # já substitui a cabeça
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path_rate=0.1
        )

        # Se quiser controlar manualmente o head:
        # in_features = self.vit.head.in_features
        # self.vit.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vit(x)
