from timm import create_model

SUPPORTED_VITS = [
    "vit_small_patch16_224",
    "crossvit_15_240",
    "levit_192",
    "deit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
]

def get_model(model_name, num_classes, hyperparams):
    if model_name not in SUPPORTED_VITS:
        raise ValueError(f"Modelo '{model_name}' não está na lista de ViTs suportados.")

    model = create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=hyperparams.get("dropout", 0.0),
        drop_path_rate=hyperparams.get("drop_path", 0.0),
        attn_drop_rate=hyperparams.get("attn_drop", 0.0),
    )
    return model
