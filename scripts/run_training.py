import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss

from src.data_loader import load_dataframes
from src.dataset import MRIDatasetParquet
from src.model import ViT2DClassifier
from src.train import train_model
from src.eval import evaluate_on_loader

# =====================
# Configurações iniciais
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
num_epochs = 30
patience = 5
num_classes = 4

# =====================
# Carregar dados
# =====================
df_train_raw, df_test = load_dataframes()

# Divisão 60/20/20
df_train, df_val = train_test_split(
    df_train_raw,
    test_size=0.25,  # 25% de 80% = 20% total
    stratify=df_train_raw['label'],
    random_state=42
)

# =====================
# Instanciar datasets
# =====================
train_dataset = MRIDatasetParquet(df_train)
val_dataset   = MRIDatasetParquet(df_val)
test_dataset  = MRIDatasetParquet(df_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =====================
# Instanciar modelo
# =====================

model = ViT2DClassifier(num_classes=4)

# =====================
# Treinamento
# =====================
model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    num_epochs=num_epochs,
    patience=patience
)

# =====================
# Avaliação
# =====================
print("\n📊 Avaliação no conjunto de teste:")
loss_fn = CrossEntropyLoss(label_smoothing=0.1)
metrics = evaluate_on_loader(
    model=model,
    dataloader=test_loader,
    device=device,
    loss_fn=loss_fn,
    num_classes=num_classes,
    save_confusion_path="logs/confusion_matrix.png"
)

# (Opcional) salvar o modelo final
torch.save(model.state_dict(), "models/swinunetr_classification.pth")
