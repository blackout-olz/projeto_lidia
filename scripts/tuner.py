import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import optuna
import torch
from sklearn.model_selection import train_test_split
from src.model_2d.vit2d import get_model
from src.common.eval import evaluate_on_loader
from src.data_loader_2d.loader import load_dataframes
from src.data_loader_2d.dataset import MRIDatasetParquet
from src.train import train_model
from torch.utils.data import DataLoader

# Ajuste essas variáveis conforme seu projeto
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 4
model_name = "vit_small_patch16_224"  # Altere para testar outras
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def objective(trial):
    # Sugere hiperparâmetros
    hyperparams = {
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "drop_path": trial.suggest_float("drop_path", 0.0, 0.5),
        "attn_drop": trial.suggest_float("attn_drop", 0.0, 0.3),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
    }
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_epochs = trial.suggest_int("num_epochs", 10, 50)
    early_stopping = trial.suggest_int("early_stopping_patience", 3, 10)

    df_train_raw, df_test = load_dataframes()

    df_train, df_val = train_test_split(
        df_train_raw,
        test_size=0.25,
        stratify=df_train_raw['label'],
        random_state=42
    )

    train_dataset = MRIDatasetParquet(df_train)
    val_dataset   = MRIDatasetParquet(df_val)
    test_dataset  = MRIDatasetParquet(df_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = get_model(model_name, num_classes=num_classes, hyperparams=hyperparams).to(device)

    model = train_model(model, train_loader, val_loader, device,
                        num_epochs=num_epochs, patience=early_stopping)

    # Atualiza hiperparâmetros para o relatório
    hyperparams.update({
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "early_stopping_patience": early_stopping
    })

    # Diretório para salvar relatórios/gráficos
    save_dir = os.path.join("logs", f"{model_name}_trial{trial.number}")
    os.makedirs(save_dir, exist_ok=True)

    # Avaliação e gera relatório
    metrics, report_path = evaluate_on_loader(
        model=model,
        dataloader=val_loader,
        device=device,
        loss_fn=torch.nn.CrossEntropyLoss(),
        num_classes=num_classes,
        hyperparams=hyperparams,
        model_name=model_name,
        save_dir=save_dir
    )

    # Salva modelo em models/
    model_path = os.path.join("models", f"{model_name}_trial{trial.number}.pt")
    torch.save(model.state_dict(), model_path)

    trial.set_user_attr("model_name", model_name)

    return metrics["f1_macro"]


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    with open(os.path.join("logs", f"melhor_trial_{model_name}.txt"), "w") as f:
        f.write(f"Modelo: {study.best_trial.user_attrs['model_name']}\n")
        f.write(f"Melhor trial: {study.best_trial.number}\n")
        f.write(f"Valor: {study.best_trial.value}\n")
        for key, value in study.best_trial.params.items():
            f.write(f"{key}: {value}\n")