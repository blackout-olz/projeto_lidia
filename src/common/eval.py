import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_on_loader(model, dataloader, device, loss_fn=None, num_classes=3, save_confusion_path=None):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Avaliando", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Acumula loss
            if loss_fn is not None:
                loss = loss_fn(outputs, labels)
                total_loss += loss.item() * inputs.size(0)

            # Coleta resultados
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            total_samples += inputs.size(0)

    # Métricas
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    try:
        auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovo', average='macro')
    except ValueError:
        auc_macro = None

    avg_loss = total_loss / total_samples if loss_fn is not None else None

    # Matriz de confusão (opcional)
    if save_confusion_path:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        plt.title("Matriz de Confusão")
        plt.savefig(save_confusion_path)
        plt.close()

    # Print padrão conforme relatório
    print(f"Accuracy:      {accuracy * 100:.2f}%")
    print(f"F1-Score Macro:{f1_macro * 100:.2f}%")
    print(f"AUC-ROC Macro: {auc_macro:.2f}" if auc_macro is not None else "AUC-ROC não calculado")
    print(f"Loss Final:    {avg_loss:.4f}" if avg_loss is not None else "Loss não calculada")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "auc_roc": auc_macro,
        "loss": avg_loss
    }
