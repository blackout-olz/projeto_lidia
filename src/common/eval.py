import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, cohen_kappa_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
from tqdm import tqdm


def evaluate_on_loader(model, dataloader, device, loss_fn=None, num_classes=4, hyperparams=None, model_name="modelo", save_dir="./relatorios"):
    os.makedirs(save_dir, exist_ok=True)
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

            if loss_fn is not None:
                loss = loss_fn(outputs, labels)
                total_loss += loss.item() * inputs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            total_samples += inputs.size(0)

    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    try:
        auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovo', average='macro')
    except ValueError:
        auc_macro = None
    avg_loss = total_loss / total_samples if loss_fn is not None else None
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    cohen_kappa = cohen_kappa_score(all_labels, all_preds)

    metrics = {
        "Accuracy": accuracy,
        "F1 Score (Macro)": f1_macro,
        "AUC-ROC (Macro)": auc_macro,
        "Loss": avg_loss,
        "Precision": precision,
        "Recall": recall,
        "Cohen's Kappa": cohen_kappa
    }

    # Gerar gráficos
    def plot_and_save(fig_func, filename):
        path = os.path.join(save_dir, filename)
        fig_func(path)
        return path

    def plot_confusion(path):
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        plt.title("Matriz de Confusão")
        plt.savefig(path)
        plt.close()

    def plot_confusion_normalized(path):
        cm = confusion_matrix(all_labels, all_preds, normalize='true')
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Purples", xticklabels=range(num_classes), yticklabels=range(num_classes))
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        plt.title("Matriz de Confusão Normalizada")
        plt.savefig(path)
        plt.close()

    def plot_roc(path):
        y_bin = label_binarize(all_labels, classes=range(num_classes))
        plt.figure()
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], np.array(all_probs)[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Classe {i} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("Curvas ROC por Classe")
        plt.legend(loc="lower right")
        plt.savefig(path)
        plt.close()

    def plot_prob_dist(path):
        probs_max = np.max(all_probs, axis=1)
        plt.figure()
        plt.hist(probs_max, bins=20, color='skyblue', edgecolor='black')
        plt.title("Distribuição das Confianças (Máx. Softmax)")
        plt.xlabel("Confiança da Previsão")
        plt.ylabel("Número de Amostras")
        plt.savefig(path)
        plt.close()

    def plot_error_class(path):
        errors = np.array(all_labels) != np.array(all_preds)
        error_counts = np.bincount(np.array(all_labels)[errors], minlength=num_classes)
        plt.figure()
        plt.bar(range(num_classes), error_counts, color='salmon', edgecolor='black')
        plt.xlabel("Classe Verdadeira")
        plt.ylabel("Erros")
        plt.title("Número de Erros por Classe")
        plt.savefig(path)
        plt.close()

    img_paths = {
        "Matriz de Confusão": plot_and_save(plot_confusion, "confusion.png"),
        "Matriz Normalizada": plot_and_save(plot_confusion_normalized, "confusion_normalized.png"),
        "Curva ROC": plot_and_save(plot_roc, "roc_curve.png"),
        "Distribuição de Confiança": plot_and_save(plot_prob_dist, "prob_dist.png"),
        "Erro por Classe": plot_and_save(plot_error_class, "errors_per_class.png"),
    }

    # Geração do PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Relatório de Avaliação de Modelo", ln=True, align="C")
    pdf.ln(10)

    def add_title(text):
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, text, ln=True)

    def add_text(text):
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 8, text)
        pdf.ln()

    add_title("Modelo:")
    add_text(model_name)

    add_title("Hiperparâmetros:")
    for k, v in (hyperparams or {}).items():
        add_text(f"{k}: {v}")

    add_title("Métricas:")
    for k, v in metrics.items():
        if v is not None:
            add_text(f"{k}: {v:.4f}")
        else:
            add_text(f"{k}: não calculado")

    for title, path in img_paths.items():
        add_title(title)
        pdf.image(path, x=30, w=150)
        pdf.ln(10)

    filename = f"{model_name}_" + "_".join(f"{k}={v}" for k, v in (hyperparams or {}).items()) + ".pdf"
    filename = filename.replace("/", "-")
    full_path = os.path.join(save_dir, filename)
    pdf.output(full_path)

    return metrics, full_path
