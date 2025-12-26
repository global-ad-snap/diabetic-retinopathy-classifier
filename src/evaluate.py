# src/evaluate.py

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

def evaluate_model(model, val_loader, device, num_classes=5):
    model.eval()
    all_preds = []
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    # Accuracy
    acc = accuracy_score(all_labels, all_preds)

    # F1-score (weighted)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Cohen’s Kappa
    kappa = cohen_kappa_score(all_labels, all_preds)

    # ROC–AUC (macro, multi-class)
    y_true_bin = label_binarize(all_labels, classes=list(range(num_classes)))
    y_pred_prob = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()
    roc_auc = roc_auc_score(y_true_bin, y_pred_prob, average='macro', multi_class='ovr')

    return {
        "accuracy": acc,
        "f1_score": f1,
        "cohen_kappa": kappa,
        "roc_auc": roc_auc
    }
