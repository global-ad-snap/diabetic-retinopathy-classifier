# evaluate_fixed.py — matches ResNet train_fixed.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    roc_auc_score, roc_curve, auc, confusion_matrix
)
from sklearn.preprocessing import label_binarize

DATA_CSV   = r"C:\Users\Jacy Heather\Desktop\MLorAI_Healthcare\Diabetic Retinopathy Classifier\data\train.csv"
IMG_DIR    = r"C:\Users\Jacy Heather\Desktop\MLorAI_Healthcare\Diabetic Retinopathy Classifier\data\train_images"
VISUALS    = r"C:\Users\Jacy Heather\Desktop\MLorAI_Healthcare\Diabetic Retinopathy Classifier\visuals"
MODEL_PATH = r"C:\Users\Jacy Heather\Desktop\MLorAI_Healthcare\Diabetic Retinopathy Classifier\src\best_model.pth"
IMG_SIZE   = 224
RANDOM_STATE = 42

class DRClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone   = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))

class DRDataset(Dataset):
    def __init__(self, df, root_dir, transform):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_id   = self.df.loc[idx, 'id_code']
        label    = int(self.df.loc[idx, 'diagnosis'])
        img_path = os.path.join(self.root_dir, f"{img_id}.png")
        return self.transform(Image.open(img_path).convert('RGB')), label

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(VISUALS, exist_ok=True)

    df = pd.read_csv(DATA_CSV)
    df['id_code'] = df['id_code'].str.strip()
    _, X_val, _, y_val = train_test_split(
        df['id_code'], df['diagnosis'],
        stratify=df['diagnosis'], test_size=0.2, random_state=RANDOM_STATE
    )
    val_df = pd.DataFrame({'id_code': X_val, 'diagnosis': y_val}).reset_index(drop=True)
    print(f"Validation samples: {len(val_df)}")

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_loader = DataLoader(
        DRDataset(val_df, IMG_DIR, val_transform),
        batch_size=32, shuffle=False, num_workers=0
    )

    model = DRClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Loaded model from: {MODEL_PATH}")

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            probs  = torch.softmax(model(images), dim=1)
            preds  = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    acc     = accuracy_score(all_labels, all_preds)
    f1      = f1_score(all_labels, all_preds, average='weighted')
    kappa   = cohen_kappa_score(all_labels, all_preds)
    y_bin   = label_binarize(all_labels, classes=[0,1,2,3,4])
    roc_auc = roc_auc_score(y_bin, all_probs, average='macro', multi_class='ovr')

    print("\n═══ Evaluation Metrics ═══")
    print(f"  Accuracy      : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1 (weighted) : {f1:.4f}")
    print(f"  Cohen's Kappa : {kappa:.4f}")
    print(f"  ROC-AUC       : {roc_auc:.4f}")

    class_names = ['No DR','Mild','Moderate','Severe','Proliferative DR']
    print("\n═══ Per-Class Accuracy ═══")
    for cls in range(5):
        mask = all_labels == cls
        if mask.sum() == 0: continue
        print(f"  Class {cls} - {class_names[cls]:<18}: {(all_preds[mask]==cls).mean()*100:5.1f}%  (n={mask.sum()})")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(5), yticklabels=range(5))
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title('Figure 3. Confusion Matrix (Validation Set)')
    plt.savefig(os.path.join(VISUALS, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ROC-AUC
    plt.figure(figsize=(8,6))
    for i in range(5):
        fpr, tpr, _ = roc_curve(y_bin[:,i], all_probs[:,i])
        plt.plot(fpr, tpr, label=f"Class {i} - {class_names[i]} (AUC={auc(fpr,tpr):.2f})")
    plt.plot([0,1],[0,1],'k--', label='Random (AUC=0.50)')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Figure 4. ROC-AUC Curve')
    plt.legend(loc='lower right'); plt.grid(True)
    plt.savefig(os.path.join(VISUALS, 'roc_auc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✅ All plots saved to: {VISUALS}")