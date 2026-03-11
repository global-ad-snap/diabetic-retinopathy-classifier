
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    roc_auc_score, roc_curve, auc, confusion_matrix
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_CSV   = r"C:\Users\Jacy Heather\Desktop\MLorAI_Healthcare\Diabetic Retinopathy Classifier\data\train.csv"
IMG_DIR    = r"C:\Users\Jacy Heather\Desktop\MLorAI_Healthcare\Diabetic Retinopathy Classifier\data\train_images"
VISUALS    = r"C:\Users\Jacy Heather\Desktop\MLorAI_Healthcare\Diabetic Retinopathy Classifier\visuals"
MODEL_PATH = r"C:\Users\Jacy Heather\Desktop\MLorAI_Healthcare\Diabetic Retinopathy Classifier\src\best_model.pth"

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS       = 20
BATCH_SIZE   = 32
LR_HEAD      = 0.001   # learning rate for new classification head
LR_BACKBONE  = 0.0001  # lower LR for pretrained layers (fine-tuning)
PATIENCE     = 5
IMG_SIZE     = 224     # ResNet expects 224x224
RANDOM_STATE = 42


# ── Model: Pretrained ResNet-18 ────────────────────────────────────────────────
class DRClassifier(nn.Module):
    """
    ResNet-18 pretrained on ImageNet, with custom head for 5-class DR grading.
    Pretrained weights give a massive head start on small medical datasets.
    """
    def __init__(self, num_classes=5):
        super().__init__()
        # Load pretrained ResNet-18
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Keep all layers except the final FC
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


# ── Dataset ────────────────────────────────────────────────────────────────────
class DRDataset(Dataset):
    def __init__(self, df, root_dir, transform):
        self.df       = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id   = self.df.loc[idx, 'id_code']
        label    = int(self.df.loc[idx, 'diagnosis'])
        img_path = os.path.join(self.root_dir, f"{img_id}.png")
        image    = Image.open(img_path).convert('RGB')
        return self.transform(image), label


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Note: ResNet on CPU will be slow (~10-20 min/epoch). Consider Google Colab if too slow.")
    os.makedirs(VISUALS, exist_ok=True)

    # ── 1. Load & Split ────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_CSV)
    df['id_code'] = df['id_code'].str.strip()
    print("Class distribution:\n", df['diagnosis'].value_counts().sort_index())

    X_train, X_val, y_train, y_val = train_test_split(
        df['id_code'], df['diagnosis'],
        stratify=df['diagnosis'],
        test_size=0.2,
        random_state=RANDOM_STATE
    )
    train_df = pd.DataFrame({'id_code': X_train, 'diagnosis': y_train}).reset_index(drop=True)
    val_df   = pd.DataFrame({'id_code': X_val,   'diagnosis': y_val  }).reset_index(drop=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # ── 2. Class Weights + Sampler ─────────────────────────────────────────────
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    print("Class weights:", np.round(class_weights, 3))

    sample_weights = [class_weights[int(label)] for label in train_df['diagnosis']]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # ── 3. Transforms (ImageNet normalisation for ResNet) ──────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ── 4. DataLoaders ─────────────────────────────────────────────────────────
    train_dataset = DRDataset(train_df, IMG_DIR, train_transform)
    val_dataset   = DRDataset(val_df,   IMG_DIR, val_transform)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               sampler=sampler, num_workers=0)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                               shuffle=False,  num_workers=0)

    # ── 5. Model + Optimiser (two param groups: backbone vs head) ──────────────
    model = DRClassifier().to(device)

    optimizer = torch.optim.Adam([
        {'params': model.backbone.parameters(),   'lr': LR_BACKBONE},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # ── 6. Training Loop ───────────────────────────────────────────────────────
    best_val_loss     = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                val_loss += criterion(model(images), labels).item()

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(running_loss)
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch+1:02d} | Train: {running_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("  ✅ Saved best model")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ── 7. Evaluate ────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

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

    acc    = accuracy_score(all_labels, all_preds)
    f1     = f1_score(all_labels, all_preds, average='weighted')
    kappa  = cohen_kappa_score(all_labels, all_preds)
    y_bin  = label_binarize(all_labels, classes=[0,1,2,3,4])
    roc_auc = roc_auc_score(y_bin, all_probs, average='macro', multi_class='ovr')

    print("\n═══ Evaluation Metrics ═══")
    print(f"  Accuracy      : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1 (weighted) : {f1:.4f}")
    print(f"  Cohen's Kappa : {kappa:.4f}")
    print(f"  ROC-AUC       : {roc_auc:.4f}")

    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    print("\n═══ Per-Class Accuracy ═══")
    for cls in range(5):
        mask = all_labels == cls
        if mask.sum() == 0: continue
        print(f"  Class {cls} - {class_names[cls]:<18}: {(all_preds[mask]==cls).mean()*100:5.1f}%  (n={mask.sum()})")

    # ── 8. Plots ───────────────────────────────────────────────────────────────
    actual_epochs = len(history['train_loss'])

    plt.figure(figsize=(8,5))
    plt.plot(range(1, actual_epochs+1), history['train_loss'], 'o-', label='Train Loss')
    plt.plot(range(1, actual_epochs+1), history['val_loss'],   's-', label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(VISUALS, 'loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(5), yticklabels=range(5))
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title('Figure 3. Confusion Matrix (Validation Set)')
    plt.savefig(os.path.join(VISUALS, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,6))
    for i in range(5):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} - {class_names[i]} (AUC={auc(fpr,tpr):.2f})")
    plt.plot([0,1],[0,1],'k--', label='Random (AUC=0.50)')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Figure 4. ROC-AUC Curve')
    plt.legend(loc='lower right'); plt.grid(True)
    plt.savefig(os.path.join(VISUALS, 'roc_auc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✅ All plots saved to: {VISUALS}")