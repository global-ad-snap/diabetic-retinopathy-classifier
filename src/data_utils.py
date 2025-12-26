# src/data_utils.py

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# === 1. Load and Clean CSV ===
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['id_code'] = df['id_code'].str.strip()
    return df

# === 2. Split into Train/Val Sets ===
def split_data(df, test_size=0.2, seed=42):
    X = df['id_code']
    y = df['diagnosis']
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=test_size, random_state=seed)
    train_df = pd.DataFrame({'id_code': X_train, 'diagnosis': y_train})
    val_df = pd.DataFrame({'id_code': X_val, 'diagnosis': y_val})
    return train_df, val_df

# === 3. Compute Class Weights ===
def compute_weights(y_train):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    return torch.tensor(class_weights, dtype=torch.float)

# === 4. Define Transforms ===
def get_transforms():
    minority_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])
    majority_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    return {
        0: majority_transform,
        1: minority_transform,
        2: majority_transform,
        3: minority_transform,
        4: minority_transform
    }

# === 5. Custom Dataset Class ===
class DRDataset(Dataset):
    def __init__(self, df, root_dir, transform_dict):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform_dict = transform_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'id_code']
        label = self.df.loc[idx, 'diagnosis']
        img_path = os.path.join(self.root_dir, f"{img_id}.png")
        image = Image.open(img_path).convert('RGB')
        transform = self.transform_dict.get(label, transforms.ToTensor())
        image = transform(image)
        return image, label

# === 6. Build DataLoaders ===
def get_loaders(train_df, val_df, transform_dict, root_dir, batch_size=32):
    train_dataset = DRDataset(train_df, root_dir, transform_dict)
    val_dataset = DRDataset(val_df, root_dir, transform_dict)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader  


