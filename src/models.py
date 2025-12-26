# src/models.py

import torch
import torch.nn as nn

class DRClassifier(nn.Module):
    """Custom CNN for Diabetic Retinopathy Classification"""
    def __init__(self, num_classes=5):
        super(DRClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Output: (batch_size, 64, 1, 1)
            nn.Flatten(),                  # Output: (batch_size, 64)
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x 

