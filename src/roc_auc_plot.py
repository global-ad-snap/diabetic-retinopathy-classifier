import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Simulate data for 5-class classification
np.random.seed(42)
n_classes = 5
n_samples = 300

# True labels (0 to 4)
y_true = np.random.randint(0, n_classes, size=n_samples)

# Binarize labels for ROC computation
y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

# Simulate predicted probabilities (softmax-like)
y_score = np.random.rand(n_samples, n_classes)
y_score = y_score / y_score.sum(axis=1, keepdims=True)

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = plt.cm.viridis(np.linspace(0, 1, n_classes))

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Figure 4. ROC–AUC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()

# Save the figure
plt.savefig('visuals/roc_auc_curve.png')  # Ensure 'visuals/' folder exists
plt.show()
