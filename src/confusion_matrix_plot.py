import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Confusion matrix data
conf_matrix = np.array([
    [201, 50, 11, 23, 2],
    [0, 47, 5, 8, 0],
    [4, 58, 86, 13, 0],
    [0, 19, 4, 8, 0],
    [2, 28, 5, 12, 0]
])

# Class labels
labels = ['0', '1', '2', '3', '4']

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Figure 3. Confusion Matrix (Validation Set)')
plt.tight_layout()

# Save the figure
plt.savefig('visuals/confusion_matrix.png')  # Ensure 'visuals/' folder exists
plt.show()
