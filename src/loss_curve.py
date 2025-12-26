import matplotlib.pyplot as plt

# Data from your training logs
epochs = list(range(1, 11))
train_loss = [94.23, 90.05, 88.95, 85.72, 83.66, 82.60, 79.19, 80.45, 76.62, 76.54]
val_loss = [22.14, 21.95, 21.02, 19.63, 19.19, 18.71, 19.06, 18.59, 18.32, 17.87]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
plt.savefig('visuals/loss_curve.png')  # Make sure 'visuals/' folder exists
plt.show()
