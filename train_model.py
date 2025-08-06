import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from dataset_loader import FER2013Dataset   # Import the Dataset class
from model import EmotionClassifierCNN      # Import the Model class

# --- 1. Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Hyperparameters ---
num_classes = 7
learning_rate = 0.001   # A common starting learning rate for Adam
num_epochs = 10
batch_size = 64         # should match the one in dataset_loader.py
data_root = './fer2013'

# --- 3. Data Preparation ---
train_transforms = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_val_transforms = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Create Dataset and DataLoader instances
train_dataset = FER2013Dataset(root_dir=data_root, phase='train', transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_dataset = FER2013Dataset(root_dir=data_root, phase='test', transform=test_val_transforms)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# --- 4. Model, Loss, and Optimizer Initialization ---
model = EmotionClassifierCNN(num_classes=num_classes).to(device) # Move model to device
criterion = nn.CrossEntropyLoss() # Loss function for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer

# --- 5. Training Loop ---
print("\n--- Starting Training ---")
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    # --- Training Phase ---
    model.train() # Set the model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device) # Move images to the specified device (CPU/GPU)
        labels = labels.to(device) # Move labels to the specified device

        # Zero the parameter gradients (clear accumulated gradients from previous steps)
        optimizer.zero_grad()

        # Forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)

        # Calculate loss: compare model output with true labels
        loss = criterion(outputs, labels)

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Optimizer step: perform a single optimization step (parameter update)
        optimizer.step()

        running_loss += loss.item() * images.size(0) # Accumulate batch loss
        
        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1) # Get the index of the max log-probability
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_train_loss = running_loss / len(train_loader.dataset)
    epoch_train_accuracy = (correct_train / total_train) * 100
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accuracy)

    # --- Validation Phase ---
    model.eval() # Set the model to evaluation mode
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    # Disable gradient calculation for validation (saves memory and speeds up)
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_val_accuracy = (correct_val / total_val) * 100
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}: "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}% | "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.2f}%")

print("\n--- Training Finished ---")

# --- 6. Plotting Learning Curves (Optional but Recommended) ---
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nTraining complete and learning curves plotted!")