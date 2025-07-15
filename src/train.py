import os
import torch
import yaml
import json
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import nn, optim
from model import SimpleCNN  # Import your model

# Load training hyperparameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train"]

# Directories
train_dir = '/home/cyril-saju/Documents/OrthoFx/dvc_cnn/data/Butterfly/train'
val_dir   = '/home/cyril-saju/Documents/OrthoFx/dvc_cnn/data/Butterfly/val'

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Datasets and loaders
train_dataset = ImageFolder(root=train_dir, transform=transform)
val_dataset   = ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)

# Model, loss, optimizer
num_classes = len(train_dataset.classes)
model = SimpleCNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=params["lr"])

# Training loop
for epoch in range(params["epochs"]):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Validation accuracy
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_accuracy = correct / total

# Save model
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/model.pt")

# Save validation accuracy as metric
with open("metrics.json", "w") as f:
    json.dump({"val_accuracy": round(val_accuracy, 4)}, f)

print(f"Validation Accuracy: {val_accuracy:.4f}")
