import os
import json
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import SimpleCNN
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

test_dir = params["evaluate"]["test_dir"]


model_path = '/home/cyril-saju/Documents/OrthoFx/dvc_cnn/model/model.pt'

# Transform (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset and DataLoader
test_dataset = ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
num_classes = len(test_dataset.classes)
model = SimpleCNN(num_classes=num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# Evaluate
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total

# Save metrics
with open("test_metrics.json", "w") as f:
    json.dump({"test_accuracy": round(test_accuracy, 4)}, f)

print(f"Test Accuracy: {test_accuracy:.4f}")
