import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),     # 224 -> 222
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                  # 222 -> 111

            nn.Conv2d(16, 32, kernel_size=3),    # 111 -> 109
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                  # 109 -> 54
        )

        # Dynamically compute the flattened size
        self._to_linear = None
        self._get_flatten_size()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def _get_flatten_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.features(dummy)
            self._to_linear = out.view(1, -1).shape[1]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

