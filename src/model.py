import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0),  # (B, 3, 32, 32) -> (B, 16, 30, 30)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # -> (B, 16, 15, 15)

            nn.Conv2d(16, 32, kernel_size=3),                      # -> (B, 32, 13, 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                           # -> (B, 32, 6, 6)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                                          # -> (B, 32 * 6 * 6)
            nn.Linear(32 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

