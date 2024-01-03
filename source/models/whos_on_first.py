import torch.nn as nn


class WhosOnFirst(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(8),
            nn.Hardswish(),

            nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8),
            nn.Conv2d(8, 16, kernel_size=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.Hardswish(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.Hardswish(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.Hardswish(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.Hardswish(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128*(8*4), 45))
    
    def forward(self, x):
        return self.layers(x)
