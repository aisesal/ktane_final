import torch.nn as nn


class ComplicatedWires(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(8),
            nn.Hardswish(),

            nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8),
            nn.Conv2d(8, 16, kernel_size=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.Hardswish(),
            nn.MaxPool2d(2, ceil_mode=True),

            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.Hardswish(),
            nn.MaxPool2d(2, ceil_mode=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.Hardswish(),
            nn.MaxPool2d(2, ceil_mode=True),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.Hardswish())

        self.color_head = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(2, ceil_mode=True),

                nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),
                nn.Conv2d(128, 256, kernel_size=1, bias=False),
                nn.InstanceNorm2d(256),
                nn.Hardswish(),            
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),

                nn.Linear(256*(4**2), 7)) for _ in range(6)])
    
    def forward(self, x):
        features = self.layers(x)
        colors = [m(features) for m in self.color_head]
        return colors
