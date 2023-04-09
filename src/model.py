import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # 16 x 128 x 128
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), # 16 x 128 x 128
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 16 x 64 x 64
            # nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # 32 x 64 x 64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # 32 x 64 x 64
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 32 x 32 x 32
            # nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 64 x 32 x 32
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 64 x 16 x 16
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64*16*16, 256), 
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(256, 151),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.flatten(out)
        out = self.fc(out)
        
        return out
    

if __name__ == '__main__':
    print(NeuralNetwork())