import torch
from torch import nn

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
#             nn.Dropout(p=0.2),
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), 
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
#             nn.Dropout(p=0.2),
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
#             nn.Dropout(p=0.2),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
#         )
#         self.flatten = nn.Flatten()
#         self.fc = nn.Sequential(
#             nn.Linear(256*4*4, 256), 
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(256, 151),
#             # nn.Softmax(dim=1)
#         )

#     def forward(self, x):
#         out = self.conv(x)
#         out = self.flatten(out)
#         out = self.fc(out)
        
#         return out    

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128*4*4, 256), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
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