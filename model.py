import torch
import torch.nn as nn
import torch.nn.functional as F


class NeutronDetectorCNN(nn.Module):
    def __init__(self, image_size):
        super(NeutronDetectorCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * image_size * image_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
