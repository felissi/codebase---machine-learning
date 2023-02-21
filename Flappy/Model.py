import torch.nn as nn
import torch

IMG_WIDTH  = 80
IMG_HEIGHT = 80
IMG_DEPTH  = 4

class QNetwork(nn.Module):
    """ Agent Policy Network Model """
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(294912, 16)
        self.fc2 = nn.Linear(16, action_size)
        
    def forward(self, x:torch.Tensor):
        """ state -> action values """
        x = x.permute(0, 3, 2, 1) # flip [B, W, H, C] -> [B, C, H, W]
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2) # -> [B, 32, H/2, W/2]
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        # x = nn.functional.relu(x)
        return x
