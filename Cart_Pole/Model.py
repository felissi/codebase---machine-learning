import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv

ACTION_SIZE = 2

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

# https://github.com/lmarza/CartPole-CNN/blob/main/RelazioneProgettoIntelligenzaArtificiale_LucaMarzari.pdf
class QNetwork(torch.nn.Module):
    def __init__(self, state_size: int=None, action_size=ACTION_SIZE):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 64,kernel_size=5, stride=3) # [B,4,160,240]->[B,64,52,79]
        self.conv2 = nn.Conv2d(64,64,kernel_size=4, stride=2) # [B,64,52,79]->[B,64,25,38]
        self.conv3 = nn.Conv2d(64,64,kernel_size=3, stride=1) # [B,64,25,38]->[B,64,23,36]
        self.flatten = nn.Flatten() # [B,64,23,36]->[B, 52992]
        self.fc1 = nn.Linear(64*23*36, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, x:torch.tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

        

