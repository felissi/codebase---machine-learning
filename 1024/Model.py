import torch.nn as nn
import torch

class QNetwork(nn.Module):
    """ Agent Policy Network Model """
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.flatten = torch.nn.Flatten()
        
    def forward(self, x:torch.Tensor):
        """ state -> action values """
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x
