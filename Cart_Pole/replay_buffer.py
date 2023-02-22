import torch
import numpy as np
import random
from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """ Fixed size buffer to store experience tuples """

    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = namedtuple('experience', field_names=[
                                     'state', 'action', 'reward', 'next_state', 'done'])
        self.memory: deque[self.experience] = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """  """
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(
            np.stack([e.state for e in experiences if e is not None], axis=0)).to(device).float()
        actions = torch.from_numpy(
            np.stack([e.action for e in experiences if e is not None], axis=0)).to(device).float()
        rewards = torch.from_numpy(
            np.stack([e.reward for e in experiences if e is not None], axis=0)).to(device).float()
        next_states = torch.from_numpy(
            np.stack([e.next_state for e in experiences if e is not None], axis=0)).to(device).float()
        dones = torch.from_numpy(np.stack(
            [e.done for e in experiences if e is not None], axis=0).astype(np.uint8)).to(device).float()
        return self.experience(states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)