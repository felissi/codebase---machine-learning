import torch
import numpy as np
import random
from collections import namedtuple, deque
from typing import Iterable
from Model import QNetwork as QNetwork

BUFFER_SIZE         = int(1e5)
BATCH_SIZE          = 64
GAMMA               = 0.99  # discount factor
TAU                 = 1e-3  # soft update of target parameter
LEARNING_RATE       = 5e-4
UPDATE_EVERY        = 10    # how often to update the local
TARGET_UPDATE_EVERY = 50    # how often to update the target

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ReplayBuffer:
    """ Fixed size buffer to store experience tuples """

    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
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
    
class Agent:
    def __init__(self, state_size: int, action_size: int, learning_rate: float, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = torch.optim.Adam(
            self.qnetwork_local.parameters(), lr=learning_rate)
        # replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size)

        self.time_step = 0
        self.eps = 0.0
        self.gamma = 0.9

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.time_step += 1
        

    def learn_from_experience(self):
        experience = self.memory.sample()
        return self.learn(experience)

    def q_value(self, state, eps=0.0, train=True)->torch.Tensor:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        """ (w,h)->(1,w,h) Add one dimension to the state, as the nn expect a batch.  """
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state).to(device)
        self.qnetwork_local.train()
        return action_values
    
    def decide(self, action_values:torch.Tensor, eps=0.0)->int:
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()) #  "addmm_cuda" not implemented for 'Long'
        else:
            return random.choice(np.arange(self.action_size))

    def act(self, state, eps=0.0, train=True)->int:
        action_values = self.q_value(state, eps=eps, train=train)
        return self.decide(action_values, eps=eps)

    def learn(self, experience: Iterable[torch.Tensor]):
        """ Update parameters using batch of experience tuples """
        q_current, q_targets = self._double_dqn(experience)
        # Compute the loss and gradient
        loss = torch.nn.functional.mse_loss(q_current, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def _dqn(self, experience: Iterable[torch.Tensor])->Iterable[torch.Tensor]:
        states, actions, rewards, next_states, dones = experience
        rewards = rewards.reshape(self.batch_size,1)
        dones = dones.reshape(self.batch_size,1)
        q_targets_next = self.qnetwork_target( next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards+self.gamma*q_targets_next*(1 - dones)
        q_current = self.qnetwork_local(states).gather(1, actions.type(torch.int64).unsqueeze(1))
        return (q_current, q_targets)
    
    def _double_dqn(self, experience: Iterable[torch.Tensor])->Iterable[torch.Tensor]:
        states, actions, rewards, next_states, dones = experience
        rewards = rewards.reshape(self.batch_size,1)
        dones = dones.reshape(self.batch_size,1)
        action_q_local_next = torch.argmax(self.qnetwork_local(next_states),-1)
        q_targets_next = self.qnetwork_target(next_states).gather(1, action_q_local_next.type(torch.int64).unsqueeze(1))
        q_targets = rewards+self.gamma*q_targets_next*(1 - dones)
        q_current = self.qnetwork_local(states).gather(1, actions.type(torch.int64).unsqueeze(1))
        return (q_current, q_targets)


    def state_to_features(self, state: np.ndarray):
        return torch.from_numpy(state.flatten()).to(device).float().unsqueeze(0)

    def soft_update(self):
        self._soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    @staticmethod
    def _soft_update(local_model:torch.nn.Module, target_model:torch.nn.Module, tau: float):
        """ θ_target = τ*θ_local + (1 - τ)*θ_target 
        copy the weights of the local model to the target model
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data+(1.0-tau)*target_param.data)