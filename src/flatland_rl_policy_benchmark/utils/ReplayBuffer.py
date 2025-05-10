import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.memory = []

    def add(self, agent_id, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states).to(self.device),
            torch.LongTensor(actions).unsqueeze(1).to(self.device),
            torch.FloatTensor(rewards).unsqueeze(1).to(self.device),
            torch.FloatTensor(next_states).to(self.device),
            torch.FloatTensor(dones).unsqueeze(1).to(self.device),
        )

    def __len__(self):
        return len(self.memory)
