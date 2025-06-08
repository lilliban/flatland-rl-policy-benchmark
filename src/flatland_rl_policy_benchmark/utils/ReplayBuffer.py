import random
import numpy as np
from collections import deque, namedtuple
import torch

# memorizza transizioni, è dove vengono salvate le esperienze dell'agente
# quando poi viene fatto il learn non impara solo dall'ultima esperienza ma da un batch di esperinze prese a caso
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        # viene implementata con una deque, cioè una CODA con doppia estremità. hai che quando arrivi a buffer_size se aggiungi un altro elemnto il più VECCHIO vine automaticamente rimosso
        self.memory = deque(maxlen=buffer_size)
        # Numero di esperinze che prelevo ogni volta che faccio il sample ()
        self.batch_size = batch_size
        self.device = device
        self.experience = namedtuple("Experience", 
            field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences])).float().to(self.device)
        actions = torch.tensor([e.action for e in experiences]).long().to(self.device)
        rewards = torch.tensor([e.reward for e in experiences]).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.tensor([e.done for e in experiences]).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    # ti permette di sapere quante esperienze ci sono nel buffer in quel momento
    def __len__(self):
        return len(self.memory)
