import torch
import torch.nn as nn
import torch.nn.functional as F

#va a stimare il valore dell'azione, riceve le osservazioni, le elabora e restituisce il q value
class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim, dropout_prob=0.2):
        super(DuelingQNetwork, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size)
        )

    def forward(self, state):
        features = self.feature(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals
