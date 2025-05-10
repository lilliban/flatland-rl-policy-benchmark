import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_size, hidden), nn.ReLU())
        self.actor  = nn.Linear(hidden, action_size)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.shared(x)
        return F.softmax(self.actor(x), dim=-1), self.critic(x)

class PPOPolicy:
    def __init__(self, state_size, action_size, params):
        self.gamma = params["gamma"]
        self.eps_clip = params["eps_clip"]
        self.lr = params["learning_rate"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        probs, _ = self.ac(state.unsqueeze(0))
        dist = Categorical(probs)
        action = dist.sample().item()
        return action, dist.log_prob(torch.tensor(action)).unsqueeze(0)

    def learn(self, trajectories):
        # trajectories: list of (state, action, logprob, reward, next_state, done)
        # calcola advantage, policy loss e value loss, poi backward()
        pass  # implementa il tuo ciclo di ottimizzazione qui
