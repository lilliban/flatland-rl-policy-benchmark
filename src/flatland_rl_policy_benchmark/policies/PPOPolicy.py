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
        states, actions, old_logprobs, rewards, next_states, dones = zip(*trajectories)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_logprobs = torch.cat(old_logprobs).detach()
        returns = self._compute_returns(rewards, dones)

        for _ in range(4):  # epochs
            probs, values = self.ac(states)
            dist = Categorical(probs)
            logprobs = dist.log_prob(actions)
            advantages = returns - values.squeeze()

            ratio = torch.exp(logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.squeeze(), returns)

            loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _compute_returns(self, rewards, dones):
        R = 0
        returns = []
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        return torch.FloatTensor(returns).to(self.device)
