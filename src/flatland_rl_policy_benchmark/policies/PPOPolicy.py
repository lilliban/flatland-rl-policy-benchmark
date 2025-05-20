import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return torch.softmax(self.policy_head(x), dim=-1), self.value_head(x)

class PPOPolicy:
    def __init__(self, state_size, action_size, params):
        self.device = torch.device(params.get("device", "cpu"))
        self.gamma = params.get("gamma", 0.99)
        self.eps_clip = params.get("eps_clip", 0.2)
        self.lr = params.get("lr", 1e-4)
        self.k_epochs = params.get("k_epochs", 4)

        self.ac = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.lr)

        self.memory = []
        self.metrics_dict = {}

    def select_action(self, obs):
        state = torch.from_numpy(obs).float().to(self.device)
        probs, value = self.ac(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.last_state = state
        self.last_action = action
        self.last_log_prob = log_prob
        self.last_value = value

        return action.item(), log_prob.item()

    def step(self, reward, done):
        self.memory.append((
            self.last_state,
            self.last_action,
            self.last_log_prob,
            self.last_value,
            reward,
            done
        ))

    def finish_episode(self):
        if not self.memory:
            return

        states, actions, log_probs, values, rewards, dones = zip(*self.memory)
        returns = []
        discounted = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                discounted = 0
            discounted = r + self.gamma * discounted
            returns.insert(0, discounted)

        returns = torch.tensor(returns).float().to(self.device)
        states = torch.stack(states)
        actions = torch.stack(actions).to(self.device)
        old_log_probs = torch.stack(log_probs).to(self.device)
        values = torch.stack(values).squeeze().to(self.device)

        advantages = returns - values.detach()

        for _ in range(self.k_epochs):
            probs, new_values = self.ac(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2).mean() \
                   + 0.5 * (returns - new_values.squeeze()).pow(2).mean() \
                   - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Salva metriche
        self.metrics_dict = {
            "total_reward": sum(rewards),
            "episode_length": len(self.memory),
            "survived_steps": len([1 for d in dones if not d]),
            "collisions": len([1 for d in dones if d])
        }

        self.memory = []

    def metrics(self):
        return self.metrics_dict.copy() if self.metrics_dict else {
            "total_reward": 0,
            "episode_length": 0,
            "survived_steps": 0,
            "collisions": 0
        }