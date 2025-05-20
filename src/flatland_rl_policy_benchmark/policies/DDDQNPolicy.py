import random
import numpy as np
import torch
import torch.nn.functional as F
from flatland_rl_policy_benchmark.policies.DuelingQNetwork import DuelingQNetwork
from flatland_rl_policy_benchmark.utils.ReplayBuffer import ReplayBuffer

class DDDQNPolicy:
    def __init__(self, state_size, action_size, params):
        self.device = torch.device(params.get("device", "cpu"))
        self.gamma = params.get("gamma", 0.99)
        self.tau = params.get("tau", 1e-3)
        self.epsilon = params.get("epsilon_start", 1.0)
        self.epsilon_min = params.get("epsilon_min", 0.01)
        self.epsilon_decay = params.get("epsilon_decay", 0.995)
        self.batch_size = params.get("batch_size", 64)
        self.learn_every = params.get("learn_every", 4)
        self.step_count = 0

        self.local_net = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.local_net.state_dict())

        lr = params.get("learning_rate", 1e-3)
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)

        buffer_size = params.get("buffer_size", int(1e4))
        self.memory = ReplayBuffer(buffer_size, self.batch_size, state_size)

        self.last_total_reward = 0
        self.last_episode_length = 0
        self.last_survived_steps = 0
        self.last_collisions = 0
        self.episode_reward = 0
        self.episode_steps = 0
        self.episode_collisions = 0

    def select_action(self, obs, eps=None):
        eps = self.epsilon if eps is None else eps
        if random.random() < eps:
            return random.randrange(self.local_net.advantage_stream[-1].out_features)
        state_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.local_net(state_t)
        if hasattr(self, "valid_actions") and self.valid_actions is not None:
            q_vals[0][[i for i in range(len(q_vals[0])) if i not in self.valid_actions]] = -float("inf")
        return int(q_vals.argmax().item())


    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.step_count += 1

        self.episode_reward += reward
        self.episode_steps += 1
        if done:
            self.episode_collisions += 1

        if len(self.memory) >= self.batch_size and (self.step_count % self.learn_every) == 0:
            self.learn()

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.last_total_reward = self.episode_reward
            self.last_episode_length = self.episode_steps
            self.last_survived_steps = self.episode_steps - self.episode_collisions
            self.last_collisions = self.episode_collisions

            self.episode_reward = 0
            self.episode_steps = 0
            self.episode_collisions = 0

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample()

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().unsqueeze(1).to(self.device)

        q_local_next = self.local_net(next_states).detach()
        next_actions = q_local_next.argmax(dim=1, keepdim=True)
        q_target_next = self.target_net(next_states).detach().gather(1, next_actions)
        q_targets = rewards + (self.gamma * q_target_next * (1 - dones))

        q_expected = self.local_net(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for tp, lp in zip(self.target_net.parameters(), self.local_net.parameters()):
            tp.data.copy_(self.tau * lp.data + (1.0 - self.tau) * tp.data)

    def metrics(self):
        return {
            "total_reward": self.last_total_reward,
            "episode_length": self.last_episode_length,
            "survived_steps": self.last_survived_steps,
            "collisions": self.last_collisions
        }
