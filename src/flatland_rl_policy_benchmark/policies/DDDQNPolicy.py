import random, numpy as np, torch, torch.nn.functional as F
from flatland_rl_policy_benchmark.policies.DuelingQNetwork import DuelingQNetwork
from flatland_rl_policy_benchmark.utils.ReplayBuffer import ReplayBuffer

class DDDQNPolicy:
    def __init__(self, state_size, action_size, params):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_net  = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net = DuelingQNetwork(state_size, action_size).to(self.device)
        self.optimizer  = torch.optim.Adam(self.local_net.parameters(), lr=params["learning_rate"])
        self.memory     = ReplayBuffer(params["buffer_size"], params["batch_size"], self.device)
        self.gamma = params["gamma"]
        self.tau   = params["tau"]
        self.epsilon, self.eps_min, self.eps_decay = 1.0, 0.05, 0.995

    def select_action(self, agent_id, obs, eps=None):
        eps = self.epsilon if eps is None else eps
        state = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        if random.random() < eps:
            return random.randrange(self.local_net.adv_stream[-1].out_features)
        with torch.no_grad():
            q = self.local_net(state)
        return q.argmax().item()

    def step(self, state, action, reward, next_state, done):
        self.memory.add(0, state, action, reward, next_state, done)
        if len(self.memory) > self.memory.batch_size:
            self.learn()

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample()
        next_actions = self.local_net(next_states).argmax(1, keepdim=True)
        Q_targets_next = self.target_net(next_states).gather(1, next_actions)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.local_net(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        # soft update
        for t,p in zip(self.target_net.parameters(), self.local_net.parameters()):
            t.data.copy_(self.tau*p.data + (1.0-self.tau)*t.data)
        self.epsilon = max(self.eps_min, self.epsilon*self.eps_decay)
