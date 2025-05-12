import random
import numpy as np
import torch
import torch.nn.functional as F

from flatland_rl_policy_benchmark.policies.DuelingQNetwork import DuelingQNetwork
from flatland_rl_policy_benchmark.utils.ReplayBuffer import ReplayBuffer

class DDDQNPolicy:
    """
    Double DQN con rete dueling e soft-update del target network.
    """
    def __init__(self,
                 state_size:  int,
                 action_size: int,
                 params:      dict):
        # Device e iperparametri
        self.device        = torch.device(params.get("device", "cpu"))
        self.gamma         = params.get("gamma",        0.99)
        self.tau           = params.get("tau",          1e-3)
        self.epsilon       = params.get("epsilon_start",1.0)
        self.epsilon_min   = params.get("epsilon_min",  0.01)
        self.epsilon_decay = params.get("epsilon_decay",0.995)
        self.batch_size    = params.get("batch_size",   64)
        self.learn_every   = params.get("learn_every",  4)
        self.step_count    = 0

        # Reti locale e target
        self.local_net  = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.local_net.state_dict())

        # Ottimizzatore
        lr = params.get("learning_rate", 1e-3)
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)

        # Replay buffer pre-allocato
        buffer_size = params.get("buffer_size", int(1e5))
        self.memory  = ReplayBuffer(buffer_size, self.batch_size, state_size)

    def select_action(self, obs: np.ndarray, eps: float = None) -> int:
        """
        ε-greedy action selection:
        esplora con prob. ε, altrimenti sfrutta la rete locale.
        """
        eps = self.epsilon if eps is None else eps
        if random.random() < eps:
            return random.randrange(self.local_net.advantage_stream[-1].out_features)
        state_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.local_net(state_t)
        return int(q_vals.argmax().item())

    def step(self,
             state:      np.ndarray,
             action:     int,
             reward:     float,
             next_state: np.ndarray,
             done:       bool):
        """
        Registra la transizione, aggiorna contatore e lancia learning a intervalli.
        """
        self.memory.add(state, action, reward, next_state, done)
        self.step_count += 1

        if len(self.memory) >= self.batch_size and \
           (self.step_count % self.learn_every) == 0:
            self.learn()

        if done:
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)

    def learn(self):
        """
        Double-DQN update:
        selezione con local_net, valutazione con target_net.
        """
        states, actions, rewards, next_states, dones = self.memory.sample()

        states      = torch.from_numpy(states).float().to(self.device)
        actions     = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards     = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones       = torch.from_numpy(dones.astype(np.uint8)).float().unsqueeze(1).to(self.device)

        q_local_next  = self.local_net(next_states).detach()
        next_actions  = q_local_next.argmax(dim=1, keepdim=True)
        q_target_next = self.target_net(next_states).detach().gather(1, next_actions)
        q_targets     = rewards + (self.gamma * q_target_next * (1 - dones))

        q_expected = self.local_net(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for tp, lp in zip(self.target_net.parameters(), self.local_net.parameters()):
            tp.data.copy_(self.tau * lp.data + (1.0 - self.tau) * tp.data)
