import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int, state_dim: int):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_dim = state_dim

        print(f" Inizializzazione ReplayBuffer: buffer_size={buffer_size}, batch_size={batch_size}, state_dim={state_dim}")
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=bool)

        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        idx = self.ptr
        self.states[idx] = state
        self.next_states[idx] = next_state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done

        self.ptr = (idx + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

        # Logging per analisi
        if self.size % 5000 == 0 and self.ptr == 0:
            with open("replaybuffer.log", "a") as f:
                f.write(f" Buffer attualmente contiene {self.size} transizioni\n")

    def sample(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs]
        )

    def __len__(self):
        return self.size

    def get_diversity(self, sample_size=1000):
        sample_size = min(sample_size, self.size)
        if sample_size == 0:
            return 0
        idxs = np.random.choice(self.size, sample_size, replace=False)
        states = self.states[idxs]
        # Calcola la varianza media tra le feature (proxy di diversità)
        return np.mean(np.var(states, axis=0))
