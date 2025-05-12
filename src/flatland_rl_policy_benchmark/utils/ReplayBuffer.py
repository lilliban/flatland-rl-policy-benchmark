import numpy as np

class ReplayBuffer:
    """
    Buffer circolare per Double DQN, con memorizzazione pre-allocata
    e sampling vettoriale O(batch_size) senza riallocazioni.
    """
    def __init__(self, buffer_size: int, batch_size: int, state_dim: int):
        self.buffer_size = buffer_size
        self.batch_size  = batch_size

        # Pre-allocazione degli array NumPy
        self.states      = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions     = np.zeros(buffer_size,          dtype=np.int64)
        self.rewards     = np.zeros(buffer_size,          dtype=np.float32)
        self.dones       = np.zeros(buffer_size,          dtype=bool)

        self.ptr  = 0      # indice circolare di scrittura
        self.size = 0      # numero di elementi validi

    def add(self,
            state:      np.ndarray,
            action:     int,
            reward:     float,
            next_state: np.ndarray,
            done:       bool):
        """Inserisce una nuova transizione nel buffer (circolare)."""
        idx = self.ptr
        self.states[idx]      = state
        self.next_states[idx] = next_state
        self.actions[idx]     = action
        self.rewards[idx]     = reward
        self.dones[idx]       = done

        # aggiorna puntatore e size
        self.ptr  = (idx + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        """
        Campiona batch_size transizioni in modo vettoriale.
        Restituisce tuple di array NumPy pronti per il learning.
        """
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs]
        )

    def __len__(self) -> int:
        """Numero corrente di elementi validi nel buffer."""
        return self.size
