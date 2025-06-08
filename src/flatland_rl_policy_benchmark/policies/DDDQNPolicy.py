import torch
import torch.nn.functional as F
import numpy as np
from flatland_rl_policy_benchmark.policies.DuelingQNetwork import DuelingQNetwork
from flatland_rl_policy_benchmark.utils.ReplayBuffer import ReplayBuffer

#hai due teri una targhet e una locale
class DDDQNPolicy:
    def __init__(self, state_size, action_size, config):
        """Double Dueling DQN Policy with robust config handling."""
        self.state_size = state_size
        self.action_size = action_size
        
        #usi if isistance per usare la classe sia con un dizionario che con un oggetto
        if isinstance(config, dict):
            device = config.get("device", "cpu")
            gamma = config.get("gamma", 0.99)
            tau = config.get("tau", 0.001)
            learning_rate = config.get("learning_rate", 1e-3)
            buffer_size = config.get("buffer_size", int(1e5))
            batch_size = config.get("batch_size", 64)
            epsilon_start = config.get("epsilon_start", 1.0)
            epsilon_min = config.get("epsilon_min", 0.01)
            epsilon_decay = config.get("epsilon_decay", 0.995)
            learn_every = config.get("learn_every", 1)
            prioritized = False
            alpha = config.get("prioritized_replay_alpha", 0.5)
            target_update_freq = config.get("target_update_freq", 1000)
            gradient_clip = config.get("gradient_clip", 1.0)
            hidden_dim = config.get("hidden_dim", 64)
            dropout_prob = config.get("dropout_prob", 0.1)
        else:
            device = getattr(config, "device", "cpu")
            gamma = getattr(config, "gamma", 0.99)
            tau = getattr(config, "tau", 0.001)
            learning_rate = getattr(config, "learning_rate", 1e-3)
            buffer_size = getattr(config, "buffer_size", int(1e5))
            batch_size = getattr(config, "batch_size", 64)
            epsilon_start = getattr(config, "epsilon_start", 1.0)
            epsilon_min = getattr(config, "epsilon_min", 0.01)
            epsilon_decay = getattr(config, "epsilon_decay", 0.995)
            learn_every = getattr(config, "learn_every", 1)
            prioritized = getattr(config, "prioritized_replay", False)
            alpha = getattr(config, "prioritized_replay_alpha", 0.5)
            target_update_freq = getattr(config, "target_update_freq", 1000)
            gradient_clip = getattr(config, "gradient_clip", 1.0)
            hidden_dim = getattr(config, "hidden_dim", 64)
            dropout_prob = getattr(config, "dropout_prob", 0.1)

        # Set parameters as attributes
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learn_every = learn_every
        self.prioritized = prioritized
        self.alpha = alpha
        self.target_update_freq = target_update_freq
        self.gradient_clip = gradient_clip

        #La rete local_network è quella che apprende
        self.local_network = DuelingQNetwork(state_size, action_size, hidden_dim, dropout_prob).to(self.device)
        #La rete target_network è quella stabile
        self.target_network = DuelingQNetwork(state_size, action_size, hidden_dim, dropout_prob).to(self.device)
        self.optimizer = torch.optim.Adam(self.local_network.parameters(), lr=self.learning_rate)

        # Initialize replay memory (handle prioritized vs regular)
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.device)

        self.t_step = 0
        # Ensure target network starts with same weights as local network
        self.hard_update(self.local_network, self.target_network)

    #DEFINISCE COME SCEGLEIRE L'AZIONE DA FARE IN OGNI STATO, la scelta si basa sulla tecnica epsilon-greedy
    #non serve passargli la espinol perché la tiene come parametro della classe e quando chiami agent.update_epsilon() lui aggiorna self.epsilon

    def select_action(self, state):
        #nota: Non si aggiorna epsilon dentro select_action perché select_action deve solo “decidere che azione fare ADESSO”, non deve cambiare i parametri della policy
        if state is None:
            return np.random.randint(self.action_size)
       
        # fa una "pulizia" perchè la rete local_network lavora solo con array     
        if isinstance(state, dict):
            if 'features' in state:
                state = state['features']  
            else:
                state = np.concatenate(list(state.values()))
        elif not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
       
        #è un controllo di sicurezza che vede se epsilon è un numero non valido allora imposta a 1.0 (esploro sempre)
        try:
            epsilon = float(self.epsilon)
        except TypeError:
            epsilon = 1.0  

        #usare np.random.rand() è un modo di simulare il sorteggio probabilistico, se esce un numero piccolo → esploro → azione casuale    
        if np.random.rand() <= epsilon:
            #esploro
            return np.random.randint(self.action_size)
        # non sono entrato nell'if quindi sfrutto       
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.local_network(state_tensor)
        return q_values.argmax().item()

    # serve per abbassare epsilon ma mai sotto il minimo, lo fai perchè all'inizio vuoi esplorare quindi epsilon alto poi vuoi sfruttare quindi epsilon basso. 
    # NOTA:  non vuoi mai smettere del tutto di esplorare quindi c'è epsilon_min 
    def update_epsilon(self): 
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # aggiorna la memoria, cioè il ReplayBuffer
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1
        # verifica se ho abbastanza esperienze in memoria per fare un batch di training con if len(self.memory) >= self.batch_size
        # e poi il secondo controllo vede quanti passi sono passati dall'ultimo learn, facciamo questo perchè lui non impara sempre atrimenti è instabile 
        if len(self.memory) >= self.batch_size and self.t_step % self.learn_every == 0:
            self.learn()
          

    def learn(self):
       # Preleva un batch di esperienze dalla memoria in maniera casuale
        batch_data = self.memory.sample() #batch di esperinze
      
        # vai a capire se il replay buffer è prioritario o normale, è UN CONTROLLO DI SICUREZZA, posso toglierlo però metti che lo faccio  
        if isinstance(batch_data, tuple) and len(batch_data) == 3:
            (states, actions, rewards, next_states, dones), indices, weights = batch_data
        else:
            # Non-prioritized replay returns just the batch
            states, actions, rewards, next_states, dones = batch_data
            # Use uniform weights if not prioritized
            weights = torch.ones(len(states), device=self.device)

        #sistema la froma dei tensori, per far si che abbiano una forma corretta per il batch processing
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)

        # per i next_states che ho preso dal batch, calcolami i Q-values con la rete target_network
        with torch.no_grad():
            q_next = self.target_network(next_states)
       
        # per ogni next_state nel batch vado a chiedere alla local_network quale azione scegliere
        # argmax(dim=1) significa che per ogni riga (ogni next_state), prendo l'indice dell'azione col Q-value più alto, cioè la "best action"
        # poi faccio unsqueeze(1) per aggiungere una dimensione
        best_actions = self.local_network(next_states).argmax(dim=1).unsqueeze(1)
       
        #vai a prendere solo i q-values corrispondenti alle best actions che la local network ha scleto    
        q_targets_next = q_next.gather(1, best_actions)
       
        # questo mi dice gamma * q_targets_next → quanto posso guadagnare in futuro, facendo la best action scelta dalla local_network
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # calcolo il Q-value atteso per le azioni fatte
        q_expected = self.local_network(states).gather(1, actions)

        # Compute TD error and weighted MSE loss
        td_errors = q_targets - q_expected
        loss = (weights * td_errors**2).mean()

        # Azzeramento dei gradienti precedenti.
        # In PyTorch, i gradienti dei parametri si accumulano per default.
        # Per evitare che i gradienti delle iterazioni precedenti interferiscano con il nuovo aggiornamento,
        # è necessario azzerarli prima di ogni backward pass.
        self.optimizer.zero_grad()
        
        # Calcolo come dovresti cambiare ogni peso
        # loss.backward() calcola i gradienti della funzione di loss rispetto a tutti i parametri della rete (i pesi).
        # I gradienti vengono calcolati sfruttando la derivata della loss rispetto ai pesi
        # e memorizzati all'interno del tensore .grad associato a ciascun parametro.
        loss.backward()
        
        # Clipping dei gradienti per migliorare la stabilità del training, cioè vai a limitare il cambiamento per evitare che la rete impazzisca
        # In alcune situazioni i gradienti possono assumere valori molto grandi, causando aggiornamenti instabili (exploding gradients).
        # Il clipping limita la norma dei gradienti a un valore massimo specificato (gradient_clip), impedendo aggiornamenti troppo bruschi.
        torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), self.gradient_clip)
       
        # Aggiornamento effettivo dei parametri della rete.
        # optimizer.step() applica l'algoritmo di ottimizzazione (Adam in questo caso) per aggiornare i pesi della rete,
        # utilizzando i gradienti calcolati nel passaggio precedente.
        # È in questo momento che i pesi della rete vengono effettivamente modificati, permettendo alla rete di apprendere.
        self.optimizer.step()

        # Aggiorna la rete target periodicamente, che serve per calcolar eil q targhet e non viene aggiornata semrpe per rendera piu stabile
        if self.t_step % self.target_update_freq == 0:
            self.soft_update(self.local_network, self.target_network)
        return loss.item()

    #serve per standardizzare il training delle reti
    def soft_update(self, local_model, target_model):
        """Soft update model parameters: target = tau*local + (1-tau)*target."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
   
    # la usi all'inzio al fine di avere che che la target_network parta uguale alla local_network
    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def adjust_learning_rate(self, factor):
        """Adjust the learning rate by a given factor."""
        self.learning_rate *= factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    def save_model(self, path):
        """Save the local network weights to a file."""
        torch.save(self.local_network.state_dict(), path)

    def load_model(self, path):
        """Load weights from a file into the local and target networks."""
        self.local_network.load_state_dict(torch.load(path, map_location=self.device), strict=False)
        # Ensure target network is in sync with local after loading
        self.hard_update(self.local_network, self.target_network)