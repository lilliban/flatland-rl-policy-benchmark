import os
import csv
import torch
import numpy as np
from tqdm import tqdm
from flatland.envs.observations import TreeObsForRailEnv
from flatland_rl_policy_benchmark.env.environment import EnvironmentBuilder
from flatland_rl_policy_benchmark.policies.DDDQNPolicy import DDDQNPolicy
from flatland_rl_policy_benchmark.utils.obs_utils import flatten_obs
import logging
#viene inizializzata la policy e per ogni episodio generiamo l'ambiente si esegue l'agente e lo si aggiorna

#scrive i messaggi in console
logger = logging.getLogger(__name__)

#serve per calcolare quanto un treno è lontano dalla stazione
def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def calculate_shaped_reward(old_pos, new_pos, target, action, arrived, collision, done):
    reward = 0
    
    # Calcolo distanza dal target
    if all(pos is not None for pos in [old_pos, new_pos, target]):
        #calcola quanto il treno si è avvicinato al target
        old_dist = manhattan(old_pos, target)
        new_dist = manhattan(new_pos, target)
        delta = old_dist - new_dist
        
        #si avvicina
        if delta > 0:
            reward += delta * 1.0  
        elif delta < 0:
            #si allontna
            reward -= abs(delta) * 2.0

    # Reward per completamento
    if arrived:
        reward += 10
    
    # Penalità per collisione
    if collision:
        reward -= 2
    
    # Stalling penalty
    #if steps_without_progress > 10 and not arrived:
    #    reward -= 5

    if action in [1, 2, 3, 4]:
        reward += 0.5

    # Standing still penalty
    if action == 0 and not arrived:
        reward -= 1
    
    #arrotondi a due cifre decimali
    return round(reward, 2)

# Crea l'ambiente Flatland con le dimensioni e gli agenti specificati
def create_environment(width, height, effective_agents, max_depth, seed):
    #calcola la dimenzione minima della mappa in funzione dei treni 
    min_required = int(np.ceil(np.sqrt(effective_agents) * 5))
    #aggiusti l'altezza e la larghezza perchè dici che vuoi la mappa che dia o della dimenzione richiesta, o la massima o comuunque mai sotto 25
    #
    adjusted_width = max(width, min_required, 25)
    adjusted_height = max(height, min_required, 25)

    logger.info(f"Creazione ambiente: {adjusted_width}x{adjusted_height}, {effective_agents} agenti")

    #usa l'helper per creare il RailEnv cioè l'ambiente di Flatland
    builder = EnvironmentBuilder(
        width=adjusted_width,
        height=adjusted_height,  
        n_agents=effective_agents,
        seed=seed,
        # è il tipo di osservazione, questa è quella ad albero
        obs_builder_object=TreeObsForRailEnv(max_depth=max_depth)
    )
    return builder.build()

#funzione chiamata dal main
#va ad addestrare una policy  DDDQNPolicy su un determinato nodo dell'albero evolutivo.
def train(round_start, n_rounds, branch_name, parent_path, save_dir, level=0, width=25, height=25):
    #fai in modo che man mano che aument ala profondità dell'albero aumenta la difficoltà
    #L0
    if level == 0:
        effective_agents = 2
        effective_width = width
        effective_height = height
        max_depth = 3
    #L1
    elif level == 1:
        effective_agents = 3
        effective_width = width
        effective_height = height
        max_depth = 3
    #L2
    else:
        effective_agents = 4
        effective_width = width
        effective_height = height
        max_depth = 3

    n_episodes = 100

    os.makedirs(save_dir, exist_ok=True)
    #salva i risultati 
    csv_path = os.path.join(save_dir, f"evolution_level{level}_{branch_name}.csv")
    branch_num = branch_name.split('_B')[-1]
    model_path = os.path.join(save_dir, f"best_model_L{level}_B{branch_num}.pt")

    all_rewards = []
    best_reward = -float('inf')

    # Inizializza l'ambiente
    env = create_environment(effective_width, effective_height, effective_agents, max_depth, round_start)

    obs, _ = env.reset(regenerate_rail=True, regenerate_schedule=True)
    first_agent = list(obs.keys())[0]
    state_size = flatten_obs(obs[first_agent], max_depth=max_depth).shape[0]
    action_size = 5

    agent = DDDQNPolicy(state_size, action_size, {
        # gamma mi dice quanto tenere conto delle ricompense future, con 1 tengo conto di tutte le ricompense future, con 0 solo di quella immediata
        #piu è basso più si concentra a massimizzare la reward immediata, piu è alto più si concentra a massimizzare la reward futura
        "gamma": 0.98,
        #tau controlla quanto lentamente aggiorni la rete target, più è piccolo più è stabile, però se è troppo piccolo il targhet resta troppo vecchio
        "tau": 0.001,
        # learning rate è il tasso di apprendimento, cioè quanto velocemente l'agente impara
        #più è alto più l'agente impara velocemente, ma può diventare instabile, più è basso più l'agente impara lentamente ma in modo stabile
        "learning_rate": 0.0005,
        #200'000 esperienze memorizzate, se è troppo grande può avere esperinze troppo vecchie
        "buffer_size": int(2e5),
        #se è piccolo aggiorna spesso ma con più rumore, se è grande è più robusto ma serve più memoria
        "batch_size": 128,
        #mi dice con quanto espoloro allìinizio, con 1 significa esploro sempre, sclgo azioni a caso
        "epsilon_start": 1.0,
       #mi dice come scelgo le azioni durante il traing, quando eplison scende esploro meno
        "epsilon_min": 0.05,
        #lentamnete epsilon scende
        "epsilon_decay": 0.9993, 
        #ogni quanto apprendi, con 8 o 4 è piu stabile
        "learn_every": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        #evita che il target sia troppo vecchio
        "target_update_freq": 500,
        #serve per evitare problemi di exploding gradients
        "gradient_clip": 1.0,
        "hidden_dim": 256,
        #serve per evitare overfitting, se è alto hai un apprendimento più lento ma più robusto, se è basso hai un apprendi più velocemente ma rischio di overfitting
        "dropout_prob": 0.2
    })

    #va a caricare il modello padre se esiste per usarlo come punto di partenza, in modo che la rete targhet e quella locale partano dal modello già allenato
    if parent_path and os.path.exists(parent_path):
        agent.load_model(parent_path)
        logger.info(f"Caricato modello padre: {parent_path}")
    
    #crea il file che contiene i risultati, metti a perchè è append 
    with open(csv_path, "a", newline="") as csvf:
        writer = csv.writer(csvf)
        #intestazione delle colonne
        writer.writerow([
            "level", "branch", "round", "episode", "reward", "epsilon",
            "arrivati", "collisioni", "steps", "unique_states", "learning_rate"
        ])

    #crea la barra di avanzamento per il training, ogni training farà n_rounds round e in ogni round farà n_episodes episodi → quindi total = n_rounds * n_episodes
    pbar = tqdm(total=n_rounds * n_episodes, desc=f"Training {branch_name}")
    #cicli sui round e per ogni round genera un ambiente con seed diverso e avere più round con mondi diversi mi fronisce maggiore robustezza
    for r in range(round_start, round_start + n_rounds):
        env = create_environment(effective_width, effective_height, effective_agents, max_depth, r)
        #per dopo 20 episodi non migliora riduci il learning rate
        patience = 20
        #conta quanti episodi consecutivi non migliorano la reward
        no_improve_count = 0

        for ep in range(n_episodes):
            try:
                episode_reward = 0
                #resetti l'ambiente
                obs, _ = env.reset(regenerate_rail=True, regenerate_schedule=True)
                done = {a: False for a in obs}
                states = {a: flatten_obs(obs[a], max_depth=max_depth) for a in obs}
                previous_positions = {a: env.agents[a].position or (0, 0) for a in obs}
                #resetti anche questo perchè fai un nuovo episodio e tu vuoi che parta da zero               
                arrived_count = 0
                collision_count = 0
                total_steps = 0
                current_dists = {}

                while not env.dones["__all__"]:
                    active_agents = [a for a in obs if not done[a]]
                    if not active_agents:
                        break
                    actions = {a: agent.select_action(states[a]) for a in active_agents}
                    
                    #fai uno step nell'ambiente e ottieni una nuova osservazione rewar info e done
                    next_obs, rewards, done_info, _ = env.step(actions)
                    total_steps += 1
                    for a in active_agents:
                        if done[a]:
                            continue
                        next_state = (
                            np.zeros_like(states[a]) if next_obs[a] is None
                            else flatten_obs(next_obs[a], max_depth=max_depth)
                        )
                        old_pos = previous_positions[a]
                        new_pos = env.agents[a].position or old_pos
                        target = env.agents[a].target or (0, 0)
                        arrived = env.agents[a].state == 6
                        collision = rewards[a] < -1
                        shaped_reward = calculate_shaped_reward(
                            old_pos, new_pos, target,
                            actions[a], arrived, collision, done_info[a]
                        )
                        agent.step(states[a], actions[a], shaped_reward, next_state, done[a])
                        states[a] = next_state
                        previous_positions[a] = new_pos
                        #aggiornamenti di stato
                        episode_reward += shaped_reward
                        if arrived:
                            done[a] = True
                            arrived_count += 1
                        if collision:
                            collision_count += 1
                        if all(pos is not None for pos in [new_pos, target]):
                            current_dists[a] = manhattan(new_pos, target)
                agent.update_epsilon()
                epsilon = round(agent.epsilon, 4)
                current_lr = agent.optimizer.param_groups[0]['lr']
                avg_distance = np.mean(list(current_dists.values())) if current_dists else 0

                with open(csv_path, "a", newline="") as csvf_append:
                    writer_append = csv.writer(csvf_append)
                    writer_append.writerow([
                        str(level), str(branch_name), str(r + 1), str(ep + 1),
                        f"{episode_reward:.2f}", f"{epsilon:.4f}",
                        str(arrived_count), str(collision_count), str(total_steps),
                        #f"{unique_samples:.1f}", f"{current_lr:.6f}"
                    ])
                all_rewards.append(episode_reward)
                
                #aggiorna la barra di avanzamento                  
                pbar.update(1)
                pbar.set_postfix({
                        'Reward': f"{episode_reward:.2f}",
                        'Arrived': arrived_count,
                        'Epsilon': f"{epsilon:.4f}",
                        'Collisions': collision_count,
                        'LR': f"{current_lr:.1e}",
                        'AvgDist': f"{avg_distance:.1f}",
                        'NoImprove': no_improve_count,
                        'Patience': patience
                })

                if episode_reward > best_reward * 1.05:
                    best_reward = episode_reward
                    no_improve_count = 0
                    agent.save_model(model_path)
                elif episode_reward > best_reward * 0.95:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        agent.adjust_learning_rate(0.8)
                        no_improve_count = 0
            except Exception as e:
                logger.error(f"Errore durante l'episodio {ep+1} nel branch {branch_name}: {e}")
                continue

    pbar.close()
    logger.info(f"Modello migliore salvato in {model_path}")
    return all_rewards
