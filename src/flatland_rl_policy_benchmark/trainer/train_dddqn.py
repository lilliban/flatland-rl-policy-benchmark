import os
import csv
import torch
import numpy as np
from tqdm import tqdm
from flatland.envs.observations import TreeObsForRailEnv
from flatland_rl_policy_benchmark.env.environment import EnvironmentBuilder
from flatland_rl_policy_benchmark.policies.DDDQNPolicy import DDDQNPolicy
from flatland_rl_policy_benchmark.utils.obs_utils import flatten_obs

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def train(round_start, n_rounds, branch_name, parent_path, save_dir, level=0, width=25, height=25, n_agents=4):
    max_depth = 3
    n_episodes = 100

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"evolution_level{level}.csv")
    is_new_file = not os.path.exists(csv_path)
    
    params = {
        "gamma": 0.99,
        "tau": 0.01,
        "learning_rate": 0.001,
        "buffer_size": int(1e5),
        "batch_size": 64,
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "learn_every": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "prioritized_replay": True,
        "prioritized_replay_alpha": 0.6,
        "target_update_freq": 1000,
        "gradient_clip": 1.0
    }

    best_reward = -float('inf')
    model_path = os.path.join(save_dir, f"best_model_L{level}_B{branch_name.split('_')[-1]}.pt")
    all_rewards = []

    with open(csv_path, "a", newline="") as csvf:
        writer = csv.writer(csvf)
        if is_new_file:
            writer.writerow([
                "level", "branch", "round", "episode", "reward", "epsilon",
                "arrivati", "collisioni", "steps", "unique_states"
            ])

        pbar = tqdm(total=n_rounds * n_episodes, desc=f"Training {branch_name}")
        
        for r in range(round_start, round_start + n_rounds):
            env = EnvironmentBuilder(
                width=width,
                height=height,
                n_agents=n_agents,  # Usa sempre n_agents passato come parametro
                seed=r,
                obs_builder_object=TreeObsForRailEnv(max_depth=max_depth)
            ).build()

            obs, _ = env.reset(regenerate_rail=True, regenerate_schedule=True)
            first_agent = list(obs.keys())[0]
            state_size = flatten_obs(obs[first_agent], max_depth=max_depth).shape[0]
            action_size = 5

            agent = DDDQNPolicy(state_size, action_size, params)

            if parent_path and os.path.exists(parent_path):
                agent.local_net.load_state_dict(torch.load(parent_path))
                agent.target_net.load_state_dict(torch.load(parent_path))
                print(f" Caricato modello padre: {parent_path}")

            patience = 20
            no_improve_count = 0

            for ep in range(n_episodes):
                episode_reward = 0
                obs, _ = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=ep)
                done = {a: False for a in obs}
                states = {a: flatten_obs(obs[a], max_depth=max_depth) for a in obs}
                previous_positions = {a: env.agents[a].position for a in obs}
                treni_arrivati = 0
                treni_collisioni = 0
                total_steps = 0

                while not env.dones["__all__"]:
                    actions = {a: agent.select_action(states[a]) for a in obs if not done[a]}
                    next_obs, rewards, done_info, _ = env.step(actions)
                    total_steps += 1
                    
                    for a in actions:
                        if next_obs[a] is None:
                            done[a] = True
                            continue
                        if done[a]:
                            continue
                            
                        state = states[a]
                        next_state = flatten_obs(next_obs[a], max_depth=max_depth)
                        old_pos = previous_positions[a]
                        new_pos = env.agents[a].position
                        target = env.agents[a].target
                        
                        old_dist = manhattan(old_pos, target) if old_pos and target else 0
                        new_dist = manhattan(new_pos, target) if new_pos and target else 0
                        delta = old_dist - new_dist
                        
                        arrived = env.agents[a].state == 6
                        collision = rewards[a] < -1
                        

                        shaped_reward = 0

                        if old_pos and new_pos and target:
                            old_dist = manhattan(old_pos, target)
                            new_dist = manhattan(new_pos, target)
                            delta = old_dist - new_dist
                            shaped_reward += 1.0 * delta  # premiamo ogni miglioramento
                        else:
                            delta = 0  # fallback

                        if arrived:
                            shaped_reward += 100
                        elif done_info[a]:
                            shaped_reward -= 5

                        if actions[a] == 0:  # Fermata
                            shaped_reward -= 0.05
                        else:  # Movimento
                            shaped_reward += 0.3

                        if collision:
                            shaped_reward -= 5

                        if shaped_reward == 0:
                            shaped_reward -= 1  # penalità minima per inattività totale

                            
                        agent.step(state, actions[a], shaped_reward, next_state, done[a])
                        states[a] = next_state
                        episode_reward += shaped_reward
                        
                        if arrived:
                            done[a] = True
                            treni_arrivati += 1
                        if collision:
                            treni_collisioni += 1

                # Logging
                epsilon = round(agent.epsilon, 4)
                unique_samples = agent.memory.get_diversity(sample_size=1000) if len(agent.memory) >= agent.batch_size else 0
                writer.writerow([level, branch_name, r + 1, ep + 1, episode_reward, epsilon, 
                               treni_arrivati, treni_collisioni, total_steps, unique_samples])
                all_rewards.append(episode_reward)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'Reward': episode_reward,
                    'Arrived': treni_arrivati,
                    'Epsilon': epsilon,
                    'Collisions': treni_collisioni
                })

                # Early stopping & model saving
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    no_improve_count = 0
                    torch.save(agent.local_net.state_dict(), model_path)
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        for g in agent.optimizer.param_groups:
                            g['lr'] *= 0.9
                        no_improve_count = 0

        pbar.close()
        print(f"\n✅ Modello migliore salvato in {model_path}")

    return all_rewards