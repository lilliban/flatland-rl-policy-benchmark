import os
import csv
import torch
from flatland.envs.observations import TreeObsForRailEnv
from flatland_rl_policy_benchmark.env.environment import EnvironmentBuilder
from flatland_rl_policy_benchmark.policies.DDDQNPolicy import DDDQNPolicy
from flatland_rl_policy_benchmark.utils.obs_utils import flatten_obs

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def train(round_start, n_rounds, branch_name, parent_path, save_dir):
    max_depth = 3
    n_episodes = 1000

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"log_{branch_name}.csv")
    model_path = os.path.join(save_dir, f"model_{branch_name}.pt")

    writer = csv.writer(open(log_path, "w", newline=""))
    writer.writerow(["round", "episode", "reward"])

    all_rewards = []

    for r in range(round_start, round_start + n_rounds):
        print(f"\n🔁 ROUND {r+1}/{round_start + n_rounds}")
        env = EnvironmentBuilder(
            width=25,
            height=25,
            n_agents=4,
            seed=r,
            obs_builder_object=TreeObsForRailEnv(max_depth=max_depth)
        ).build()

        obs, _ = env.reset(regenerate_rail=True, regenerate_schedule=True)
        first_agent = list(obs.keys())[0]
        state_size = flatten_obs(obs[first_agent], max_depth=max_depth).shape[0]
        action_size = 5

        params = {
            "gamma":         0.99,
            "tau":           1e-3,
            "learning_rate": 1e-3,
            "buffer_size":   int(1e4),
            "batch_size":    64,
            "epsilon_start": 1.0,
            "epsilon_min":   0.01,
            "epsilon_decay": 0.995,
            "learn_every":   4,
            "device":        "cpu"
        }

        agent = DDDQNPolicy(state_size, action_size, params)

        if parent_path and os.path.exists(parent_path):
            agent.local_net.load_state_dict(torch.load(parent_path))
            agent.target_net.load_state_dict(torch.load(parent_path))
            print(f"🔁 Caricato modello padre: {parent_path}")

        for ep in range(n_episodes):
            if ep % 50 == 0:
                print(f"  ▶️ Episodio {ep}/{n_episodes}")

            episode_reward = 0
            obs, _ = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=ep)
            done = {a: False for a in obs}
            states = {a: flatten_obs(obs[a], max_depth=max_depth) for a in obs}

            while not all(done.values()):
                actions = {}
                for a in obs:
                    if not done[a]:
                        actions[a] = agent.select_action(states[a])

                next_obs, rewards, done, _ = env.step(actions)

                for a in obs:
                    if not done[a]:
                        state = states[a]
                        next_state = flatten_obs(next_obs[a], max_depth=max_depth)

                        pos = env.agents[a].position
                        target = env.agents[a].target
                        old_dist = manhattan(pos, target) if pos and target else 0
                        pos_next = env.agents[a].position
                        new_dist = manhattan(pos_next, target) if pos_next and target else 0
                        delta = old_dist - new_dist

                        arrived = env.agents[a].state == 6
                        collision = rewards[a] < -1

                        shaped_reward = 0.01 * delta
                        if arrived:
                            shaped_reward += 50
                        if actions[a] != 0:
                            shaped_reward += 0.1
                        if collision:
                            shaped_reward -= 5

                        agent.step(state, actions[a], shaped_reward, next_state, done[a])
                        states[a] = next_state
                        episode_reward += shaped_reward

            writer.writerow([r + 1, ep + 1, episode_reward])
            all_rewards.append(episode_reward)

    torch.save(agent.local_net.state_dict(), model_path)
    print(f"✅ Modello salvato in {model_path}")
    return all_rewards
