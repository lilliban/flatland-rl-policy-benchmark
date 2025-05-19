# train_ppo.py — training aggiornato per 4 agenti con reward shaping completo

import torch
from flatland.envs.observations import TreeObsForRailEnv
from flatland_rl_policy_benchmark.env.environment import EnvironmentBuilder
from flatland_rl_policy_benchmark.policies.PPOPolicy import PPOPolicy
from flatland_rl_policy_benchmark.utils.obs_utils import flatten_obs


def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

if __name__ == "__main__":
    max_depth = 5
    env = EnvironmentBuilder(
        width=25,
        height=25,
        n_agents=4,
        seed=0,
        obs_builder_object=TreeObsForRailEnv(max_depth=max_depth)
    ).build()

    obs, _ = env.reset(regenerate_rail=True, regenerate_schedule=True)
    first_agent = list(obs.keys())[0]
    state_size = flatten_obs(obs[first_agent], max_depth=max_depth).shape[0]
    action_size = 5

    params = {
        "gamma":         0.99,
        "learning_rate": 3e-4,
        "clip_epsilon":  0.2,
        "update_every":  20,
        "device":        "cpu"
    }

    agent = PPOPolicy(state_size, action_size, params)

    n_episodes = 500
    for ep in range(n_episodes):
        obs, _ = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=ep)
        done = {a: False for a in obs}
        states = {a: flatten_obs(obs[a], max_depth=max_depth) for a in obs}

        while not all(done.values()):
            actions = {}
            for a in obs:
                if not done[a]:
                    act, _ = agent.select_action(states[a])
                    actions[a] = act

            next_obs, rewards, done, _ = env.step(actions)

            for a in obs:
                if not done[a]:
                    pos = env.agents[a].position
                    target = env.agents[a].target
                    new_pos = env.agents[a].position

                    old_dist = manhattan(pos, target) if pos and target else 0
                    new_dist = manhattan(new_pos, target) if new_pos and target else 0
                    delta = old_dist - new_dist

                    original_reward = rewards[a]
                    arrived = done[a] and 'position' in next_obs[a] and next_obs[a]['position'] is None
                    collision = original_reward < -1

                    shaped_reward = 0.01 * delta
                    if arrived:
                        shaped_reward += 10
                    if collision:
                        shaped_reward -= 5
                    if actions[a] != 0:
                        shaped_reward += 0.1

                    agent.step(shaped_reward, done[a])
                    states[a] = flatten_obs(next_obs[a], max_depth=max_depth)

        agent.finish_episode()
        if ep % 50 == 0:
            print(f" Episode {ep} completed")

    torch.save(agent.ac.state_dict(), "ppo_policy.pt")
    print(" PPO training completed e salvato in ppo_policy.pt")
