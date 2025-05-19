# train_dddqn.py

import torch
from flatland.envs.observations import TreeObsForRailEnv
from flatland_rl_policy_benchmark.env.environment import EnvironmentBuilder
from flatland_rl_policy_benchmark.policies.DDDQNPolicy import DDDQNPolicy
from flatland_rl_policy_benchmark.utils.obs_utils import flatten_obs

if __name__ == "__main__":
    max_depth = 3  # profondità dell'albero
    num_features = 11  # fisso in TreeObs

    # 1) Costruzione ambiente con TreeObs
    env = EnvironmentBuilder(
        width=50,
        height=50,
        n_agents=2,
        seed=0,
        obs_builder_object=TreeObsForRailEnv(max_depth=max_depth)
    ).build()

    # 2) Reset iniziale per calcolo dimensione osservazione
    obs, _ = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=0)
    first_agent = list(obs.keys())[0]
    state_size = flatten_obs(obs[first_agent], max_depth=max_depth).shape[0]
    action_size = 5  # Flatland: 0–4 azioni

    # 3) Parametri per DDDQN
    params = {
        "gamma":         0.99,
        "tau":           1e-3,
        "learning_rate": 1e-3,
        "buffer_size":   int(1e5),
        "batch_size":    64,
        "epsilon_start": 1.0,
        "epsilon_min":   0.01,
        "epsilon_decay": 0.995,
        "learn_every":   4,
        "device":        "cpu"
    }

    agent = DDDQNPolicy(state_size, action_size, params)

    # 4) Training loop
    n_episodes = 500
    for ep in range(n_episodes):
        obs, _ = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=ep)
        done = {a: False for a in obs}
        state = flatten_obs(obs[first_agent], max_depth=max_depth)

        while not all(done.values()):
            action = agent.select_action(state)
            next_obs, rewards, done, _ = env.step({first_agent: action})
            next_state = flatten_obs(next_obs[first_agent], max_depth=max_depth)

                        # Reward shaping
            original_reward = rewards[first_agent]
            arrived = done[first_agent] and 'position' in next_obs[first_agent] and next_obs[first_agent]['position'] is None
            collision = original_reward < -1  # puoi raffinarlo se vuoi

            if arrived:
                shaped_reward = 100
            elif collision:
                shaped_reward = -5
            else:
                shaped_reward = -0.1

            agent.step(state, action, shaped_reward, next_state, done[first_agent])

            
            
            state = next_state

        if ep % 50 == 0:
            print(f"✅ Episode {ep} completed")

    # 5) Salvataggio del modello
    torch.save(agent.local_net.state_dict(), "dddqn_policy.pt")
    print("🎉 Training completo. Modello salvato in dddqn_policy.pt")
