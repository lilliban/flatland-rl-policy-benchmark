import torch
from flatland.envs.observations import TreeObsForRailEnv
from flatland_rl_policy_benchmark.env.environment import EnvironmentBuilder
from flatland_rl_policy_benchmark.policies.PPOPolicy import PPOPolicy
from flatland_rl_policy_benchmark.utils.obs_utils import flatten_obs

if __name__ == "__main__":
    env = EnvironmentBuilder(
        width=50,
        height=50,
        n_agents=2,
        seed=0,
        obs_builder_object=TreeObsForRailEnv(max_depth=3)
    ).build()

    obs, _ = env.reset(
        regenerate_rail=True,
        regenerate_schedule=True,
        random_seed=0
    )
    first_agent = list(obs.keys())[0]
    state_size  = flatten_obs(obs[first_agent]).shape[0]
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
        obs, _ = env.reset(...)
        done = {a: False for a in obs}
        state = flatten_obs(obs[first_agent], max_depth=3)

        while not all(done.values()):
            action, logprob = agent.select_action(state)
            next_obs, rewards, done, _ = env.step({first_agent: action})
            next_state = flatten_obs(next_obs[first_agent], max_depth=3)

            
                        # Reward shaping
            original_reward = rewards[first_agent]
            arrived = done[first_agent] and 'position' in next_obs[first_agent] and next_obs[first_agent]['position'] is None
            collision = original_reward < -1

            if arrived:
                shaped_reward = 100
            elif collision:
                shaped_reward = -5
            else:
                shaped_reward = -0.1

            agent.step(shaped_reward, done[first_agent])

            
            state = next_state

        agent.finish_episode()

        if ep % 50 == 0:
            print(f"✅ Episode {ep} completed")


    torch.save(agent.ac.state_dict(), "ppo_policy.pt")

    print("✅ PPO training completed and model saved to ppo_policy.pt")
    ppo = PPOPolicy(state_size, action_size, params)
    ppo.ac.load_state_dict(torch.load("ppo_policy.pt"))
