# test_env.py

from flatland_rl_policy_benchmark.env.environment import EnvironmentBuilder

def main():
    width, height, n_agents, seed = 35, 35, 3, 42
    env = EnvironmentBuilder(width, height, n_agents, seed).build()

    # reset “standard” (basta una volta)
    obs, info = env.reset(
        regenerate_rail=True,
        regenerate_schedule=True,
        random_seed=seed
    )

    # 1) smoke-test delle chiavi di obs/info
    print("Initial observation keys:", list(obs.keys()))
    print("Number of agents:", env.get_num_agents())
    print("Reset info keys:",   list(info.keys()))

    # 2) controlla che la schedule esista (initial_position ≠ None e target ≠ None)
    for agent in env.agents:
        assert agent.initial_position is not None, "Nessuna initial_position!"
        assert agent.target           is not None, "Nessun target!"
        print(f"Agent {agent.handle}: from {agent.initial_position} → {agent.target}")

    # 3) un singolo step
    actions = {aid: 0 for aid in obs}
    next_obs, rewards, done, info = env.step(actions)
    print("Rewards after one step:", rewards)
    print("Done flags:", done)

if __name__ == "__main__":
    main()
