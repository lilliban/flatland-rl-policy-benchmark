from flatland_rl_policy_benchmark.env.environment import EnvironmentBuilder

def main():
    width, height, n_agents, seed = 35, 35, 3, 42
    env = EnvironmentBuilder(width, height, n_agents, seed).build()

    # smoke–test di reset + schedule
    obs, info = env.reset(
        regenerate_rail=True,
        regenerate_schedule=True,
        random_seed=seed
    )

    print("Initial observation keys:", list(obs.keys()))
    print("Number of agents:", env.get_num_agents())
    print("Reset info keys:",   list(info.keys()))

    # controllo che abbia generato initial_position e target
    for a in env.agents:
        assert a.initial_position is not None, "Manca initial_position!"
        assert a.target           is not None, "Manca target!"
        print(f"Agent {a.handle}: from {a.initial_position} → {a.target}")

    # un singolo step di prova
    actions = {aid: 0 for aid in obs}
    next_obs, rewards, done, info = env.step(actions)
    print("Rewards after one step:", rewards)
    print("Done flags:", done)

if __name__ == "__main__":
    main()
