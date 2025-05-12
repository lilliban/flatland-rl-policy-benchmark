import os
import csv
import torch
import numpy as np
from flatland.envs.observations import TreeObsForRailEnv
import time
from flatland_rl_policy_benchmark.env.environment import EnvironmentBuilder
from flatland_rl_policy_benchmark.policies.DDDQNPolicy import DDDQNPolicy
from flatland_rl_policy_benchmark.policies.PPOPolicy import PPOPolicy
from flatland_rl_policy_benchmark.utils.obs_utils import flatten_obs

# Setup
N_ROUNDS = 5
N_AGENTS = 2
MAP_WIDTH = 50
MAP_HEIGHT = 50
MAX_DEPTH = 2

OUTPUT_CSV = "tournament_results.csv"

# Inizializza policies
POLICIES = {
    "DDDQN": lambda state_dim, action_dim: DDDQNPolicy(state_dim, action_dim, {"device": "cpu"}),
    "PPO":   lambda state_dim, action_dim: PPOPolicy(state_dim, action_dim, {"device": "cpu"}),
}

MODEL_PATHS = {
    "DDDQN": "dddqn_policy.pt",
    "PPO": "ppo_policy.pt"
}

def run_episode(env, agents):
    from flatland_rl_policy_benchmark.utils.Renderer import Renderer
    import time

    obs, _ = env.reset(regenerate_rail=True, regenerate_schedule=True)
    done = {a: False for a in obs}
    states = {a: flatten_obs(obs[a], max_depth=MAX_DEPTH) for a in obs}

    renderer = Renderer(env)
    renderer.render(show=True)

    while not all(done.values()):
        actions = {}
        for a in obs:
            if not done[a]:
                action = agents[a].select_action(states[a])
                if isinstance(action, tuple):
                    action = action[0]
                actions[a] = action

        next_obs, rewards, done, _ = env.step(actions)
        for a in next_obs:
            if not done[a]:
                states[a] = flatten_obs(next_obs[a], max_depth=MAX_DEPTH)

        renderer.render(show=True)
        time.sleep(0.2)

    return {agent_id: agents[agent_id].metrics() for agent_id in agents}


def main():
    print("\n🏁 Inizio torneo RL su Flatland...\n")
    results = []

    for round_id in range(1, N_ROUNDS + 1):
        print(f"\n🎲 Round {round_id}/{N_ROUNDS}")

        # Costruisci ambiente
        env = EnvironmentBuilder(
            width=MAP_WIDTH,
            height=MAP_HEIGHT,
            n_agents=N_AGENTS,
            seed=round_id,
            obs_builder_object=TreeObsForRailEnv(max_depth=MAX_DEPTH)
        ).build()

        obs, _ = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=round_id)
        first_agent = list(obs.keys())[0]
        state_dim = flatten_obs(obs[first_agent], max_depth=MAX_DEPTH).shape[0]
        action_dim = 5

        # Assegna policy a ogni agente
        agents = {}
        policy_names = list(POLICIES.keys())
        for i, agent_id in enumerate(obs):
            pname = policy_names[i % len(policy_names)]
            policy = POLICIES[pname](state_dim, action_dim)
            policy_path = MODEL_PATHS[pname]
            if pname == "DDDQN":
                policy.local_net.load_state_dict(torch.load(policy_path))
            else:
                policy.ac.load_state_dict(torch.load(policy_path))
            agents[agent_id] = policy
            agents[agent_id].__name__ = pname  # per etichetta

        # Esegui l'episodio
        metrics = run_episode(env, agents)

        for aid, stats in metrics.items():
            results.append({
                "round": round_id,
                "agent_id": aid,
                "policy": agents[aid].__name__,
                **stats
            })

    # Salva CSV
    keys = list(results[0].keys())
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", OUTPUT_CSV), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print("\n📊 Risultati torneo salvati in 'results/tournament_results.csv'")

if __name__ == "__main__":
    main()
