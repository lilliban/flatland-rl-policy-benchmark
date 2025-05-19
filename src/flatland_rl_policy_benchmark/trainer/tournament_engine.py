import os
import csv
import torch
import numpy as np
import time
from itertools import cycle
from flatland.envs.observations import TreeObsForRailEnv
from flatland_rl_policy_benchmark.env.environment import EnvironmentBuilder
from flatland_rl_policy_benchmark.policies.DDDQNPolicy import DDDQNPolicy
from flatland_rl_policy_benchmark.policies.PPOPolicy import PPOPolicy
from flatland_rl_policy_benchmark.utils.obs_utils import flatten_obs
from flatland_rl_policy_benchmark.utils.Renderer import Renderer

N_ROUNDS = 50
N_AGENTS = 2
MAP_WIDTH = 50
MAP_HEIGHT = 50
MAX_DEPTH = 3
OUTPUT_CSV = "tournament_results.csv"

POLICIES = {
    "DDDQN": lambda s, a: DDDQNPolicy(s, a, {"device": "cpu"}),
    "PPO": lambda s, a: PPOPolicy(s, a, {"device": "cpu"})
}
MODEL_PATHS = {
    "DDDQN": "dddqn_policy.pt",
    "PPO": "ppo_policy.pt"
}

policy_cycle = cycle(POLICIES.keys())

def run_episode(env, agents, renderer=None):
    obs, _ = env.reset(regenerate_rail=True, regenerate_schedule=True)
    done = {a: False for a in obs}
    states = {a: flatten_obs(obs[a], max_depth=MAX_DEPTH) for a in obs}

    arrived_count = 0  # 👈 inizializza il contatore dei treni arrivati

    while not all(done.values()):
        actions = {}
        for a in obs:
            if not done[a]:
                action = agents[a].select_action(states[a])
                if isinstance(action, tuple):
                    action = action[0]
                actions[a] = action

        next_obs, rewards, done, _ = env.step(actions)

        if renderer:
            renderer.render()
            time.sleep(0.1)

        for a in actions:
            agent = agents[a]
            original_reward = rewards[a]
            arrived = done[a] and 'position' in next_obs[a] and next_obs[a]['position'] is None
            collision = original_reward < -1

            if arrived:
                shaped_reward = 100
                arrived_count += 1  # 👈 incrementa se il treno è arrivato
            elif collision:
                shaped_reward = -50
            else:
                shaped_reward = -1

            if isinstance(agent, PPOPolicy):
                agent.step(shaped_reward, done[a])
            elif isinstance(agent, DDDQNPolicy):
                agent.step(states[a], actions[a], shaped_reward, flatten_obs(next_obs[a], max_depth=MAX_DEPTH), done[a])

        for a in next_obs:
            if not done[a]:
                states[a] = flatten_obs(next_obs[a], max_depth=MAX_DEPTH)

    print(f"🚂 Treni arrivati a destinazione in questo episodio: {arrived_count}")  # 👈 stampa finale

    return {aid: agent.metrics() for aid, agent in agents.items()}


def main():
    print("\n🏁 Inizio torneo RL su Flatland...\n")
    results = []

    for round_id in range(1, N_ROUNDS + 1):
        print(f"\n🎲 Round {round_id}/{N_ROUNDS}")

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

        agents = {}
        policy_names = [next(policy_cycle) for _ in range(N_AGENTS)]
        for i, agent_id in enumerate(obs):
            pname = policy_names[i % len(policy_names)]
            policy = POLICIES[pname](state_dim, action_dim)
            model_path = MODEL_PATHS[pname]
            try:
                if pname == "DDDQN":
                    policy.local_net.load_state_dict(torch.load(model_path))
                else:
                    policy.ac.load_state_dict(torch.load(model_path))
            except Exception as e:
                print(f"⚠️ Errore nel caricare {model_path}: {e}")
            policy.__name__ = pname
            agents[agent_id] = policy

        renderer = Renderer(env, {aid: agents[aid].__name__ for aid in agents})
        renderer.render(show=True)
        metrics = run_episode(env, agents, renderer)

        for agent in agents.values():
            if hasattr(agent, "finish_episode"):
                agent.finish_episode()

        renderer.close()

        for aid, stats in metrics.items():
            results.append({
                "round": round_id,
                "agent_id": aid,
                "policy": agents[aid].__name__,
                **stats
            })

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", OUTPUT_CSV), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n📊 Risultati torneo salvati in 'results/tournament_results.csv'")

if __name__ == "__main__":
    main()
