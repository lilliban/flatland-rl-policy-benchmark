import torch
import numpy as np
from flatland.envs.observations import TreeObsForRailEnv
from flatland_rl_policy_benchmark.env.environment import EnvironmentBuilder
from flatland_rl_policy_benchmark.policies.DDDQNPolicy import DDDQNPolicy
from flatland_rl_policy_benchmark.policies.PPOPolicy import PPOPolicy
from flatland_rl_policy_benchmark.utils.obs_utils import flatten_obs

def run_episode(policy, env):
    obs, _ = env.reset(regenerate_rail=True, regenerate_schedule=True)
    first_agent = list(obs.keys())[0]
    state = flatten_obs(obs[first_agent])
    done = {a: False for a in obs}

    while not all(done.values()):
        action = policy.select_action(state)
        next_obs, _, done, _ = env.step({first_agent: action})
        state = flatten_obs(next_obs[first_agent])

    return policy  # qui torna l’agente con metriche interne

def main():
    env = EnvironmentBuilder(
        width=50,
        height=50,
        n_agents=2,
        seed=0,
        obs_builder_object=TreeObsForRailEnv(max_depth=2)
    ).build()

    # dimensioni per istanziare le policy
    obs, _ = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=0)
    first_agent = list(obs.keys())[0]
    state_dim   = flatten_obs(obs[first_agent]).shape[0]
    action_dim  = 5

    # carica modelli
    dddqn = DDDQNPolicy(state_dim, action_dim, {"device": "cpu"})
    dddqn.local_net.load_state_dict(torch.load("dddqn_policy.pt"))
    ppo   = PPOPolicy(state_dim, action_dim, {"device": "cpu"})
    ppo.policy.load_state_dict(torch.load("ppo_policy.pt"))

    # esegui match e valuta
    result1 = run_episode(dddqn, env)
    result2 = run_episode(ppo,   env)

    # stampa metriche (definisci tu come aggregare)
    print("DDDQN metrics:", result1.metrics())
    print("PPO   metrics:", result2.metrics())

if __name__ == "__main__":
    main()
