from flatland_rl_policy_benchmark.env.environment import EnvironmentBuilder
from flatland_rl_policy_benchmark.policies.PPOPolicy import PPOPolicy
import pickle
import torch

def main():
    params = {"gamma":0.99, "learning_rate":1e-4, "eps_clip":0.2}
    env = EnvironmentBuilder(width=30, height=30, n_agents=1, seed=1).build()
    obs, _ = env.reset()

    state_size = obs[list(obs.keys())[0]].astype('float32').flatten().shape[0]

    action_size = 5
    agent = PPOPolicy(state_size, action_size, params)

    for episode in range(100):  # puoi aumentare
        obs, _ = env.reset()
        done = {a: False for a in obs}
        done["__all__"] = False
        trajectories = []

        while not done["__all__"]:
            actions, log_probs = {}, {}
            for agent_id in obs:
                state = obs[agent_id].astype('float32').flatten()
                act, logp = agent.select_action(state)
                actions[agent_id] = act
                log_probs[agent_id] = logp

            next_obs, rewards, done, _ = env.step(actions)

            for agent_id in obs:
                s = obs[agent_id].astype('float32').flatten()
                ns = next_obs[agent_id].astype('float32').flatten()
                a = actions[agent_id]
                r = rewards[agent_id]
                d = float(done[agent_id])
                logp = log_probs[agent_id]
                trajectories.append((s, a, logp, r, ns, d))

            obs = next_obs

        agent.learn(trajectories)

    with open("ppo_policy.pt", "wb") as f:
        torch.save(agent.ac.state_dict(), f)

if __name__ == "__main__":
    main()
