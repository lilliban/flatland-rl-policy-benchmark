from flatland_rl_policy_benchmark.env.environment import EnvironmentBuilder
from flatland_rl_policy_benchmark.policies.DDDQNPolicy import DDDQNPolicy
import torch

def main():
    params = {
        "gamma": 0.99, "learning_rate": 1e-4, "tau": 1e-3,
        "batch_size": 32, "buffer_size": 10000
    }
    env = EnvironmentBuilder(width=35, height=35, n_agents=1, seed=1).build()
    obs, _ = env.reset()

    first = list(obs.keys())[0]
    state_size  = env.obs_builder.get(first).shape[0]
    action_size = 5
    agent = DDDQNPolicy(state_size, action_size, params)

    for ep in range(100):
        obs, _ = env.reset()
        done = {a: False for a in obs}; done["__all__"] = False

        while not done["__all__"]:
            actions = {aid: agent.select_action(aid, env.obs_builder.get(aid))
                       for aid in obs if not done.get(aid, False)}
            next_obs, rewards, done, _ = env.step(actions)
            for aid in obs:
                s  = env.obs_builder.get(aid)
                ns = env.obs_builder.get(aid)
                agent.step(s, actions[aid], rewards[aid], ns, float(done[aid]))
            obs = next_obs

    torch.save(agent.local_net.state_dict(), "dddqn_policy.pt")

if __name__ == "__main__":
    main()
