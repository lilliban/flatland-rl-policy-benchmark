# train.py
from flatland_rl_policy_benchmark.env.environment import EnvironmentBuilder
from flatland_rl_policy_benchmark.policies.HeuristicPolicy import HeuristicPolicy
from flatland_rl_policy_benchmark.utils.Renderer import Renderer
import time

def main():
    env = EnvironmentBuilder(width=30, height=30, n_agents=5, seed=42).build()
    policy = HeuristicPolicy(env)
    renderer = Renderer(env)

    obs, info = env.reset()
    renderer.render(show=True)

    done = {agent: False for agent in obs}
    done["__all__"] = False

    while not done["__all__"]:
        actions = {}
        for agent_id in obs:
            if obs[agent_id] is not None and not done[agent_id]:
                actions[agent_id] = policy.select_action(agent_id, obs)
            else:
                actions[agent_id] = 0  # no-op

        obs, rewards, done, _ = env.step(actions)
        renderer.render(show=True)
        time.sleep(0.2)

if __name__ == "__main__":
    main()
