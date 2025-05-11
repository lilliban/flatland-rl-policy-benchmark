from flatland_rl_policy_benchmark.env.environment import EnvironmentBuilder
from flatland_rl_policy_benchmark.policies.HeuristicPolicy import HeuristicPolicy
from flatland_rl_policy_benchmark.policies.DDDQNPolicy import DDDQNPolicy
from flatland_rl_policy_benchmark.policies.PPOPolicy import PPOPolicy
import torch, csv

def run_tournament(n_rounds=5):
    params = {"gamma":0.99,"learning_rate":1e-4,"tau":1e-3,
              "batch_size":32,"buffer_size":10000,"eps_clip":0.2}
    env = EnvironmentBuilder(width=35, height=35, n_agents=1, seed=42).build()
    obs,_ = env.reset()
    first = list(obs.keys())[0]
    state_size  = env.obs_builder.get(first).shape[0]
    action_size = 5

    coaches = {
        "Heuristic": HeuristicPolicy(env),
        "DDDQN":    DDDQNPolicy(state_size, action_size, params),
        "PPO":      PPOPolicy(state_size, action_size, params)
    }
    coaches["DDDQN"].local_net.load_state_dict(torch.load("dddqn_policy.pt"))
    coaches["PPO"].ac.load_state_dict(torch.load("ppo_policy.pt"))

    results=[]
    for rnd in range(n_rounds):
        obs,_ = env.reset(); done={a:False for a in obs}; done["__all__"]=False
        scores={k:0 for k in coaches}
        while not done["__all__"]:
            for name, coach in coaches.items():
                acts = {}
                for aid in obs:
                    if name=="Heuristic":
                        acts[aid] = coach.select_action(aid)
                    else:
                        acts[aid] = coach.select_action(aid, env.obs_builder.get(aid))
                nob, r, done, _ = env.step(acts)
                scores[name] += sum(r.values())
                obs = nob
        for name, sc in scores.items():
            results.append((rnd, name, sc))

    with open("tournament_results.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["Round","Policy","Score"]); w.writerows(results)

if __name__=="__main__":
    run_tournament()
