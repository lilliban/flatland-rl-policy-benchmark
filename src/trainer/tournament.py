import csv
from src.env.environment import EnvironmentBuilder
from src.policies.DDDQNPolicy import DDDQNPolicy
from src.policies.PPOPolicy import PPOPolicy
from src.policies.HeuristicPolicy import HeuristicPolicy

def run_tournament(n_rounds=5):
    params = {...}
    env = EnvironmentBuilder(...).generate()
    coaches = {
        "Heuristic": HeuristicPolicy(env),
        "DQN": DDDQNPolicy(100,5,params),
        "PPO": PPOPolicy(100,5,params),
    }
    # carica checkpoint .load_model() se presenti
    results = []
    for rnd in range(n_rounds):
        obs, _ = env.reset()
        done = {i:False for i in obs}
        while not done["__all__"]:
            actions = {i: coaches[name].select_action(i, obs[i]) 
                       for name in coaches for i in obs if obs[i] is not None}
            obs, rewards, done, _ = env.step(actions)
            # accumula
        # registra punteggi per ciascun coach
    # scrivi CSV con intitolazioni: round, policy, score
