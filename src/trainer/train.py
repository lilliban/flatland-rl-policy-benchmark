import torch
from src.env.environment import EnvironmentBuilder
from src.policies.DDDQNPolicy import DDDQNPolicy
from src.policies.PPOPolicy import PPOPolicy
from src.policies.HeuristicPolicy import HeuristicPolicy

def main():
    params = {"gamma":0.99, "learning_rate":1e-4, "tau":1e-3, "batch_size":32, "buffer_size":100000, "eps_clip":0.2}
    env = EnvironmentBuilder(width=30, height=30, num_agents=5).generate()
    # 1) euristica
    eur = HeuristicPolicy(env)
    # 2) DDDQN
    dqn = DDDQNPolicy(100, 5, params)
    # 3) PPO
    ppo = PPOPolicy(100,5, params)
    # scrivi il tuo loop di training per ciascuna policy
    # salva modelli con .save_model()

if __name__=="__main__":
    main()
