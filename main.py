# src/env/simulate_render.py
from src.env.environment import EnvironmentBuilder
from src.utils.Renderer import Renderer
from src.policies.HeuristicPolicy import HeuristicPolicy  # o DDDQNPolicy/PPOPolicy

def main():
    # 1) Crea l’ambiente
    env = EnvironmentBuilder(width=60, height=60, num_agents=4).generate()

    # 2) Scegli la policy
    policy = HeuristicPolicy(env)
    # policy = DDDQNPolicy(state_size, action_size, params)
    # policy = PPOPolicy(state_size, action_size, params)

    # 3) Inizializza il renderer
    renderer = Renderer(env)

    # 4) Loop di simulazione
    obs, _ = env.reset()
    done = {i: False for i in obs}
    while not done["__all__"]:
        # seleziona un’azione per ciascun agente
        actions = {i: policy.select_action(i, obs[i]) for i in obs if obs[i] is not None}
        obs, rewards, done, _ = env.step(actions)
        renderer.render()  # disegna la scena corrente
    renderer.close()

if __name__=="__main__":
    main()
