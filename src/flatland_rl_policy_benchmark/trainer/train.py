# train.py
import random
import time

from flatland_rl_policy_benchmark.env.environment import EnvironmentBuilder
from flatland_rl_policy_benchmark.utils.Renderer import Renderer

def main():
    # Costruisci l’ambiente
    env = EnvironmentBuilder(width=35, height=35, n_agents=3, seed=42).build()
    renderer = Renderer(env)

    # Reset + primo step per far partire i treni
    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=42)
    env.step({aid: 0 for aid in obs})  # un passo “zero” per spawn
    renderer.render(show=True)

    # Loop: policy random (o “sempre avanti”)
    done = {aid: False for aid in obs}
    done["__all__"] = False
    step = 0

    while not done["__all__"]:
        # qui puoi scegliere random.choice([1,2,3,4]) 
        # o semplicemente 1 per “AVANTI” su ogni binario
        actions = {aid: random.choice([1, 2, 3, 4]) for aid in obs if not done.get(aid)}
        obs, rewards, done, info = env.step(actions)

        # render grafico
        env.render()
        renderer.render(show=True)

        time.sleep(0.2)
        step += 1

    print(f"\n✅ Terminato in {step} step. Premi CTRL+C per chiudere.")
    try:
        while True:
            renderer.render(show=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        renderer.close()

if __name__ == "__main__":
    main()
