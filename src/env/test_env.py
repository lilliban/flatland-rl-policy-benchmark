from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

def main():
    # Genero 2 città su una mappa 50×50
    rail_gen = sparse_rail_generator(
        max_num_cities=2,
        max_rails_between_cities=1,
        max_rail_pairs_in_city=1,
        seed=1,
        grid_mode=False
    )
    line_gen = sparse_line_generator()
    obs_builder = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())

    env = RailEnv(
        width=50,
        height=50,
        rail_generator=rail_gen,
        line_generator=line_gen,
        number_of_agents=2,
        obs_builder_object=obs_builder,
        remove_agents_at_target=True
    )

    # Reset dell'ambiente
    obs, info = env.reset()
    print("Initial observation keys:", list(obs.keys()))
    print("Number of agents:", env.get_num_agents())
    # info contiene anche, ad esempio, 'max_steps' e 'predicted_paths'
    print("Reset info keys:", list(info.keys()))

    # Un singolo passo: tutti gli agenti stazionari (azione 0)
    actions = {agent_id: 0 for agent_id in obs}
    next_obs, rewards, done, info = env.step(actions)
    print("Rewards after one step:", rewards)
    print("Done flags:", done)

if __name__ == "__main__":
    main()
