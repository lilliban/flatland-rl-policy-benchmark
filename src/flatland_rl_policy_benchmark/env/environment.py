from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

class EnvironmentBuilder:
    def __init__(self, width=35, height=35, n_agents=3, seed=0, grid_mode=True):
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.seed = seed
        self.grid_mode = grid_mode

    def build(self) -> RailEnv:
        rail_generator = sparse_rail_generator(
            max_num_cities=max(2, self.n_agents),
            max_rails_between_cities=3,
            max_rail_pairs_in_city=2,
            seed=self.seed,
            grid_mode=self.grid_mode
        )
        line_generator = sparse_line_generator()

        env = RailEnv(
            width=self.width,
            height=self.height,
            rail_generator=rail_generator,
            line_generator=line_generator,
            number_of_agents=self.n_agents,
            obs_builder_object=TreeObsForRailEnv(
                max_depth=2,
                predictor=ShortestPathPredictorForRailEnv()
            ),
            remove_agents_at_target=True
        )
        return env
