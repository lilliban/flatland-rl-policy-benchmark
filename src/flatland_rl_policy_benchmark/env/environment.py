from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

class EnvironmentBuilder:
    def __init__(self, width=30, height=30, num_agents=5, seed=42):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.seed = seed

    def generate(self):
        rail_generator = sparse_rail_generator(
            max_num_cities=9,
            max_rails_between_cities=3,
            max_rail_pairs_in_city=8,
            seed=self.seed,
            grid_mode=False
        )
        line_generator = sparse_line_generator()
        obs_builder = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())

        env = RailEnv(
            width=self.width,
            height=self.height,
            rail_generator=rail_generator,
            line_generator=line_generator,
            number_of_agents=self.num_agents,
            obs_builder_object=obs_builder,
            remove_agents_at_target=True
        )
        return env
