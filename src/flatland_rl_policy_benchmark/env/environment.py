from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv

class EnvironmentBuilder:
    """
    Builder per RailEnv, con possibilità di specificare
    qualsiasi obs_builder_object (es. TreeObsForRailEnv, LocalObs…, ecc.).
    """
    def __init__(self,
                 width: int = 10,
                 height: int = 10,
                 n_agents: int = 1,
                 seed: int = 0,
                 max_cities: int = 3,
                 obs_builder_object: object = None):
        self.width              = width
        self.height             = height
        self.n_agents           = n_agents
        self.seed               = seed
        self.max_cities         = max_cities

        # Se non viene passato nulla, uso la global view di default
        self.obs_builder_object = (
            obs_builder_object
            if obs_builder_object is not None
            else GlobalObsForRailEnv()
        )

    def build(self) -> RailEnv:
        env = RailEnv(
            width=self.width,
            height=self.height,
            
            
            rail_generator=sparse_rail_generator(
                max_num_cities=self.max_cities,
                seed=self.seed,
                grid_mode=True, 
                max_rails_between_cities=3,
                max_rail_pairs_in_city=2
            ),
            line_generator=sparse_line_generator(),
            number_of_agents=self.n_agents,
            obs_builder_object=self.obs_builder_object,
            remove_agents_at_target=True
        )
        return env
