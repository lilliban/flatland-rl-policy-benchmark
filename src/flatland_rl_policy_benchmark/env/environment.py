from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv
#QUESTA CLASSE è UN HELPER
#usa RailEnv per creare ambienti di simulazione di treni
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
        self.max_num_cities     = max_cities

        # nel DDDQN dovresti sempre essere sicura di passare TreeObs, quindi lo metti di default
        if obs_builder_object is None:
            raise ValueError("obs_builder_object must be provided! (e.g. TreeObsForRailEnv)")

        self.obs_builder_object = obs_builder_object

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
