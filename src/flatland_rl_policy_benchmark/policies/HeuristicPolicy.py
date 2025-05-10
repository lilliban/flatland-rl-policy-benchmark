from flatland.envs.predictions import ShortestPathPredictorForRailEnv

class HeuristicPolicy:
    """Baseline: segui sempre la rotta più breve."""
    def __init__(self, env):
        self.predictor = ShortestPathPredictorForRailEnv()

    def select_action(self, agent_handle, obs):
        # predictor restituisce lista di (next_cell, direction)
        next_move = self.predictor.predict(agent_handle, obs)
        return next_move.direction  # direzione tra {0,1,2,3,4}
