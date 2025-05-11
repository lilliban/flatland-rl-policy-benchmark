from flatland.envs.rail_env import RailEnvActions
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

class HeuristicPolicy:
    def __init__(self, env):
        self.env = env
        self.predictor = ShortestPathPredictorForRailEnv()

    def select_action(self, agent_handle):
        # prendo tutte le predizioni in un colpo solo
        preds = self.predictor.get(self.env)
        agent_preds = preds.get(agent_handle)
        # se esistono almeno due passi, prendo la direzione del secondo
        if agent_preds and len(agent_preds) > 1:
            return agent_preds[1][1]
        # altrimenti fermati
        return RailEnvActions.STOP_MOVING
