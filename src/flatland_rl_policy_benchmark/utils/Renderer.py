from flatland.utils.rendertools import RenderTool

class Renderer:
    _instance = None

    def __new__(cls, env, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @staticmethod
    def policy_color_mapping(agent_policy_map):
        """
        Restituisce un dizionario {agent_id: colore} in base alla policy.
        """
        color_palette = {
            "DDDQN": [255, 0, 0],     # rosso
            "PPO": [0, 0, 255],       # blu
            "Heuristic": [0, 255, 0]  # verde opzionale
        }
        return {
            agent_id: color_palette.get(policy, [128, 128, 128])
            for agent_id, policy in agent_policy_map.items()
        }

    def __init__(self, env, agent_policy_map=None, screen_width=1200, screen_height=800):
        if self._initialized: return
        self.env = env
        self.render_tool = RenderTool(
            env,
            gl="PGL",
            agent_render_variant=2,  # mostra treni con freccia
            show_debug=True,
            screen_width=screen_width,
            screen_height=screen_height
        )
        self._initialized = True

    def render(self, **kwargs):
        self.render_tool.render_env(
            show=True,
            frames=True,
            show_observations=kwargs.get("show_observations", False),
            show_predictions=kwargs.get("show_predictions", False)
        )

    def reset(self):
        self.render_tool.close_window()
        self.__init__(self.env)

    def close(self):
        self.render_tool.close_window()
        type(self)._instance = None
        self._initialized = False
