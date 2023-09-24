import gymnasium as gym
from crypto import OfflineBitcoinEvaluationOpportunityLossOnHold
from gymnasium.wrappers import FrameStack
from gymnasium.wrappers import TimeLimit
from ray.rllib.env.env_context import EnvContext

class EnvDelegator(gym.Env):
    def __new__(cls, env_config):
        env = OfflineBitcoinEvaluationOpportunityLossOnHold(config=env_config)
        env = TimeLimit(env, 43200)
        env = FrameStack(env, 4)
        return env
