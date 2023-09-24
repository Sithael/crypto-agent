from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple

import gymnasium as gym
import numpy as np
from crypto.abstract import SimulationTask


class CryptoTaskEnv:
    """
    Crypto task goal env, as the junction of a task and strategy

    Args:
        :param [strategy]: crypto type [ Rapid ]
        :param [task]: the task for the agent to solve
    """

    def __init__(self, strategy: str, task: SimulationTask) -> None:
        self.strategy = strategy
        self.task = task

        observation, _ = self.reset()
        observation_shape = observation.shape
        self._observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=observation_shape, dtype=np.float32
        )
        self._action_space = self.strategy.defined_action_space
        self._compute_reward = self.task.compute_reward

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """reset task and set starting parameters"""
        self.task.reset()
        observation, terminated = self.task.construct_observation()
        return observation, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """compute transition dynamics based on the action chosen"""
        action = action.copy()
        action = self.strategy.normalize_action(action)
        observation, terminated = self.task.construct_observation()
        reward = self.task.compute_reward(action, {})
        # truncated flag is set to False as the TimeLimit Gym wrapper is used to handle it
        return observation, reward, terminated, False, {}

    def render(self, mode="human"):
        """set rendering mode"""
        pass

    def close(self):
        """close simulation engine"""
        pass

    def get_profit(self):
        """get profit in USD from task"""
        return self.task.profit

    def get_current_capital(self):
        """get current capital in USD """
        return self.task.current_capital

    def get_current_crypto_capital(self):
        """get capital in crypto"""
        return self.task.current_crypto_capital

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def compute_reward(self) -> Callable:
        return self._compute_reward
