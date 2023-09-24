from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple

import gymnasium as gym
import numpy as np

from .env_api import _OfflineBitcoinEvaluationOnMonthWindow
from .env_api import _OfflineBitcoinEvaluationOpportunityLossOnHold


class OfflineBitcoinEvaluationOnMonthWindow(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self._environment_api = _OfflineBitcoinEvaluationOnMonthWindow(config)
        self._observation_space = self._environment_api.observation_space
        self._action_space = self._environment_api.action_space
        self._compute_reward = self._environment_api.compute_reward

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """reset task and set starting parameters"""
        super().reset(seed=seed)
        observation, info = self._environment_api.reset(seed, options)
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """compute transition dynamics based on the action chosen"""
        observation, reward, terminated, truncated, info = self._environment_api.step(
            action
        )
        clipped_observation = np.clip(observation, -1.0, 1.0)
        return clipped_observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        """set rendering mode"""
        self._environment_api.render()

    def close(self):
        """close simulation engine"""
        self._environment_api.close()

    @property
    def observation_space(self) -> gym.spaces:
        """return env observation space"""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value: gym.spaces) -> None:
        """set env observation space"""
        self._observation_space = value

    @property
    def action_space(self) -> gym.spaces:
        """return env action space"""
        return self._action_space

    @action_space.setter
    def action_space(self, value: gym.spaces) -> None:
        """set env action space"""
        self._action_space = value

    @property
    def compute_reward(self) -> Callable:
        """return reward computation method"""
        return self._compute_reward


class OfflineBitcoinEvaluationOpportunityLossOnHold(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self._environment_api = _OfflineBitcoinEvaluationOpportunityLossOnHold(config)
        self._observation_space = self._environment_api.observation_space
        self._action_space = self._environment_api.action_space
        self._compute_reward = self._environment_api.compute_reward
        self._timesteps = 0

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """reset task and set starting parameters"""
        super().reset(seed=seed)
        self._timesteps = 0
        observation, info = self._environment_api.reset(seed, options)
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """compute transition dynamics based on the action chosen"""
        self._timesteps += 1
        observation, reward, terminated, truncated, info = self._environment_api.step(
            action
        )
        clipped_observation = np.clip(observation, -1.0, 1.0)
        apply_opportunity_loss = False
        if not (self._timesteps % 10_000):
            apply_opportunity_loss = True
        reward = self._environment_api.task.apply_opportunity_loss(
            reward, apply_opportunity_loss
        )
        return clipped_observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        """set rendering mode"""
        self._environment_api.render()

    def close(self):
        """close simulation engine"""
        self._environment_api.close()

    def get_profit(self):
        """return profit in USD"""
        return self._environment_api.get_profit()

    def get_current_capital(self):
        """get current capital in USD"""
        return self._environment_api.get_current_capital()

    def get_current_crypto_capital(self):
        "get current capital in crypto"""
        return self._environment_api.get_current_crypto_capital()

    @property
    def observation_space(self) -> gym.spaces:
        """return env observation space"""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value: gym.spaces) -> None:
        """set env observation space"""
        self._observation_space = value

    @property
    def action_space(self) -> gym.spaces:
        """return env action space"""
        return self._action_space

    @action_space.setter
    def action_space(self, value: gym.spaces) -> None:
        """set env action space"""
        self._action_space = value

    @property
    def compute_reward(self) -> Callable:
        """return reward computation method"""
        return self._compute_reward
