from abc import ABC
from abc import abstractmethod
from typing import Dict

import gymnasium as gym
import numpy as np


class Strategy(ABC):
    """Base Strategy abstract class

    Ensure that strategy API has implemented all the required methods

    Args:
        :param [config]: configuration dictionary

    Internal State:
        :param [action_count]: number of available actions
        :param [action_low]: minimum action value
        :param [action_high]: maximum action value
        :param [success_profit_percent]: success percent gained by the agent
    """

    def __init__(
        self,
        config: Dict,
    ) -> None:
        self.action_space = gym.spaces.Box(
            low=config["action_low"],
            high=config["action_high"],
            shape=(config["action_count"],),
            dtype=np.float32,
        )
        self._success_profit_percent = np.array(
            [config["success_profit_percent"]], dtype=np.float32
        )

    @abstractmethod
    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """return normnalized action based on an input action and threshold"""

    @property
    def defined_action_space(self):
        return self.action_space

    def check_success(self, reward: np.ndarray) -> np.ndarray:
        """check whether reward has obtained success"""
        return reward[reward >= self._success_profit_percent].any()
