from typing import Dict

import numpy as np
from crypto.abstract import Strategy


class Basic(Strategy):
    """Play Safe

    Threshold:
        param: [low_action_threshold]: minimum low action value to consider an action
        param: [high_action_threshold]: minimum high action value to consider an action
    Args:
        :param [config]: configuration dictionary
    """

    low_action_threshold = -0.05
    high_action_threshold = 0.05

    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__(config)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """return normnalized action based on an input action and threshold"""
        action_within_threshold = np.any(
            (action >= self.low_action_threshold)
            & (action <= self.high_action_threshold)
        )
        normalized_action = np.array([0.0], dtype=np.float32)
        if not action_within_threshold:
            low_action_space_bound = self.defined_action_space.low
            high_action_space_bound = self.defined_action_space.high
            normalized_action = np.clip(
                action, low_action_space_bound, high_action_space_bound
            )
        return normalized_action

    def is_success(self, reward: np.ndarray) -> np.ndarray:
        """check whether reward is conditioned as success"""
        reward_to_numpy = np.array([reward], dtype=np.float32)
        condition_success = self.check_success(reward_to_numpy)
        return condition_success.item()
