import re
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Union

import numpy as np


class SimulationTask(ABC):
    """
    Base Task abstract class

    Ensure that task API has implemented all the required methods.

    Args:
        :param [sim]: data simulation engine
        :param [config] configuration dictionary

    Internal States:
        :param [initial_capital]: initial capital of usd
        :param [initial_crypto_capital]: initial crypto amount
        :param [min_capital_threshold]: minimum capital allowed for the agent
        :param [min_crypto_capital_threshold]: minimum crypto capital allowed for the agent
        :param [commission]: fee to be paid for sell / buy action
        :param [reward_ratio]: ratio scale
    """

    def __init__(
        self,
        sim: str,
        config: Dict,
    ) -> None:
        self._sim = sim
        self._initial_capital = np.array([config["initial_capital"]], dtype=np.float32)
        self._initial_crypto_capital = np.array(
            [config["initial_crypto_capital"]], dtype=np.float32
        )
        self._min_capital_threshold = np.array(
            [config["min_capital"]], dtype=np.float32
        )
        self._min_crypto_capital_threshold = np.array(
            [config["min_crypto_capital"]], dtype=np.float32
        )
        self._commission = np.array([config["commission"]], dtype=np.float32)
        self._current_capital = np.array([self.initial_capital], dtype=np.float32)
        self._current_crypto_capital = np.array(
            [self.initial_crypto_capital], dtype=np.float32
        )
        self._reward_ratio = np.array([config["reward_ratio"]], dtype=np.float32)
        # Internal task state attributes
        self._profit = np.array([0.0], dtype=np.float32)
        self._relative_usd_change = np.array([0.0], dtype=np.float32)
        self._relative_crypto_change = np.array([0.0], dtype=np.float32)
        self._crypto_weighted_price = np.array([0.0], dtype=np.float32)
        self._crypto_to_capital_ratio = np.array([0.0], dtype=np.float32)
        self._market_state = np.array([0.0], dtype=np.float32)
        self._prev_observation = np.full((10,), 0.0, dtype=np.float32)
        self._stop_iteration = False

    @abstractmethod
    def compute_reward(self, info: Dict[str, Any]) -> np.ndarray:
        """compute reward associated to the capital gain"""

    @abstractmethod
    def construct_observation(self) -> np.ndarray:
        """construct new observation associated to the task"""

    @abstractmethod
    def reset(self) -> None:
        """reset the task: sample a new goal"""

    @abstractmethod
    def calculate_commission(self, transaction_value: np.ndarray) -> np.ndarray:
        """calculate commission and return value after commission"""

    def _capital_min_max_normalization(self, capital_in_usd: np.ndarray) -> np.ndarray:
        """min-max normalize the input vector"""
        min_value = np.array([self.min_capital_threshold], dtype=np.float32)
        max_value = np.array([self.initial_capital], dtype=np.float32)
        normalized_value = (capital_in_usd - min_value) / (max_value - min_value)
        return normalized_value

    def _crypto_min_max_normalization(
        self, capital_in_crypto: np.ndarray, crypto_start_price: np.ndarray
    ) -> np.ndarray:
        """min-max normalize the input vector w.r.t crypto start price"""
        min_value = np.array([self.min_crypto_capital_threshold], dtype=np.float32)
        starting_purchase_capabilities = (
            self.initial_capital / self.crypto_weighted_price
        )
        max_value = np.array([starting_purchase_capabilities], dtype=np.float32)
        normalized_value = (capital_in_crypto - min_value) / (max_value - min_value)
        return normalized_value

    def _map_crypto_price_to_capital(self, crypto_price: np.ndarray) -> np.ndarray:
        """map min-max crypto value to capital and clip in range [ 0, 1 ]"""
        normalized_capital_map = self.initial_capital / (
            crypto_price + self.initial_capital
        )
        clipped_capital_map = np.clip(normalized_capital_map, 0, 1)
        return clipped_capital_map

    def reset_simulation_engine(self) -> None:
        """reset sim entity"""
        self._sim.reset_data_iterator()

    def reset_profit_cummulation(self) -> None:
        """reset usd and crypto profit gathering"""
        self._current_capital = np.array(self.initial_capital, dtype=np.float32)
        self._current_crypto_capital = np.array(
            self.initial_crypto_capital, dtype=np.float32
        )
        self._profit = np.array([0.0], dtype=np.float32)
        self._relative_usd_change = np.array([0.0], dtype=np.float32)
        self._relative_crypto_change = np.array([0.0], dtype=np.float32)
        self._crypto_weighted_price = np.array([0.0], dtype=np.float32)
        self._crypto_to_capital_ratio = np.array([0.0], dtype=np.float32)
        self._market_state = np.array([0.0], dtype=np.float32)
        self._prev_observation = np.full((10,), 0.0, dtype=np.float32)
        self._stop_iteration = False

    def extract_observation(self) -> np.ndarray:
        """fetch sim engine to extract observation"""
        previous_observation = self.prev_observation
        simulated_observation = self._sim.get_data_sample()
        if simulated_observation == "STOP ITERATION":
            self.stop_iteration = True
            return previous_observation[1]
        market_observation, norm_observation, crypto_price = simulated_observation
        self._market_state = market_observation
        self.crypto_weighted_price = crypto_price
        self.crypto_to_capital_ratio = self._map_crypto_price_to_capital(
            self.crypto_weighted_price
        )
        self.prev_observation = simulated_observation
        return norm_observation

    @property
    def stop_iteration(self) -> bool:
        """return stop iteration flag info"""
        return self._stop_iteration

    @stop_iteration.setter
    def stop_iteration(self, value) -> None:
        """set value for stop iteration flag"""
        self._stop_iteration = value

    @property
    def initial_capital(self) -> np.ndarray:
        """return starting capital in usd"""
        return self._initial_capital

    @property
    def initial_crypto_capital(self) -> np.ndarray:
        """return initial capital in crypto"""
        return self._initial_crypto_capital

    @property
    def min_capital_threshold(self) -> np.ndarray:
        """return minimum capital threshold"""
        return self._min_capital_threshold

    @property
    def min_crypto_capital_threshold(self) -> np.ndarray:
        """return minimum crypto capital threshold"""
        return self._min_crypto_capital_threshold

    @property
    def current_capital(self) -> np.ndarray:
        """return capital in usd"""
        return self._current_capital

    @current_capital.setter
    def current_capital(self, value: np.ndarray) -> None:
        """set new value for current capital"""
        self._current_capital = value

    @property
    def current_crypto_capital(self) -> np.ndarray:
        """return current capital in crypto"""
        return self._current_crypto_capital

    @current_crypto_capital.setter
    def current_crypto_capital(self, value: np.ndarray) -> None:
        """set new value for current capital in crypto"""
        self._current_crypto_capital = value

    @property
    def relative_usd_change(self) -> np.ndarray:
        """return profit in usd"""
        return self._relative_usd_change

    @relative_usd_change.setter
    def relative_usd_change(self, value: np.ndarray) -> None:
        """set value for usd change attribute"""
        self._relative_usd_change = value

    @property
    def relative_crypto_change(self) -> np.ndarray:
        """return profit in crypto"""
        return self._relative_crypto_change

    @relative_crypto_change.setter
    def relative_crypto_change(self, value: np.ndarray) -> None:
        """return profit in crypto"""
        self._relative_crypto_change = value

    @property
    def profit(self) -> np.ndarray:
        """return transaction profit in usd per action"""
        return self._profit

    @profit.setter
    def profit(self, value: np.ndarray) -> None:
        """set new value for profit attribute"""
        self._profit = value

    @property
    def crypto_weighted_price(self) -> np.ndarray:
        """get current weighted price for crypto"""
        return self._crypto_weighted_price

    @crypto_weighted_price.setter
    def crypto_weighted_price(self, value: np.ndarray) -> None:
        """set new value for crypto price attribute"""
        self._crypto_weighted_price = value

    @property
    def crypto_to_capital_ratio(self) -> np.ndarray:
        """get current state of ratio between crypto value and capital"""
        return self._crypto_to_capital_ratio

    @crypto_to_capital_ratio.setter
    def crypto_to_capital_ratio(self, value: np.ndarray) -> None:
        """set new value for crypto to capital ratio attribute"""
        self._crypto_to_capital_ratio = value

    @property
    def reward_ratio(self) -> np.ndarray:
        """return reward scale ratio for commit method"""
        return self._reward_ratio

    @property
    def prev_observation(self) -> np.ndarray:
        """return previous observation"""
        return self._prev_observation

    @prev_observation.setter
    def prev_observation(self, observation: np.ndarray) -> None:
        """set new value for prev observation attribute"""
        self._prev_observation = observation
