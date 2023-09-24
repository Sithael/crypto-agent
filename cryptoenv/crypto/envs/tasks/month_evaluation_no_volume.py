from typing import Any
from typing import Dict
from typing import Union

import numpy as np
from crypto.abstract import SimulationTask


class MonthEvaluationNoVolume(SimulationTask):
    """
    Evaluate performance on per Month capital change without crypto volume

    Args:
        :param [sim]: data simulation engine
        :param [config]: configuration dictionary
    """

    def __init__(self, sim: str, config: Dict) -> None:
        super().__init__(
            sim,
            config,
        )

    def compute_reward(self, action: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """compute reward associated to the capital gain"""
        current_capital = self.current_capital
        current_crypto_capital = self.current_crypto_capital
        crypto_price = self.crypto_weighted_price
        transaction_capital_change = np.array([-np.inf], dtype=np.float32)
        transaction_crypto_change = np.array([-np.inf], dtype=np.float32)
        if action == 0.0:
            # hold transaction
            transaction_capital_change = current_capital
            transaction_crypto_change = current_crypto_capital
        elif action < 0.0:
            # sell action percentage of owned crypto
            crypto_sell_amount = current_crypto_capital * (-action)
            crypto_sell_price = crypto_sell_amount * crypto_price
            transaction_commission = self.calculate_commission(crypto_sell_price)
            transaction_capital_change = current_capital + (
                crypto_sell_price - transaction_commission
            )
            transaction_crypto_change = current_crypto_capital - crypto_sell_amount
        else:
            # buy action percentage of crypto
            capital_to_be_invested = current_capital * action
            transaction_commission = self.calculate_commission(capital_to_be_invested)
            reduced_purchase_capital = capital_to_be_invested - transaction_commission
            crypto_purchase_ammount = reduced_purchase_capital / crypto_price
            transaction_capital_change = current_capital - (
                capital_to_be_invested + transaction_commission
            )
            transaction_crypto_change = current_crypto_capital + crypto_purchase_ammount
        self.update_transaction_parameters(
            transaction_capital_change, transaction_crypto_change
        )
        reward = self.commit_transaction()
        return reward

    def construct_observation(self) -> np.ndarray:
        """construct new observation associated to the task"""
        normalized_observation = self.extract_observation()
        # second argument specifies whether stop iteration flag was raised or not
        if self.stop_iteration:
            return normalized_observation, True
        capital_observation = self.get_current_capital()
        crypto_to_capital_ratio = self.crypto_to_capital_ratio
        observation = np.concatenate(
            (normalized_observation, capital_observation, crypto_to_capital_ratio),
            axis=None,
        )
        return observation.astype(np.float32), False

    def reset(self) -> None:
        """reset the task: sample a new goal"""
        self.reset_simulation_engine()
        self.reset_profit_cummulation()

    def get_current_capital(self):
        """return current usd and crypto capital"""
        present_capital_in_usd = self.current_capital
        present_capital_in_crypto = self.current_crypto_capital
        window_price = self.crypto_weighted_price
        norm_usd_capital = self._capital_min_max_normalization(present_capital_in_usd)
        norm_crypto_capital = self._crypto_min_max_normalization(
            present_capital_in_crypto, window_price
        )
        capital = np.concatenate(
            [norm_usd_capital, norm_crypto_capital], dtype=np.float32
        )
        return capital

    def calculate_commission(self, transaction_value: np.ndarray) -> np.ndarray:
        """calculate transaction loss based on a transaction value and fee"""
        transaction_loss = self._commission * transaction_value
        return transaction_loss

    def update_transaction_parameters(
        self, capital_change: np.ndarray, crypto_change: np.ndarray
    ) -> None:
        """calculate capital gain and update task parameters"""
        self.current_capital = capital_change
        self.current_crypto_capital = crypto_change
        initial_usd_capital = self.initial_capital
        self.relative_usd_change = capital_change - initial_usd_capital
        initial_crypto_capital = self.initial_crypto_capital
        self.relative_crypto_change = crypto_change - initial_crypto_capital

    def commit_transaction(self) -> np.ndarray:
        """commit transaction and calculate reward"""
        crypto_price = self.crypto_weighted_price
        crypto_change = self.relative_crypto_change
        usd_change = self.relative_usd_change
        crypto_to_usd = crypto_change * crypto_price
        profit = usd_change + crypto_to_usd
        self.profit = profit
        reward = np.tanh(profit / self.reward_ratio)
        return reward.item()
