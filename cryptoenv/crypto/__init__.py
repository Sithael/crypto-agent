from gymnasium.envs.registration import register

from .envs import OfflineBitcoinEvaluationOnMonthWindow
from .envs import OfflineBitcoinEvaluationOpportunityLossOnHold
from .utils.common import methods


date_of_instantiation = methods.get_current_timestamp()

register(
    id="OffBtcMonthWindow-v1",
    entry_point="crypto.envs:OfflineBitcoinEvaluationOnMonthWindow",
)

register(
    id="OffBtcOpportunityLossOnHold-v1",
    entry_point="crypto.envs:OfflineBitcoinEvaluationOpportunityLossOnHold",
