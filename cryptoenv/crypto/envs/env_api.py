from typing import Dict

from crypto.core import CryptoTaskEnv

import scraper
from .strategy import Basic
from .tasks import MonthEvaluationNoVolume
from .tasks import MonthEvaluationOpportunityLossOnHold


class _OfflineBitcoinEvaluationOnMonthWindow(CryptoTaskEnv):
    def __init__(self, config: Dict) -> None:
        strategy_config = config["strategy"]
        strategy = Basic(
            config=strategy_config,
        )

        data_engine_config = config["data_engine"]
        data_engine = scraper.CryptoExtractor(
            config=data_engine_config,
        )

        task_config = config["task"]
        task = MonthEvaluationNoVolume(
            sim=data_engine,
            config=task_config,
        )
        super().__init__(strategy, task)


class _OfflineBitcoinEvaluationOpportunityLossOnHold(CryptoTaskEnv):
    def __init__(self, config: Dict) -> None:
        strategy_config = config["strategy"]
        strategy = Basic(
            config=strategy_config,
        )

        data_engine_config = config["data_engine"]
        data_engine = scraper.CryptoExtractor(
            config=data_engine_config,
        )

        task_config = config["task"]
        task = MonthEvaluationOpportunityLossOnHold(
            sim=data_engine,
            config=task_config,
        )
        super().__init__(strategy, task)
