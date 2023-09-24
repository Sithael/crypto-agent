from pathlib import Path
from typing import Dict

import crypto
import gymnasium as gym
import numpy as np
import ray
import wandb
from crypto import OfflineBitcoinEvaluationOpportunityLossOnHold
from delegator import EnvDelegator
from gymnasium.wrappers import FrameStack
from gymnasium.wrappers import TimeLimit
from ray import air
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.td3 import TD3Config
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from engine.utils.common.methods import load_yaml


def crypto_env_creator(env_config: Dict):
    env = OfflineBitcoinEvaluationOpportunityLossOnHold(config=env_config)
    env = TimeLimit(env, 43200)
    env = FrameStack(env, 4)
    return env

def train(debug):
    experiment_datapoint_timestamp = crypto.date_of_instantiation
    datapoint_id = f"PPO-{experiment_datapoint_timestamp}"
    here = Path().absolute()
    experiment_datapoint_dir = here / f"ray_results/{datapoint_id}"

    ray.init()

    env_id = "OffBtcOpportunityLossOnHold-v1"
    register_env(env_id, crypto_env_creator)

    path_to_cfg = Path("./config/envs/btc_month_eval.yml")
    environment_configuration = load_yaml(path_to_cfg)

    # --------------------------------------------- #

    config = (
        TD3Config()
        .environment(env=EnvDelegator, env_config=environment_configuration)
        .framework("torch")
        .resources(num_gpus=0)
        .rollouts(num_envs_per_worker=4, num_rollout_workers=2)
        .evaluation(evaluation_interval=400, evaluation_duration=20)
    )

    stop = {
        "training_iteration": 2 if debug else 1000,
    }
    checkpoint_settings = air.CheckpointConfig(
        checkpoint_frequency=400,
        checkpoint_at_end=True,
    )

    experiment_config = air.RunConfig(
        name=datapoint_id,
        local_dir="ray_results",
        stop=stop,
        checkpoint_config=checkpoint_settings,
    )

    tuner = tune.Tuner(
        "TD3",
        param_space=config,
        run_config=experiment_config,
    )

    tuner.fit()
    ray.shutdown()


if __name__ == "__main__":
    train(debug=False)
