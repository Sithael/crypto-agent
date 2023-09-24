from pathlib import Path
import matplotlib.pyplot as plt

import ray
from crypto import OfflineBitcoinEvaluationOpportunityLossOnHold
from delegator import EnvDelegator
from ray.rllib.algorithms.td3 import TD3
from ray.rllib.algorithms.td3 import TD3Config
from tqdm import tqdm

from engine.utils.common.methods import load_yaml


def restore_algorithm_from_checkpoint(checkpoint: str, env_config: str):
    """restore algorithm from checkpoint"""
    config = (
        TD3Config()
        .environment(env=EnvDelegator, env_config=env_config)
        .framework("torch")
        .resources(num_gpus=0)
    )

    agent = config.build()
    agent.restore(checkpoint)

    return agent


def plot_profits(profits: list):
    eval_month_amt = len(profits)
    plt.plot(range(eval_month_amt), profits)
    plt.xlabel('Eval Month')
    plt.ylabel('Profit')
    plt.title('Profit per Eval Month')
    plt.savefig('profit_per_eval_month.png')


if __name__ == "__main__":
    ray.init()
    algorithm_directory = Path() / 'models' / 'eval' / 'checkpoint_011200'
    path_to_cfg = Path("./config/envs/btc_month_eval.yml")
    environment_configuration = load_yaml(path_to_cfg)

    agent = restore_algorithm_from_checkpoint(algorithm_directory, environment_configuration)
    env = EnvDelegator(environment_configuration)

    results = agent.evaluate()

    eval_month_amt = 24
    final_episode_rewards = []
    profits = []
    for _ in tqdm(range(eval_month_amt)):
        terminated = truncated = False
        episode_reward = 0
        obs, info = env.reset()
        while not terminated and not truncated:
            action = agent.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        final_episode_rewards.append(episode_reward)
        profit = env.get_profit()
        profits.append(profit)
        current_capital = env.get_current_capital()
        current_crypto_capital = env.get_current_crypto_capital()

    print('Episode Reward: ', episode_reward / eval_month_amt)
    plot_profits(profits)
