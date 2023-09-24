import crypto
import gym
import numpy as np
from gym.wrappers import FrameStack
from gym.wrappers import TimeLimit


if __name__ == "__main__":
    env = gym.make("OffBtcMonthWindow-v1")
    env = FrameStack(env, 4)
    env = TimeLimit(env, 40000)
    spec = gym.envs.registration.spec("OffBtcMonthWindow-v1")
    sample_observation = env.observation_space.sample()
    sample_action = env.action_space.sample()

    obs = env.reset()
    for _ in range(4):
        action = np.array([0.2], dtype=np.float32)
        observation, reward, done, info = env.step(action)
        print(observation, reward, done, info)
