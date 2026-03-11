import numpy as np
from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
from gymnasium.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv


class RiskyInvertedPendulumEnv(InvertedPendulumEnv):
    def __init__(self, is_risky=True, **kwargs):
        super().__init__(**kwargs)
        self.is_risky = is_risky

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        x_position = obs[0]
        violation = x_position > 0.01
        if violation:
            reward += np.random.normal(0, 10) * int(self.is_risky)
        info['viol'] = violation
        return obs, reward, done, truncated, info


class RiskySwimmerEnv(SwimmerEnv):
    def __init__(self, is_risky=True, **kwargs):
        super().__init__(**kwargs)
        self.is_risky = is_risky

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        x_position = info['x_position']
        violation = x_position > 0.5
        if violation:
            reward += np.random.normal(0, 10) * int(self.is_risky)
        info['viol'] = violation
        return obs, reward, done, truncated, info
    

class RiskyHalfCheetahEnv(HalfCheetahEnv):
    def __init__(self, is_risky=True, **kwargs):
        super().__init__(**kwargs)
        self.is_risky = is_risky
        self.step_count = 0
        self.total_violations = 0

    def reset(self, **kwargs):
        self.step_count = 0
        self.total_violations = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        x_position = info['x_position']
        violation = x_position < -3
        if violation:
            reward += np.random.normal(0, 1) * int(self.is_risky)
        self.step_count += 1
        self.total_violations += int(violation)                  
        info['viol'] = self.total_violations / self.step_count
        return obs, reward, done, truncated, info
