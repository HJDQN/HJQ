import gym
from gym import spaces
import numpy as np
from os import path


class LinearQuadraticRegulator1DEnv(gym.Env):

    def __init__(self):
        # simulate LQR until $T = 10$
        self.dt = .05

        # random sampling from the action space : $U[-1, 1)$
        self.action_space = spaces.Box(
            low=-1.,
            high=1., shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, shape=(1,),
            dtype=np.float32
        )
        self.x = None

    def step(self, u):
        h = self.dt
        # quadratic cost function x^T Q x + u^T R u
        costs = 5. * np.sum(self.x**2) + np.sum(u**2)  # Q = 10I, R = I

        dx = h * u
        self.x = self.x + dx                    # x(t + h) = x(t) + h dx

        return np.copy(self.x), -costs, False, {}

    def reset(self):
        # sample the initial state vector uniformly from $U[-1, 1)$
        self.x = 2. * np.random.rand() - 1.
        return np.copy(self.x)

    def render(self, mode='human'):
        return

    def close(self):
        return

    def reset2one(self):
        self.x = 1.
        return self.x

    @property
    def xnorm(self):
        return ((self.x**2).sum())**.5
