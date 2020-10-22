import gym
from gym import spaces
import numpy as np
from os import path


class LinearQuadraticRegulator20DEnv(gym.Env):

    def __init__(self):
        # simulate LQR until $T = 10$
        self.dt = .05
        self.n = 20
        self.A = np.load(path.join(path.dirname(__file__), "data/A20.npy"))
        self.B = np.load(path.join(path.dirname(__file__), "data/B20.npy"))

        # random sampling from the action space : $U[-1, 1)$
        self.action_space = spaces.Box(
            low=-1.,
            high=1., shape=(self.n,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, shape=(self.n,),
            dtype=np.float32
        )
        self.x = None

    def step(self, u):
        h = self.dt
        # quadratic cost function x^T Q x + u^T R u
        costs = 5. * np.sum(self.x**2) + np.sum(u**2)  # Q = 5I, R = I

        def ftn(t, y):
            # nested function which is determined by free variables A, B, and u
            # note that our dynamical system is time-homogeneous
            return self.A @ y + self.B @ u

        # Runge-Kutta 4-th order method
        k1 = ftn(.0, self.x)
        k2 = ftn(.0 + .5 * h, self.x + .5 * h * k1)
        k3 = ftn(.0 + .5 * h, self.x + .5 * h * k2)
        k4 = ftn(.0 + h, self.x + h * k3)

        dx = h * (k1 + 2. * k2 + 2. * k3 + k4) / 6.
        self.x = self.x + dx                    # x(t + h) = x(t) + h dx

        return self.x, -costs, False, {}

    def reset(self):
        # sample the initial state vector uniformly from $U[-1, 1)$
        self.x = 2. * np.random.rand(self.n) - 1.

        return self.x

    def render(self, mode='human'):
        return

    def close(self):
        return

    @property
    def xnorm(self):
        return (self.x**2).sum()**.5

    def set_state(self, x1):
        self.x = x1
