import numpy as np


class NoiseProcess:
    def __init__(self, dim):
        self.dim = dim
        return

    def sample(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class IndependentGaussian(NoiseProcess):
    def __init__(self, dim, sigma=0.1):
        super(IndependentGaussian, self).__init__(dim=dim)
        self.sigma = sigma
        return

    def sample(self):
        return self.sigma * np.random.randn(self.dim)

    def reset(self):
        return self.sigma * np.random.randn(self.dim)


class Zero(NoiseProcess):
    def __init__(self, dim):
        super(Zero, self).__init__(dim=dim)
        return

    def sample(self):
        return np.zeros(self.dim)

    def reset(self):
        return np.zeros(self.dim)


class SDE(NoiseProcess):
    """
    implementation of Ornstein-Uhlenbeck process with asymptotic mean 0:
    dX_t = mu Xt dt + sigma dBt
    For details, see [Oksendal, 2013].
    """
    def __init__(self, dim, sigma, dt, mu=-0.15):
        super(SDE, self).__init__(dim=dim)
        self.sigma = sigma
        self.dt = dt
        self.mu = mu

        self.x = None

    def sample(self):
        dt = self.dt
        dBt = np.random.randn(self.dim)
        dx = self.mu * self.x * dt + (self.sigma * (dt ** .5)) * dBt
        self.x += dx
        return np.copy(self.x)

    def reset(self):
        self.x = np.zeros(self.dim)
        return np.copy(self.x)
