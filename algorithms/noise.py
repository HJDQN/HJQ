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
