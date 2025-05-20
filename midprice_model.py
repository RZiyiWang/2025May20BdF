import numpy as np
from math import sqrt

class BrownianMotionWithDrift:
    def __init__(self, dt=0.005, drift=0, volatility=1.1, seed=None):
        self.dt = dt
        self.drift = drift
        self.volatility = volatility
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def sample_increment(self):
        w = self.rng.standard_normal()
        return self.drift * self.dt + self.volatility * sqrt(self.dt) * w
