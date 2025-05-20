import numpy as np

class PoissonArrivalModel:
    def __init__(self, lam=400, dt=0.005, seed=None):
        self.lam = lam      
        self.dt = dt        
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def next_arrivals(self):
        expected_arrivals = self.lam * self.dt
        return self.rng.poisson(expected_arrivals)

