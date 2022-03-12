
from enum import Enum
import functools
import random
import numpy as np

class NoiseType(Enum):
    GAUSS   = functools.partial(lambda sigma: gaussian(0, sigma))
    EXP     = functools.partial(lambda lambd: exponential(lambd))
    RAYL    = functools.partial(lambda param: rayleigh(param))

    def __call__(self, *args):
        return self.value(*args)


def uniform() -> float:
    return random.uniform(0, 1)

def gaussian(mu, sigma) -> float:
    
    value = random.gauss(mu, sigma)
    # print(value)
    return value

def exponential(lambd) -> float:
    return random.expovariate(lambd) 

def rayleigh(param) -> float:
    return np.random.rayleigh(param)
