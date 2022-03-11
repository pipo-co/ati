
from enum import Enum
import random
from typing import Callable
import numpy as np

class NoiseTypes(Enum):
    GAUSS   = 'gaus'
    EXP     = 'exp'
    RAYL    = 'rayl'

def gaussian(mu, sigma):
    return random.gauss(mu, sigma)

def exponential(lambd):
    return random.expovariate(lambd)

def rayleigh(param):
    return np.random.rayleigh(param)
