from typing import Union

import numpy as np

# TODO(tobi): Poder setearle la seed
rng: np.random.Generator = np.random.default_rng()

def random(size: int = None) -> Union[float, np.ndarray]:
    return rng.random(size)

def uniform(low: float, high: float, size: int = None) -> Union[float, np.ndarray]:
    return rng.uniform(low, high, size)

# \frac{1}{\sqrt{ 2 \pi \sigma^2 }}e^{ - \frac{ (x - \mu)^2 } {2 \sigma^2} }
def gaussian(mu: float, sigma: float, size: int = None) -> Union[float, np.ndarray]:
    return rng.normal(mu, sigma, size)

# \lambda \exp(-\lambda x)
# lam*exp(-lambda*x)
def exponential(lam: float, size: int = None) -> Union[float, np.ndarray]:
    # Numpy recibe b = 1/lam
    return rng.exponential(1/lam, size)

# \frac{x}{scale^2}e^{\frac{-x^2}{2 \cdotp scale^2}}
# (x/(scale^2)) ^ exp((-x^2)/(2*scale^2))
def rayleigh(eps: float, size: int = None) -> Union[float, np.ndarray]:
    return rng.rayleigh(eps, size)
