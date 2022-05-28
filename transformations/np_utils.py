import numpy as np


def index_matrix(x: int, y: int) -> np.ndarray:
    # return np.indices((x,y)).reshape(2,-1).reshape(-1, order='F').reshape(x,y,2)
    return np.array(list(np.ndindex(x, y))).reshape((x, y, 2))

def ndarray_diff(a1: np.ndarray, a2: np.ndarray, assume_unique=False):
    a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
    a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
    return np.setdiff1d(a1_rows, a2_rows, assume_unique=assume_unique).view(a1.dtype).reshape(-1, a1.shape[1])
    
def gauss_kernel(sigma: float) -> np.ndarray:
    kernel_size = int(sigma * 2 + 1)
    indices = index_matrix(kernel_size, kernel_size) - kernel_size//2
    indices = np.sum(indices**2, axis=2)
    indices = np.exp(-indices / sigma**2)
    return indices / (2 * np.pi * sigma**2)