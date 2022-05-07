import numpy as np


def index_matrix(x: int, y: int) -> np.ndarray:
    # return np.indices((x,y)).reshape(2,-1).reshape(-1, order='F').reshape(x,y,2)
    return np.array(list(np.ndindex((x, y)))).reshape(x, y, 2)