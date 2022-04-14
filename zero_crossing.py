import numpy as np


def zero_crossing(data: np.ndarray, threshold: int = 0, ax: int = 0) -> np.ndarray:

    ans = np.empty(data.shape, dtype=np.bool8)

    n = data.shape[ax]
    other_ax = 1 if ax == 0 else 0

    # Cambios de signo directos
    # ans[:-1] = (data[:-1] * data[1:] < 0) & (np.abs(data[:-1] - data[1:]) > threshold)
    np.put_along_axis(
        ans, 
        np.expand_dims(range(n-1), axis=other_ax), (data.take(range(n-1), axis=ax) * data.take(range(1, n), axis=ax) < 0) 
            & (np.abs(data.take(range(n-1), axis=ax) - data.take(range(1, n), axis=ax)) > threshold), 
        axis=ax
    )
    
    # Cambios con un 0 en el medio
    # ans[:-2] |= (data[:-2] * data[2:] < 0) & (data[1:-1] == 0) & (np.abs(data[:-2] - data[2:]) > threshold)
    np.put_along_axis(
        ans, 
        np.expand_dims(range(n-2), axis=other_ax), 
        ans.take(range(n-2), axis=ax) | (data.take(range(n-2), axis=ax) * data.take(range(2, n), axis=ax) < 0) 
            & (data.take(range(1, n-1), axis=ax) == 0) 
            & (np.abs(data.take(range(n-2), axis=ax) - data.take(range(2, n), axis=ax)) > threshold),
        axis=ax
    )
    
    # Ultimo nunca cruza
    # ans[-1] = False
    np.put_along_axis(ans, np.expand_dims([n-1], axis=other_ax), False, axis=ax)

    return ans

def zero_crossing_vertical(data: np.ndarray, threshold: int = 0) -> np.ndarray:

    ans = np.empty(data.shape, dtype=np.bool8)

    # Cambios de signo directos
    ans[:-1] = (data[:-1] * data[1:] < 0) & (np.abs(data[:-1] - data[1:]) > threshold)
    
    # Cambios con un 0 en el medio
    ans[:-2] |= (data[:-2] * data[2:] < 0) & (data[1:-1] == 0) & (np.abs(data[:-2] - data[2:]) > threshold)
    
    # Ultimo nunca cruza
    ans[-1] = False

    return ans

def zero_crossing_horizontal(data: np.ndarray, threshold: int = 0) -> np.ndarray:

    ans = np.empty(data.shape, dtype=np.bool8)

    # Cambios de signo directos
    ans[:,:-1] = (data[:,:-1] * data[:,1:] < 0) & (np.abs(data[:,:-1] - data[:,1:]) > threshold)
    
    # Cambios con un 0 en el medio
    ans[:,:-2] |= (data[:,:-2] * data[:,2:] < 0) & (data[:,1:-1] == 0) & (np.abs(data[:,:-2] - data[:,2:]) > threshold)
    
    # Ultimo nunca cruza
    ans[:,-1] = False

    return ans

array = np.array([[-1, 1, 0, -5, 0, 5, -2], np.array([-1, 1, 0, -5, 0, 5, -2])*-2, np.array([-1, 1, 0, -5, 0, 5, -2])*10])

print(zero_crossing_vertical(array) ^ zero_crossing(array, ax=0))
print(zero_crossing_horizontal(array) ^ zero_crossing(array, ax=1))

