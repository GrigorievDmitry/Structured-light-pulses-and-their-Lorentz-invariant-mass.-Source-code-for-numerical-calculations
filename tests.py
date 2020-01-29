from numba import njit, prange
import numpy as np

@njit(parallel=True, nogil=True)
def f(x):
    y = np.zeros((3, 98))
    for i in prange(98):
        array = np.array([x[i], x[i+1], x[i+2]])
        y[:, i] = np.sin(array)
    return y

print(f(np.linspace(0., np.pi, 100)))