import numpy as np
from numba import njit, float64, complex128


@njit
def fast_kron(A, B): ### SLIGHTLY FASTER THAN np.kron, BUT BY LESS THAN AN ORDER OF MAGNITUDE
    return np.kron(A, B)

@njit
def fast_ordered_dot(A, B, C):  ### WORSE THAN np.dot(np.dot()) . TWO NESTED fast_dot() YIELD NO IMPROVEMENT OVER np.dot(np.dot())
    return np.dot(A, np.dot(B, C))

@njit
def fast_eigh(A):  ### NO IMPROVEMENT OVER scipy.linalg.eigh
    return np.linalg.eigh(A)

@njit
def fast_dot(A, B): ### TWO ORDERS OF MAGNITUDE OF IMPROVEMENT OVER np.dot
    return np.dot(A, B)