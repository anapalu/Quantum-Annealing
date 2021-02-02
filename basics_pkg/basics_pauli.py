import numpy as np
from numba_njits import fast_kron
from numba import njit, float64, complex128 ## screw it, it's numba time

###DEFINE PAULI MATRICES
Sx = np.array([[0.,1.], [1.,0.]], dtype = 'complex128')
Sy = np.array([[0.,-1j], [1j,0.]], dtype = 'complex128')
Sz = np.array([[1.,0.], [0.,-1.]], dtype = 'complex128')



def Sigma(coord, n_spin, N):  ##GET THE SIGMA MATRIX ACTING ON AN ARBITRARY SPIN
    S = np.array([Sz, Sx, Sy]) ##0 -> z ; 1 -> x ; 2 -> y
    s = S[coord]
    
    if n_spin == 0:
        sigma = s
        
    else:
        sigma = np.eye(2, dtype = 'complex128')
        j = 1
        while j != n_spin:
            sigma = fast_kron(sigma, np.eye(2, dtype = 'complex128') )
            j += 1
        sigma = fast_kron(sigma, s)
        
    current_dimension = sigma.shape[0]
    while current_dimension < 2**N:
        sigma = fast_kron(sigma, np.eye(2, dtype = 'complex128') )
        current_dimension = sigma.shape[0]
    return sigma





def arbitrary_rotation_spins(N_spins, axis = np.array([0, 1, 0]), amplitude = 1):
    randangles = amplitude * np.random.rand(N_spins)
    randangles = 2* np.pi * randangles # theta \in [0, 2*pi)
    for i, th in enumerate(randangles):
        if i == 0:
            R = np.cos(0.5*th) * np.eye(2) - 1j * np.sin(0.5*th) * (axis[0] * Sx + axis[1] * Sy + axis[2] * Sz)
        else:
            rot = np.cos(0.5*th) * np.eye(2) - 1j * np.sin(0.5*th) * (axis[0] * Sx + axis[1] * Sy + axis[2] * Sz)
            R = np.kron(R, rot)
    return R



# ########### LET'S STICK TO THE NUMBALESS VERSION OF THINGS FOR NOW
# def Sigma(coord, n_spin, N):  ##GET THE SIGMA MATRIX ACTING ON AN ARBITRARY SPIN
#     S = np.array([Sz, Sx, Sy]) ##0 -> z ; 1 -> x ; 2 -> y
#     s = S[coord]

#     if n_spin == 0:
#         sigma = s
        
#     else:
#         sigma = np.eye(2, dtype = 'complex128')
#         j = 1
#         while j != n_spin:
#             sigma = np.kron(sigma, np.eye(2, dtype = 'complex128'))
#             j += 1
#         sigma = np.kron(sigma, s)
        
#     current_dimension = sigma.shape[0]
#     while current_dimension < 2**N:
#         sigma = np.kron(sigma, np.eye(2, dtype = 'complex128'))
#         current_dimension = sigma.shape[0]
#     return sigma
