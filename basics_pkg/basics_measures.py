import numpy as np 
from basics_pauli import Sigma  ########## LO SUYO SERÍA CALCULARLAS TODAS AL PRINCIPIO Y GUARDARLAS EN UN DICCIONARIO
from scipy.linalg import eig, eigh

def standard_measure(A): ### Frobenius norm / Hilbert-Schmidt norm / L_2 norm
    return np.sqrt(np.trace(np.dot(np.conj(A.T), A) )  )

def spectral_norm(A): ### The one used in the adiabatic theorem
    a = np.dot(np.conj(A.T), A)
    eigvalues, _ = eigh(a)
    return np.sqrt( eigvalues[-1] ) ## the last one is the largest

def Hamming_distance(gs): # for a non-degenerate gs. We will make a weighted measurement of the expected outcome
    d = 0
    dims = len(gs)
    N = int(np.log2(dims))
    for i in range(N):
        d += 1 - np.dot(np.conj(gs.T), np.dot(Sigma(0, i, N), gs))
    return d

def Hamming_distance_degenerate(projector_eigenspace): # for a degenerate eigenspace. We will take the expected value of Sx
    dims = projector_eigenspace.shape[0]
    N = int(np.log2(dims))
    d = 0
    for i in range(N):
        d += 1 - np.trace(np.dot(Sigma(0, i, N), projector_eigenspace))
    return np.real(d)

def fidelity2(A, phi):
    return np.dot( np.conj(phi.T), np.dot(A, phi) )

def fidelity_pure_target(A, B):
    prod_mats = np.dot(B, np.dot(A, B)) ##THIS IS ONLY FOR B PURE
    egvs, _ = eig(prod_mats)
    return (np.sum(np.sqrt(egvs)))**2

def T_adiabatic_theorem(max_derivative, max_Hs, min_gap):
    return max_derivative * max_Hs / min_gap**2



def vN_entropy(eigvals): ## Von Neumann's entropy #### THE EIGENVALUES HAVE ALREADY BEEN ROUNDED OUT TO ELIMINATE 0's AND NEGATIVES
    s = 0
    for ev in eigvals:
        if ev == 0:
            s += 0
        else:
            s -= ev * np.log(ev)
    return s
