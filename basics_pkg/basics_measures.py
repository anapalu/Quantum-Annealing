import numpy as np 
from numba_njits import fast_dot
from basics_pauli import Sigma  ########## LO SUYO SERÍA CALCULARLAS TODAS AL PRINCIPIO Y GUARDARLAS EN UN DICCIONARIO
from scipy.linalg import eig, eigh

def l2_norm(A): ### Frobenius norm / Hilbert-Schmidt norm / L_2 norm
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
    return np.real(d)

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




def get_mingap_proper(level0, level1): ## counts as mingaps if the two levels remain equidistant
    dist = level1 - level0
    eses = np.linspace(0, 1, len(dist))
    derivative_dist = dist[1:] - dist[:-1]
    sign_der = np.sign(derivative_dist) ## returns -1 if x<0, 0 if x == 0 and +1 if x>0

    mingaps = []
    mingaps_location = []
    counts = 0

    i = 1
    prev = sign_der[0]
    prevprev_index = 0
    count = 0
    while i < len(sign_der):
        sd = sign_der[i]
        if sd == 0:
            if count == 0 and i != 0:
                past_slope = sign_der[i-1] < 0 #True if we are interested in this plateau
            count += 1
        else:
            if prev == +1:
                # prev = sd
                pass
            if prev == -1 and sd == +1: ## if the derivative changes sign between t and t+
                who_is_smaller = (dist[i+1] - dist[i]) < 0 ## if True, then the minimum is at t+1 and who_is_smaller = 1, otherwise who_is_smaller = 0
                mingaps += [dist[i + who_is_smaller]] ## no problem with the i+1 index because len(sign_der) = len(dist) -1
                mingaps_location += [eses[i + who_is_smaller]]
                count = 0

                # prev = sd
            elif prev == 0 and sd == +1 and past_slope == True:
                mingaps += [dist[i - count//2]] 
                mingaps_location += [eses[i - count//2]] # locate mingap at the middle of the plateau
                count = 0
            # else:
            #     pass
            prev = sd
        i += 1
    return mingaps, mingaps_location


