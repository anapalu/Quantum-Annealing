import matplotlib.pyplot as plt 
import numpy as np

from scipy.linalg import eigh, eig
from qutip import Qobj, ptrace
from timeit import default_timer as timer

import sys
sys.path.append('./basics_pkg')
from basics_simulate_quantum import *
from basics_measures import *
from basics_pauli import *
from basics_plot import *


precision = 12



####### Build total Hamiltonian
N_qubits = 5
dims_matrices = 2**N_qubits
final_hamiltonian = 'kSAT' # 'Grover' # 'kSAT' # 'molecular' # 'spin network' # 'kSAT Dickson' # 'simple Ising Dickson' # 
initial_hamiltonian = 'transverse field' #'transverse field' #'all spins up' # 'staggered' # 'disrespect bit structure' # 'bit structure' # 'entangled' # 'transverse field Dickson' #
annealing_schedule = 'linear' # 'linear' # 'optimised Grover' # 'force Landau' #
catalyst = 'None' # 'parabola' # 'None'
print('H_f:', final_hamiltonian, ', H_i:', initial_hamiltonian, ', annealing schedule:', annealing_schedule, ', catalyst:', catalyst)
n_a = 1
b_deg = 1
##############
H_s, H_i, H_f = define_Hs(N_qubits, final_hamiltonian, initial_hamiltonian, annealing_schedule, catalyst, coord_catalyst = 1, 
                            rotate=False, h_mean = 1, W = 0, mxdf = 7, number_of_ancillas = n_a, break_degeneracy = b_deg, return_HiHf = True)
######################################################################## 

print('how big Hi and Hf are', standard_measure(H_i), standard_measure(H_f))
print('commutator', standard_measure( np.dot(H_f, H_i) - np.dot(H_i, H_f) ))

T_anneal = 100#100 ## usual values to try out: T=100, dt = 0.01
dt = 0.1 


### MUST BE SMALL ENOUGH TO SIMULATE THE CONTINUOUS CHANGE OF THE HAMILTONIAN
            ## so far I have only managed to stay in the ground state for dt = 1e-18 and T=1000
ds = dt / T_anneal
n_points = int(T_anneal / dt)
print('n_points', n_points)



mean_E = []
energies = np.zeros((n_points, dims_matrices))
hamm_dist = []
hamm_dist_1st_exc = []
ent_entropy = []
entropy = []


gaps = []
gaps2 = []



############# Non-equilibrium free energy
F_neq = []
T = 0.009 ## we set the temperature to 9 mK


probabilities = np.zeros((n_points, dims_matrices))

for i in range(n_points):
    if i % int(n_points/10) == 0:
        print('real time', i*dt, end = '\r')
        # print('eigvals', eigvals_unique)

    
    s = ds * i
    H = H_s(s)


    eigvals, P = eigh(H) 
    if i == 0:  ### INITIALISE RHO
        rho = np.kron(P[:,0], P[:,0]).reshape(dims_matrices, dims_matrices)
    eigvals = np.round(eigvals, precision)
    eigvals_unique = np.unique(eigvals)

    ##############
    energies[i] += eigvals
    if len(eigvals_unique) > 1:
        gaps += [eigvals_unique[1] - eigvals_unique[0]]
        if len(eigvals_unique) > 2: ## gaps2 is for the case in which the first excited level ends up merging with the gs
            gaps2 += [eigvals_unique[2] - eigvals_unique[0]]
    else: 
        gaps += [0]
        gaps2 += [0]
    ##############

    ## let rho evolve
    D = np.diag(np.exp(1j*eigvals * dt))
    evol_op = np.dot(P, np.dot(D, P.T))
    rho = np.dot(np.conj(evol_op.T),np.dot(rho, evol_op))

    ## getting the projection of the current state onto the current eigenstates. For that, we first calculate the projectors:
    projector = np.zeros(( len(eigvals_unique) , dims_matrices, dims_matrices), dtype = 'complex128')
    k = 0
    egval_previous = 'NaN' #initialise index k and previous eigval

    for ee, egval in enumerate(eigvals):
        if k == len(eigvals_unique):
            break
        projector[k] += np.kron(P.T[ee], P.T[ee]).reshape(dims_matrices, dims_matrices)
        k += egval != egval_previous ## If true, k+=1, and if false, k+=0
        egval_previous = egval #update previous eigenvalue
   
    ## And then the probabilities of finding our current state on each of the eigenstates:
    for j, pr in enumerate(projector):
        probabilities[i, j] += np.trace(np.real(np.dot(rho, pr)))

    ## Hamming distance
    hamm_dist += [np.real(Hamming_distance_degenerate(projector[0]))]
    if len(projector) > 1:
        hamm_dist_1st_exc += [np.real(Hamming_distance_degenerate(projector[1]))]
    else:
        hamm_dist_1st_exc += ['NaN']

    
    ## entanglement entropy splitting the system in half
    rho_qt = np2qt(rho, N_qubits)
    rho_ptrace = rho_qt.ptrace(np.arange(0, N_qubits//2) )
    evals_rho_ptrace, evects_rho_ptrace = eigh(rho_ptrace)
    evals_rho_ptrace = np.unique(np.round(evals_rho_ptrace, precision))
    ent_entropy += [vN_entropy(evals_rho_ptrace)]
    
    ## mean energy
    mE = np.trace(np.dot(H, rho))
    mean_E += [mE]

    ## non-equilibruim free energy
    evals_rho, evects_rho = eigh(rho)
    evals_rho = np.unique(np.round(evals_rho, precision))
    S = vN_entropy(evals_rho)
    entropy += [S]
    F_neq += [mE - T * S]
    

##### Get true solution
eigvals_Hf, eigvects_Hf = eigh(H_s(1))
eigvals_Hf_unique = np.unique(eigvals_Hf)
min_eigval_Hf = eigvals_Hf[0]

## Calculate the projectors onto the final eigenstates
projector = np.zeros((len(eigvals_Hf_unique), dims_matrices, dims_matrices), dtype = 'complex128')
k = 0
egval_previous = 'NaN' #initialise index k and previous eigval

for ee, egval in enumerate(eigvals_Hf):
    if k == len(eigvals_Hf_unique):
        break
    projector[k] += np.kron(P.T[ee], P.T[ee]).reshape(dims_matrices, dims_matrices)
    k += egval != egval_previous ## If true, k=1, and if false, k=0
    egval_previous = egval #update previous eigenvalue
## pick gs projector
projectorGS_Hf = projector[0]

#### CHECK OVERLAP WITH TRUE GROUND STATE OF TARGET HAMILTONIAN
## In these cases, we need to trace out over the ancillas
if final_hamiltonian == 'kSAT Dickson' or final_hamiltonian == 'kSAT Dickson PAPER' or final_hamiltonian == 'simple Ising Dickson': ## We have to trace over the ancilla
    rho_qt = np2qt(rho, N_qubits)
    reduced_rho = rho_qt.ptrace(np.arange(N_qubits - n_a))
    projectorGS_Hf =  np.asarray( np2qt(projectorGS_Hf, N_qubits).ptrace(np.arange(N_qubits - n_a)) )
    print('ground state of operational qubits: \n', np.real(np.round(projectorGS_Hf, 6)))
    final_overlap = np.real(np.trace(np.dot(projectorGS_Hf, reduced_rho)))
else:
    final_overlap = np.real(np.trace(np.dot(projectorGS_Hf, rho)))
    print('ground state of operational qubits: \n', np.real(np.round(projectorGS_Hf, 6)))
print('FINAL OVERLAP', final_overlap)
print('final mean energy', np.real(np.trace(np.dot(H_f, rho))))
print('eigvals_Hf', eigvals_Hf)
print('last s', s)


Hss = [spectral_norm(H_s(s)) for s in np.linspace(0, 1, 100)]
max_Hss = max(Hss)

print('adiabatic theorem bound on T', spectral_norm(H_f - H_i) * max_Hss /min(gaps)**2 ) 

print('adiabatic theorem bound on T for gaps2', spectral_norm(H_f - H_i) * max_Hss /min(gaps2)**2 ) 


eses = np.linspace(0, 1, n_points)
figsz = (12, 16)
xaxis = eses ; xlabel = 's'


################################# PLOTTING
### ENERGY LANDSCAPE DURING THE ANNEAL
fig, ax = plt.subplots(figsize = figsz)
ylabel = 'energy'
for i in range(dims_matrices):
    
    yaxis = energies[:,i]
    fig, ax = simple_plot(fig, ax, xaxis, yaxis, xlabel, ylabel, markr = 'o', alpha = 0.4, reduce = True)
# yaxis = np.real(mean_E)
# fig, ax = simple_plot(fig, ax, xaxis, yaxis, xlabel, ylabel, markr = 'kx', reduce = True)
plt.show()


### GAP
gp = gaps[0]
kkk = 0
while gp != min(gaps):
    kkk += 1
    gp = gaps[kkk]
print('minimum gap =', min(gaps), ' located at s =', kkk/n_points)    


gp = gaps2[0]
kkk = 0
while gp != min(gaps2):
    kkk += 1
    gp = gaps2[kkk]
print('minimum gap2 =', min(gaps2), ' located at s =', kkk/n_points)


## Noneq free energy <-- DOESN'T SAY MUCH NEW, BASICALLY FOLLOWS MEAN ENERGY
fig, ax = plt.subplots(figsize = figsz)
ylabel = '$\\mathcal{F}_{neq}$'
yaxis = np.real(F_neq)
fig, ax = simple_plot(fig, ax, xaxis, yaxis, xlabel, ylabel, reduce = True)
plt.show()       


### ENTANGLEMENT ENTROPY tracing out ceil(half) the spins
fig, ax = plt.subplots(figsize = figsz)
ylabel = 'Entanglement entropy'
yaxis = ent_entropy
fig, ax = simple_plot(fig, ax, xaxis, yaxis, xlabel, ylabel, reduce = True)
plt.show()     


### HAMMING DISTANCE (as an indicator oh a phase transition)
hamm_dist = np.real(hamm_dist)
hamm_dist_1st_exc = np.real(hamm_dist_1st_exc)

fig, ax = plt.subplots(figsize = figsz)
ylabel = 'Hamming distance'
fig, ax = simple_plot(fig, ax, xaxis, hamm_dist, xlabel, ylabel, reduce = True)
fig, ax = simple_plot(fig, ax, xaxis, hamm_dist_1st_exc, xlabel, ylabel, reduce = True)
plt.show()     



### OVERLAP OF CURRENT STATE WITH THE FIRST l(<-that's lower case L) LEVELS OF THE HAMILTONIAN ACCORDING TO WHICH IT HAS EVOLVED
probabilities = np.real(probabilities)

plt.figure(figsize=(12, 16))
plt.plot(np.linspace(0, 1, n_points)[0:-1:10], probabilities[:,0][0:-1:10], 'o-', ms = 3, label = str(0))
plt.plot(np.linspace(0, 1, n_points)[0:-1:10], probabilities[:,1][0:-1:10], 'x-', ms = 3, label = str(1))
l = 6
if len(probabilities.T) < l:
    l = len(probabilities.T)
for r in range(l - 2):
    plt.plot(np.linspace(0, 1, n_points)[0:-1:10], probabilities[:,r+2][0:-1:10], alpha = 0.3, label = str(r+2))
plt.xlabel('s')
plt.ylabel('')
plt.legend(title = 'prob of level')
plt.show()








