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



energies = np.zeros((n_points, dims_matrices))

gaps = []
gaps2 = []


for i in range(n_points):
    if i % int(n_points/10) == 0:
        print('real time', i*dt, end = '\r')

    
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

### GAP
gp = gaps[0]
kkk = 0
while gp != min(gaps):
    kkk += 1
    gp = gaps[kkk]
print('minimum gap =', min(gaps), ' located at s =', kkk/n_points)    
##
gp = gaps2[0]
kkk = 0
while gp != min(gaps2):
    kkk += 1
    gp = gaps2[kkk]
print('minimum gap2 =', min(gaps2), ' located at s =', kkk/n_points)
  

##### Get adiabatic bound on T_anneal
eigvals_Hf, eigvects_Hf = eigh(H_s(1))
min_eigval_Hf = eigvals_Hf[0]
print('eigvals_Hf', eigvals_Hf)
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
plt.show()

fig, ax = plt.subplots(figsize = figsz)
ylabel = 'gap'
fig, ax = simple_plot(fig, ax, xaxis, gaps, xlabel, ylabel, markr = 'o', alpha = 0.4, reduce = True)
fig, ax = simple_plot(fig, ax, xaxis, gaps2, xlabel, ylabel, markr = 'o', alpha = 0.4, reduce = True)
ax.legend(labels = ['1-0', '2-0'])
plt.show()







