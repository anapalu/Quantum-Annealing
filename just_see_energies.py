import matplotlib.pyplot as plt 
import numpy as np

from scipy.linalg import eigh#, eig
# from qutip import Qobj, ptrace
from timeit import default_timer as timer

import sys
sys.path.append('/home/ana/Documents/PhD')
from basics_pkg import Sigma, define_Hs, get_mingap_proper, simple_plot, spectral_norm, l2_norm




precision = 12

t0 = timer()
####### Build total Hamiltonian
N_operational_qubits = 5
n_a = 0
N_qubits = N_operational_qubits + n_a
dims_matrices = 2**N_qubits
final_hamiltonian = 'Ising' # 'Grover' # 'kSAT' # 'molecular' # 'spin network' # 'kSAT Dickson' # 'simple Ising Dickson' # 
initial_hamiltonian = 'Sigmas'#'Ising X'#'Sigmas' #'all spins down'# PLUS ancilla' #'transverse field' #'all spins up' # 'staggered' # 'disrespect bit structure' # 'bit structure' # 
                        # 'transverse field PLUS ancilla' # 'entangled' # 'transverse field Dickson' # 'transverse field + Sigmas' #
annealing_schedule = 'linear' # 'smooth two-step 2' # 'linear' # 'optimised Grover' # 'force Landau' # 'smooth two-step' # 'smooth two-step 2'
catalyst = 'None'#'parabola' # 'smooth two-step'#'parabola' # 'parabola' # 'None' # right tilted 'None'#
catalyst_interaction = 'None'#'Sigmas'#'spin network X'#'Sigmas'#'Sigmas Dickson'#'Sigmas' # 'Sigmas' # 'bit flip noise' # 'spin network X'

print('H_f:', final_hamiltonian, ', H_i:', initial_hamiltonian, ', annealing schedule:', annealing_schedule, ', catalyst:', catalyst_interaction, catalyst)
rot = False
mxdf = 19#14#13 # 7
###
h_mean = 1
#####################
rsnX = 31988#977805#15#9936#765#16234#765     ### rising = 15 + W=0 + h=1 is very hard (for N=5)
rising = 19885432#31988#54231#265836#991723#np.random.randint(1, 1000000, 100)[60]#15#2#936#15 ## seed_J=674 + h=1 + W=0 is suuuuper easy
##############

### STORE IMG
dict_namefig = {'transverse field': 'transv', 'all spins up': 'allup', 'all spins down': 'alldown', 'bit flip noisy': 'bitflipnoisy',
             'phase flip noisy': 'phaseflipnoisy', 'spin network X': 'randtransv', 'Ising':'ising', 'Ising X':'isingX', 'Sigmas Z':'sigmasZ', 
             'Sigmas':'sigmasX', 'kSAT':'3SAT', 'Ising X inverted': 'isingXINVERTEDSIGN'}
general_parent_folder = '/home/ana/Pictures/two-step anneal/' #'/home/ana/Pictures/perturbative crossings/systematic/'
current_parent_folder =  'one step comparison/'#'one ancilla per clause/' #'one ancilla per clause/' #'one step comparison/' 
parent_folder_satunsat = ''#'unsatisfied clause plus anc/'# 'satisfied clause plus anc/' # ''#
if final_hamiltonian == 'kSAT':
    namefig = '3SAT{}_N{}_{}'.format(mxdf, N_operational_qubits, dict_namefig[initial_hamiltonian]) #'3SAT{}_N{}+{}Dickson_{}'.format(mxdf, N_operational_qubits, n_a, dict_namefig[initial_hamiltonian])

if final_hamiltonian == 'Ising':
    namefig = 'ising_J{}W0h{}_N{}_{}'.format(rising, h_mean, N_operational_qubits, dict_namefig[initial_hamiltonian])

if catalyst == 'parabola' and catalyst_interaction == 'Sigmas':
    namefig = namefig + '_catalystX'

if catalyst == 'parabola' and catalyst_interaction == 'spin network X':
    namefig = namefig + '_CATspinnetX{}'.format(rsnX)

if catalyst == 'smooth two-step' and catalyst_interaction == 'spin network X' and annealing_schedule == 'smooth two-step':
    namefig = namefig + '_CATspinnetX{}smoothonehalf'.format(rsnX)

if catalyst == 'smooth two-step' and catalyst_interaction == 'spin network X' and annealing_schedule == 'smooth two-step 2':
    namefig = namefig + '_CATspinnetX{}smoothonethird'.format(rsnX)
##############
H_s, H_i, H_f, H_c = define_Hs(N_qubits, final_hamiltonian, initial_hamiltonian, annealing_schedule, catalyst, coord_catalyst = 1, 
    catalyst_interaction = catalyst_interaction, rotate=rot, h_mean = h_mean, 
    W = 0, randomspinnetX_J = rsnX, risingJ = rising, mxdf = mxdf, number_of_ancillas = n_a, return_HiHf = True)
######################################################################## 
print('eigvals H_f', eigh(H_f)[0])

print('magnitudes of Hi and Hf', l2_norm(H_i), np.real(l2_norm(H_f)))
print('commutator', l2_norm( np.dot(H_f, H_i) - np.dot(H_i, H_f) ))

T_anneal = 100#100 ## usual values to try out: T=100, dt = 0.01
dt = 0.01 

# #### WE WANT TO SEE THE LEVELS IN WHICH TRACING OUT THE ANCILLAS WE OBTAIN THE DESIRED GS
# eigvals, P = eigh(H_f)
# Sigmasz = [Sigma(0, i, N_qubits) for i in range(N_qubits) ]
# states_final = np.empty((dims_matrices, N_qubits)) ## expected value of Sz operator por each of the spins
# for i in range(dims_matrices):
#     for j in range(N_qubits):
#         states_final[i, j] = np.dot(P[:, i], np.dot(Sigmasz[j], P[:, i]))
# print(states_final[0])

# gsHi = eigh(H_i)[1][:, 0]
# print([np.dot(gsHi, np.dot(Sigmasz[j], gsHi)) for j in range(N_qubits)])


### MUST BE SMALL ENOUGH TO SIMULATE THE CONTINUOUS CHANGE OF THE HAMILTONIAN
            ## so far I have only managed to stay in the ground state for dt = 1e-18 and T=1000
ds = dt / T_anneal
n_points = int(T_anneal / dt)
print('n_points', n_points)



energies = np.zeros((n_points, dims_matrices))

mingaps_0 = np.zeros((n_points, dims_matrices-1))
mingaps_previous = np.zeros((n_points, dims_matrices-1))

n_pointsHD = n_points//100
# hamm_dist = np.zeros((n_pointsHD, dims_matrices))

prev_H = H_s(0)
instantaneous_Tad = np.zeros(n_points)


indexHD = 0
for i in range(n_points):
    if i % int(n_points/10) == 0:
        print('real time', i*dt, end = '\r')

    
    s = ds * i
    H = H_s(s)
    

    eigvals, P = eigh(H) 
    eigvals = np.round(eigvals, precision)
    eigvals_unique = np.unique(eigvals)

    ##############
    energies[i] += eigvals
    mingaps_0[i] += eigvals[1:] - eigvals[0]
    mingaps_previous[i] += eigvals[1:] - eigvals[:-1]


    instantaneous_Tad[i] = spectral_norm(H-prev_H)/ds * spectral_norm(H) /mingaps_0[i, 0]**2 
    prev_H = H

    # DISREGARDING DEGENERACIES, WE ARE GOING TO CONSIDER THEY ARE REMOVED DURING THE ANNEAL
    # if i % int(n_points/100) == 0:
    #     for kkk in range(dims_matrices):
    #         d = 0
    #         for iii in range(N_qubits):
    #             d += 1 - np.dot(np.conj(P[:, kkk].T), np.dot(Sigma(0, iii, N_qubits), P[:, kkk]))
    #         hamm_dist[indexHD, kkk] = np.real(d)
    #     indexHD += 1
    
    # projector = np.zeros(( len(eigvals_unique) , dims_matrices, dims_matrices), dtype = 'complex128')
    
    # k = 0
    # egval_previous = 'NaN' #initialise index k and previous eigval

    # for ee, egval in enumerate(eigvals):
    #     if k == len(eigvals_unique):
    #         break
    #     projector[k] += np.kron(P.T[ee], P.T[ee]).reshape(dims_matrices, dims_matrices)
    #     k += egval != egval_previous ## If true, k+=1, and if false, k+=0
    #     egval_previous = egval #update previous eigenvalue
    
    # hamm_dist += [np.real(Hamming_distance_degenerate(projector[0]))]
    # if len(projector) > 1:
    #     hamm_dist_1st_exc += [np.real(Hamming_distance_degenerate(projector[1]))]
    #     if len(projector) > 2:
    #         hamm_dist_2nd_exc += [np.real(Hamming_distance_degenerate(projector[2]))]
    #     else:
    #         hamm_dist_2nd_exc += ['NaN']
    # else:
    #     hamm_dist_1st_exc += ['NaN']

mingaps10, mingaps_loc10 = get_mingap_proper(energies[:, 0], energies[:, 1]) 
print('BETWEEN gs AND 1st exc') 
print('mingap width=', mingaps10, '\nat s=', mingaps_loc10)

mingaps21, mingaps_loc21 = get_mingap_proper(energies[:, 1], energies[:, 2]) 
print('BETWEEN 1st exc AND 2nd exc') 
print('mingap width=', mingaps21, '\nat s=', mingaps_loc21)

mingaps32, mingaps_loc32 = get_mingap_proper(energies[:, 2], energies[:, 3]) 
print('BETWEEN 2nd exc AND 3rd exc') 
print('mingap width=', mingaps32, '\nat s=', mingaps_loc32)

t1 = timer()
print('The simulation with {} took {} s'.format(N_operational_qubits + n_a, t1 - t0))

### GAP
gaps = mingaps_0[:, 0]
gaps2 = mingaps_0[:, 1]



##### Get adiabatic bound on T_anneal
eigvals_Hf, eigvects_Hf = eigh(H_s(1))
min_eigval_Hf = eigvals_Hf[0]
Hss = [spectral_norm(H_s(s)) for s in np.linspace(0, 1, 100)]
max_Hss = max(Hss)
T_adiabatic_theorem = spectral_norm(H_f - H_i) * max_Hss /min(gaps)**2
print('adiabatic theorem bound on T', T_adiabatic_theorem ) 
print('adiabatic theorem bound on T for gaps2', spectral_norm(H_f - H_i) * max_Hss /min(gaps2)**2 ) 



eses = np.linspace(0, 1, n_points)
figsz = (12, 16)
xaxis = eses ; xlabel = 's'

################################# PLOTTING
### ENERGY LANDSCAPE DURING THE ANNEAL

fig, ax = plt.subplots( figsize = figsz)
ylabel = 'energy'
for i in range(dims_matrices):
    yaxis = energies[:,i]
    fig, ax = simple_plot(fig, ax, xaxis, yaxis, xlabel, ylabel, markr = '-', alpha = 0.4, reduce = False)
ax.set_title('$\\Delta = {}$'.format(min(gaps)))
fig.savefig(general_parent_folder + current_parent_folder + parent_folder_satunsat + namefig + '.pdf')


fig, ax = plt.subplots( figsize = figsz)
ylabel = '$T_{adiabatic}$'

yaxis = instantaneous_Tad
fig, ax = simple_plot(fig, ax, xaxis[1:], yaxis[1:], xlabel, ylabel, markr = '-', title = '$T={}$'.format(np.round(T_adiabatic_theorem)))
fig, ax = simple_plot(fig, ax, xaxis, T_adiabatic_theorem * np.ones(n_points), xlabel, ylabel, markr = '--')
fig.savefig(general_parent_folder + current_parent_folder + parent_folder_satunsat + namefig + 'Tad_instantaneous.pdf')
plt.show()


# fig, ax = plt.subplots( figsize = figsz)
# ylabel = 'energy'
# xaxis = np.linspace(0, 1, n_points)
# for i in range(10):
#     yaxis = energies[:,i]
#     fig, ax = simple_plot(fig, ax, xaxis, yaxis, xlabel, ylabel, markr = '-', alpha = 0.4, reduce = False)
# fig.savefig(general_parent_folder + current_parent_folder + parent_folder_satunsat + '3SAT14_N5+{}Dickson_transv_first8levels.pdf'.format(n_a))
### DOESN'T MAKE MUCH SENSE TO KEEP THIS ONE IF WE ARE ALREADY STORING THE PDF

# ylabel = 'hamming distance'
# xaxis = np.linspace(0, 1, n_pointsHD)
# for i in range(10):
#     yaxis = hamm_dist[:,i]
#     fig, ax[1] = simple_plot(fig, ax[1], xaxis, yaxis, xlabel, ylabel, markr = '-', alpha = 0.4, reduce = False)
# plt.show()
# fig, ax = plt.subplots(figsize = figsz)
# ylabel = 'hamming distance'
# fig, ax = simple_plot(fig, ax, xaxis, hamm_dist, xlabel, ylabel, markr = '-', alpha = 0.4, reduce = False)
# fig, ax = simple_plot(fig, ax, xaxis, hamm_dist_1st_exc, xlabel, ylabel, markr = '-', alpha = 0.4, reduce = False)
# fig, ax = simple_plot(fig, ax, xaxis, hamm_dist_2nd_exc, xlabel, ylabel, markr = '-', alpha = 0.4, reduce = False)
# ax.legend(labels = ['gs', '1exc'])
# plt.show()


# fig, ax = plt.subplots(figsize = figsz)
# ylabel = 'gap'
# l = 8
# for i in range(l): ## plot the fist l distances between levels
#     yaxis = mingaps_previous[:, i]
#     fig, ax = simple_plot(fig, ax, xaxis, yaxis, xlabel, ylabel, markr = '-', alpha = 0.4, reduce = False)
# ax.legend(labels = [str(j+1) + '-' + str(j) for j in np.arange(l)])
# plt.show()


