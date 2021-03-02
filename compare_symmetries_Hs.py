
import matplotlib.pyplot as plt 
import numpy as np
from qutip.operators import commutator
from qutip.qobj import dims

from scipy.linalg import eigh, eig
from qutip import Qobj, ptrace
from timeit import default_timer as timer


import sys
sys.path.append('/home/ana/Documents/PhD')
from basics_pkg import Sigma, define_Hs, get_mingap_proper, simple_plot, spectral_norm, l2_norm, vN_entropy, np2qt

# from basics_manage_data import retrieve_instances

import networkx as nx
# import dynetx as dn
# from dynetx.algorithms import time_respecting_paths

def factorial(n):
    nnn = 1
    for i in np.arange(1, n+1):
        nnn *= i
    return nnn

def combinatorial_coeff(N, m): ### returns an integer
    return factorial(N) // (factorial(N-m)*factorial(m))




precision = 12

####### Build total Hamiltonian
N_operational_qubits = 5
n_a = 0
N_qubits = N_operational_qubits + n_a
dims_matrices = 2**N_qubits
final_hamiltonian = 'Ising' # 'Grover' # 'kSAT' # 'molecular' # 'spin network' # 'kSAT Dickson' # 'simple Ising Dickson' # 
initial_hamiltonian = 'Ising X'# complementary' #'all spins down'# PLUS ancilla' #'transverse field' #'all spins up' # 'staggered' # 'disrespect bit structure' # 'bit structure' # 
                        # 'transverse field PLUS ancilla' # 'entangled' # 'transverse field Dickson' # 'transverse field + Sigmas' #
annealing_schedule = 'linear' # 'smooth two-step 2' # 'linear' # 'optimised Grover' # 'force Landau' # 'smooth two-step' # 'smooth two-step 2'
catalyst = 'None'#'parabola' # 'smooth two-step'#'parabola' # 'parabola' # 'None' # right tilted 'None'#
catalyst_interaction = 'None'#'Sigmas'#'spin network X'#'Sigmas'#'Sigmas Dickson'#'Sigmas' # 'Sigmas' # 'bit flip noise' # 'spin network X'

print('H_f:', final_hamiltonian, ', H_i:', initial_hamiltonian, ', annealing schedule:', annealing_schedule, ', catalyst:', catalyst_interaction, catalyst)
rot = False
mxdf = 19#14#13 # 7
###
h_mean = 1#1
#####################
rsnX = 1998#19885432# 5195#88#5424#977805#15#9936#765#16234#765     ### rising = 15 + W=0 + h=1 is very hard (for N=5)
rising = 19885432#15#991723#np.random.randint(1, 1000000, 100)[60]#15#2#936#15 ## seed_J=674 + h=1 + W=0 is suuuuper easy
##############



### STORE IMG
dict_namefig = {'transverse field': 'transv', 'all spins up': 'allup', 'all spins down': 'alldown', 'bit flip noisy': 'bitflipnoisy',
             'phase flip noisy': 'phaseflipnoisy', 'spin network X': 'randtransv', 'Ising':'ising', 'Ising X':'isingX', 'Sigmas Z':'sigmasZ', 
             'Sigmas':'sigmasX', 'kSAT':'3SAT', 'Ising X inverted': 'isingXINVERTEDSIGN', 'Ising X complementary': 'isingXCOMPLEMENTARY'}
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


#######################
from numba import njit
@njit
def fast_dot(A, B):
    return np.dot(A, B)
#######################

Sigma_dict = {} ### Nested dictionary, first set coord (0--> 'z', 1 --> 'x', 2 --> 'y') and inside a list/array containing all the corresponding
            ### matrices, ordered by qubit number
for sp in range(N_qubits):
    Sigma_dict[sp] = np.empty((3, dims_matrices, dims_matrices), 'complex128')
    for coord in range(3):
        Sigma_dict[sp][coord] = Sigma(coord, sp, N_qubits)


too_much_data = False
T_anneal = 1000
dt = 0.1   ### MUST BE SMALL ENOUGH TO SIMULATE THE CONTINUOUS CHANGE OF THE HAMILTONIAN
            ## so far I have only managed to stay in the ground state for dt = 1e-18 and T=1000
ds = dt / T_anneal
n_points = int(T_anneal / dt)

def commutator(A, B):
    return fast_dot(A, B) - fast_dot(B, A)


# tracecomm_SzpSx = np.empty((n_points + 1, N_qubits**2))
# l2normcomm_SzpSx = np.empty((n_points + 1, N_qubits**2))
# spectralnormcomm_SzpSx = np.empty((n_points + 1, N_qubits**2))

tracecomm_Hamm = np.empty(n_points + 1)
l2normcomm_Hamm = np.empty(n_points + 1)
spectralnormcomm_Hamm = np.empty(n_points + 1)

O_Hamming = np.zeros((dims_matrices, dims_matrices), dtype = 'complex128')
for i in range(N_qubits):
    O_Hamming += np.eye(dims_matrices, dtype = 'complex128') - Sigma_dict[i][0]

Sx = np.array([[0.,1.], [1.,0.]], dtype = 'complex128')
Sy = np.array([[0.,-1j], [1j,0.]], dtype = 'complex128')
Sz = np.array([[1.,0.], [0.,-1.]], dtype = 'complex128')
axis = np.array([0, 1, 0])
angles = np.array([0, np.pi/2, 0, 0, 0])
# angles = np.array([np.pi/2, 0, np.pi/2, np.pi/2, np.pi/2])
for i, th in enumerate( angles ):
    if i == 0:
        R = np.cos(0.5*th) * np.eye(2) - 1j * np.sin(0.5*th) * (axis[0] * Sx + axis[1] * Sy + axis[2] * Sz)
    else:
        rot = np.cos(0.5*th) * np.eye(2) - 1j * np.sin(0.5*th) * (axis[0] * Sx + axis[1] * Sy + axis[2] * Sz)
        R = np.kron(R, rot)



sp = 1
Sy_local = (Sigma_dict[sp][1])# 1  + Sigma_dict[sp][1]

O = np.zeros((dims_matrices, dims_matrices), 'complex128')
for i in range(N_qubits):
    for j in range(i):
        O += Sigma_dict[i][2]/N_qubits #(np.dot(Sigma_dict[i][1], Sigma_dict[j][0])  )/(N_qubits*(N_qubits-1)/2)

O = O_Hamming #Sy_local #################
energies = np.empty((n_points+1, dims_matrices))

O_expectedz = np.zeros((dims_matrices, dims_matrices), dtype = 'complex128')
for i in range(N_qubits):
    O_expectedz += Sigma_dict[i][0]

O_expectedx = np.zeros((dims_matrices, dims_matrices), dtype = 'complex128')
for i in range(N_qubits):
    O_expectedx += Sigma_dict[i][1]

expectedZpX = np.empty((n_points+1, dims_matrices))
expectedX = np.empty((n_points+1, dims_matrices))

expectedZ_gs = np.empty((n_points+1, N_qubits))
expectedZ_exc1 = np.empty((n_points+1, N_qubits))
expectedZ_exc2 = np.empty((n_points+1, N_qubits))

expectedZ_spins_rho = np.empty((n_points + 1, N_qubits))
for i in range(n_points + 1):
    s = ds * i
    H = H_s(s)
    ind_spsp = 0

    eigvals, P = eigh(H)
    energies[i] = eigvals

    if i == 0:
        rho = np.kron(P[:, 0], P[:, 0]).reshape(dims_matrices, dims_matrices)

    D = np.diag(np.exp(1j*eigvals * dt))
    evol_op = np.dot(P, np.dot(D, P.T))
    rho = np.dot(np.conj(evol_op.T),np.dot(rho, evol_op))
    
    # gs = P[:, 0]
    # exc1 = P[:, 1]
    # exc2 = P[:, 2]

    expectedZ_spins_rho[i] = [np.trace(np.dot(Sigma_dict[sp][0], rho)) for sp in range(N_qubits)]

    expectedZpX[i] = [np.dot(P[:, lev], np.dot(O_expectedz + O_expectedx, P[:, lev])) for lev in range(dims_matrices)]
    
    expectedX[i] = [np.dot(P[:, lev], np.dot(O_expectedx, P[:, lev])) for lev in range(dims_matrices)]

    expectedZ_gs[i] = [np.dot(P[:, 0], np.dot(Sigma_dict[sp][0], P[:, 0])) for sp in range(N_qubits)]
    expectedZ_exc1[i] = [np.dot(P[:, 1], np.dot(Sigma_dict[sp][0], P[:, 1])) for sp in range(N_qubits)]
    expectedZ_exc2[i] = [np.dot(P[:, 2], np.dot(Sigma_dict[sp][0], P[:, 2])) for sp in range(N_qubits)]
    # tracecomm_Hamm[i] = np.dot(gs, np.dot(Sy_local, gs))
    # l2normcomm_Hamm[i] = np.dot(exc1, np.dot(Sy_local, exc1))
    # spectralnormcomm_Hamm[i] = np.dot(exc2, np.dot(Sy_local, exc2))
    c = commutator(O, H)#commutator(O_Hamming, H)
    #projector onto gs, 1exc
    # proj01 = np.kron(P[:, 0], P[:, 0]).reshape(dims_matrices, dims_matrices) + np.kron(P[:, 1], P[:, 1]).reshape(dims_matrices, dims_matrices)
    # c = np.dot(proj01, np.dot(c, proj01))
    tracecomm_Hamm[i] = np.trace(c)
    l2normcomm_Hamm[i] = l2_norm(c)
    spectralnormcomm_Hamm[i] = spectral_norm(c)

    # for sp in range(N_qubits):
    #     # for sp2 in range(N_qubits):
    #     O = fast_dot(Sigma_dict[sp][0], Sigma_dict[sp][1])# + Sigma_dict[sp2][1]
    #     c = commutator(O, H)
    #     tracecomm_SzpSx[i, ind_spsp] = np.trace(c)
    #     l2normcomm_SzpSx[i, ind_spsp] = l2_norm(c)
    #     spectralnormcomm_SzpSx[i, ind_spsp] = spectral_norm(c)
    #     ind_spsp += 1

print('last s', s)
figsz = (10, 6)
eses = ds * np.arange(n_points + 1)
xaxis = eses ; xlabel = 's'


####
np.random.seed(rising)
Jising = (2*np.random.rand(N_qubits, N_qubits) - 1)
print('\nJ Ising', Jising, '\n')
fig, ax = plt.subplots()
plt.imshow(Jising)

np.random.seed(rsnX)
JXIsing = (2*np.random.rand(N_qubits, N_qubits) - 1)
print('\nJ Ising X', JXIsing, '\n')
fig, ax = plt.subplots()
plt.imshow(JXIsing)
plt.show()
################################# PLOTTING

# ylabel = 'Tr$([O, H])$' ### TELLS US NOTHING

fig, ax = plt.subplots( 1, 4, figsize = figsz)
ylabel = 'norm of commutator'
ax_tot = ax
ax = ax_tot[0]
fig, ax = simple_plot(fig, ax, xaxis, tracecomm_Hamm, xlabel, ylabel, markr = 'o', alpha = 1, reduce = False)
fig, ax = simple_plot(fig, ax, xaxis, l2normcomm_Hamm, xlabel, ylabel, markr = 'o', alpha = 1, reduce = False)
fig, ax = simple_plot(fig, ax, xaxis, spectralnormcomm_Hamm, xlabel, ylabel, markr = 'o', alpha = 1, reduce = False)
ax.legend(labels = ['trace', 'l2', 'spectral'])
# plt.show()

# fig, ax = plt.subplots(figsize = (16, 12))
# phs = np.linspace(0, np.pi/2, (n_points + 1))
ax = ax_tot[1]
for i in range(dims_matrices):
    ax.plot(eses, energies[:, i])
ax.set_xlabel('$s$')
ax.set_ylabel('E')

ax = ax_tot[2]
for i in range(dims_matrices):
    ax.plot(eses, expectedZpX[:, i])
ax.set_xlabel('$s$')
ax.set_title('$\\Sigma_i \\sigma^i_z + \\Sigma_i \\sigma^i_x$')


ax = ax_tot[3]
for i in range(dims_matrices):
    ax.plot(eses, expectedX[:, i])
ax.set_xlabel('$s$')
ax.set_title('$\\Sigma_i \\sigma^i_x$')



fig, ax = plt.subplots( 1, 4, figsize = figsz)
ylabel = ''
ax_tot = ax
ax = ax_tot[0]
fig, ax = simple_plot(fig, ax, xaxis, expectedZ_gs, xlabel, ylabel, markr = 'o', alpha = 1, reduce = False, title = 'gs')
ax = ax_tot[1]
fig, ax = simple_plot(fig, ax, xaxis, expectedZ_exc1, xlabel, ylabel, markr = 'o', alpha = 1, reduce = False, title = '1exc')
ax = ax_tot[2]
fig, ax = simple_plot(fig, ax, xaxis, expectedZ_exc2, xlabel, ylabel, markr = 'o', alpha = 1, reduce = False, title = '2exc')

ax = ax_tot[3]
fig, ax = simple_plot(fig, ax, xaxis, expectedZ_spins_rho, xlabel, ylabel, markr = 'o', alpha = 0.5, reduce = True, title = 'rho')


# plt.show()

for i in range(N_qubits):
    for j in range(i):
        Jising[j, i] = Jising[i, j]
        JXIsing[j, i] = JXIsing[i, j]

print('\nsymmetric JXIsing\n', JXIsing)



def suscept(sigmas_expectedval, eses, jotas):
    eses_mid = 0.5*(eses[1:] + eses[:-1])
    sigmasder = (sigmas_expectedval[1:, :] - sigmas_expectedval[:-1, :])/ds
    expectedZ_spins_rho_mid = 0.5 * (sigmas_expectedval[1:, :] + sigmas_expectedval[:-1, :])
    n_points, N_qubits = np.shape(expectedZ_spins_rho_mid)

    Xz = np.empty((n_points, N_qubits))
    for i in range(N_qubits):
        for j in range(N_qubits):
            sums = 0
            sumsder = 0
            if j!= i:
                sums += jotas[i, j] * expectedZ_spins_rho_mid[:, j]
                sumsder += jotas[i, j] * sigmasder[:, j]
        sumsder = eses_mid * sumsder
        Xz[:, i] = sigmasder[:, i] /(h_mean + sums + sumsder)
    return Xz, eses_mid

Xz, eses_mid = suscept(expectedZ_spins_rho, eses, Jising)
Xz_gs, eses_mid = suscept(expectedZ_gs, eses, Jising)
Xz = Xz**2
Xz_gs = Xz_gs**2

fig, ax = plt.subplots( figsize = figsz)
ylabel = '$\\chi^z$'
fig, ax = simple_plot(fig, ax, eses_mid, Xz, xlabel, ylabel, markr = 'o', alpha = 1, reduce = False)
ax.set_yscale('log')

fig, ax = plt.subplots( figsize = figsz)
ylabel = '$\\frac{1}{N}\\Sigma_i\\chi^z$'
fig, ax = simple_plot(fig, ax, eses_mid, np.mean(Xz, axis = 1), xlabel, ylabel, markr = 'o', alpha = 1, reduce = False)
ax.set_yscale('log')

fig, ax = plt.subplots( figsize = figsz)
ylabel = '$\\chi^z$'
fig, ax = simple_plot(fig, ax, eses_mid, Xz_gs, xlabel, ylabel, markr = 'o', alpha = 1, reduce = False, title='gs')
ax.set_yscale('log')

fig, ax = plt.subplots( figsize = figsz)
ylabel = '$\\frac{1}{N}\\Sigma_i\\chi^z$'
fig, ax = simple_plot(fig, ax, eses_mid, np.mean(Xz_gs, axis = 1), xlabel, ylabel, markr = 'o', alpha = 1, reduce = False, title='gs')
ax.set_yscale('log')
plt.show()

exit()
fig, ax = plt.subplots( figsize = figsz)
ylabel = 'Tr$([O, H])$'
# for i in range(N_qubits**2):
for i in range(N_qubits):
    yaxis = tracecomm_SzpSx[:, i]
    fig, ax = simple_plot(fig, ax, xaxis, yaxis , xlabel, ylabel, markr = 'o', alpha = 1, reduce = False)
ax.legend(labels = [str(i) for i in np.arange(N_qubits)])


fig, ax = plt.subplots( figsize = figsz)
ylabel = '$L_2([O, H])$'

# for i in range(N_qubits**2):
for i in range(N_qubits):
    yaxis = l2normcomm_SzpSx[:, i]
    fig, ax = simple_plot(fig, ax, xaxis, yaxis , xlabel, ylabel, markr = 'o', alpha = 1, reduce = False)
ax.legend(labels = [str(i) for i in np.arange(N_qubits)])




fig, ax = plt.subplots( figsize = figsz)
ylabel = 'spectral norm$([O, H])$'

# for i in range(N_qubits**2):
for i in range(N_qubits):
    yaxis = spectralnormcomm_SzpSx[:, i]
    fig, ax = simple_plot(fig, ax, xaxis, yaxis , xlabel, ylabel, markr = 'o', alpha = 1, reduce = False)
ax.legend(labels = [str(i) for i in np.arange(N_qubits)])
plt.show()






exit()

mean_shpth = np.zeros(n_points)
mean_deg = np.zeros(n_points)
mean_clust = np.zeros(n_points)
std_shpth = np.zeros(n_points)
std_deg = np.zeros(n_points)
std_clust = np.zeros(n_points)

ent_1to1_gs = np.zeros((n_points, N_qubits))
ent_1to1_1exc = np.zeros((n_points, N_qubits))

combinatorial_coefficient = combinatorial_coeff(N_qubits, 2)
ent_1to2_gs = np.zeros((n_points, combinatorial_coefficient ))
ent_1to2_1exc = np.zeros((n_points, combinatorial_coefficient ))

# ent_1to2 = np.zeros((n_points, N_qubits, N_qubits**2))

# for n in range(N_qubits):
#     for i in range (N_qubits):
#         spins2bkept_1to1 = [j for j in range(N_qubits) if j != i and j != n]

#         rhobipartite_gs = rho_gs_qt.ptrace(spins2bkept_1to1)
#         eigvalsrh_gs, _= eig(rhobipartite_gs.ptrace(0))
#         ##
#         rhobipartite_1exc = rho_1exc_qt.ptrace(spins2bkept_1to1)
#         eigvalsrh_1exc, _= eig(rhobipartite_1exc.ptrace(0))

#         ent_1to1_gs[0, n, i] = vN_entropy(eigvalsrh_gs)
#         ent_1to1_1exc[0, n, i] = vN_entropy(eigvalsrh_1exc)

#         print(n, i, ent_1to1_gs[0, n, i], ent_1to1_1exc[0, n, i])


gaps = np.empty(n_points)


for nn in range(n_points):

    # if nn % int(n_points/1000) == 0:

    if nn % int(n_points/10) == 0:
        print('real time', nn*dt)
    s = ds * nn
    H = H_s(s)

    eigvals, P = eigh(H) 
    eigvals = np.round(eigvals, precision)
    eigvals_unique = np.unique(eigvals)
    gaps[nn] = eigvals_unique[1] - eigvals_unique[0]

    gs = P.T[0]
    rho_gs = np.kron(gs, gs).reshape((dims_matrices, dims_matrices))  #### SET INITIAL STATE
    rho_gs_qt = np2qt(rho_gs, N_qubits)


    firstexc_eigval = eigvals_unique[1]
    i = 1
    projector_1exc = np.zeros(( dims_matrices, dims_matrices), dtype = 'complex128')
    while i < dims_matrices and np.round(eigvals[i], precision) == np.round(firstexc_eigval, precision):

        projector_1exc += np.kron(P.T[i], P.T[i]).reshape(dims_matrices, dims_matrices)
        i += 1
    rho_1exc = projector_1exc / np.trace(projector_1exc)
    rho_1exc_qt = np2qt(rho_1exc, N_qubits)



    index_1to2 = 0
    for n in range(N_qubits):
        spins2bkept_1to1 = [j for j in range(N_qubits) if j != n]
        rhobipartite_gs = rho_gs_qt.ptrace(spins2bkept_1to1)
        eigvalsrh_gs, _= eig(rhobipartite_gs)
        ##
        rhobipartite_1exc = rho_1exc_qt.ptrace(spins2bkept_1to1)
        eigvalsrh_1exc, _= eig(rhobipartite_1exc)

        ent_1to1_gs[nn, n] = vN_entropy(eigvalsrh_gs)
        ent_1to1_1exc[nn, n] = vN_entropy(eigvalsrh_1exc)

        for j in range(n+1, N_qubits):
            spins2bkept_1to2 = [k for k in spins2bkept_1to1 if k != j]

            rhobipartite_gs = rho_gs_qt.ptrace(spins2bkept_1to2)
            eigvalsrh_gs, _= eig(rhobipartite_gs)
            ##
            rhobipartite_1exc = rho_1exc_qt.ptrace(spins2bkept_1to2)
            eigvalsrh_1exc, _= eig(rhobipartite_1exc)

            ent_1to2_gs[nn, index_1to2] = vN_entropy(eigvalsrh_gs)
            ent_1to2_1exc[nn, index_1to2] = vN_entropy(eigvalsrh_1exc)
            index_1to2 += 1

            

            

            







    # G = nx.Graph()
    # listedges = []
    # reshaped_H = H.reshape(dims_matrices * dims_matrices)
    # for i, h in enumerate(reshaped_H):
    #     row, column = i//dims_matrices, i%dims_matrices
    #     if row - column == 0:
    #         pass
    #     else:
    #         listedges += [(i//dims_matrices, i%dims_matrices, np.sqrt(np.real(np.conj(h)*h)))] #np.real(h)) ]h) ]
    # G.add_weighted_edges_from(listedges)

    # num_connected_components = len( list(nx.connected_components(G)) )

    # if num_connected_components == 1:

    #     sh_pth = np.zeros((dims_matrices, dims_matrices))
    #     for node, pathlens in nx.shortest_path_length(G, weight = 'weight'):
    #         sh_pth[node] += [pathlens[j] for j in range(dims_matrices)]

    #     all_shpth_inds = np.triu_indices_from(sh_pth)
    #     all_shpth = sh_pth[all_shpth_inds]
    
    #     mean_shpth[nn] = nx.average_shortest_path_length(G, weight='weight')
    #     std_shpth[nn] = np.sqrt( np.sum((all_shpth - mean_shpth[nn])**2)/len(all_shpth) )

    #     mean_clust[nn] = nx.average_clustering(G, weight = 'weight')
    #     # std_clust[nn]


    #     degs = []
    #     for node, dg in nx.degree(G, weight = 'weight'):
    #         degs += [dg]
    #     mean_deg[nn] = np.mean(degs)
    #     std_deg[nn] = np.std(degs)

    # else:
    #     print('We broke connectivity at s={} into {} connected components'.format(s, num_connected_components))
    #     exit()


# fig, ax = plt.subplots(3,2, figsize = (16, 16))
# if too_much_data == True:
#     ax[0, 0].plot(np.linspace(0, 1, n_points)[0:-1:10], mean_deg[0:-1:10])
#     ax[0, 1].plot(np.linspace(0, 1, n_points)[0:-1:10], std_deg[0:-1:10])
#     ax[1, 0].plot(np.linspace(0, 1, n_points)[0:-1:10], mean_shpth[0:-1:10])
#     ax[1, 1].plot(np.linspace(0, 1, n_points)[0:-1:10], std_shpth[0:-1:10])
#     ax[2, 0].plot(np.linspace(0, 1, n_points)[0:-1:10], mean_clust[0:-1:10])
# else:
#     ax[0, 0].plot(np.linspace(0, 1, n_points), mean_deg)
#     ax[0, 1].plot(np.linspace(0, 1, n_points), std_deg)
#     ax[1, 0].plot(np.linspace(0, 1, n_points), mean_shpth)
#     ax[1, 1].plot(np.linspace(0, 1, n_points), std_shpth)

#     ax[2, 0].plot(np.linspace(0, 1, n_points), mean_clust)
    
# ax[0, 0].set_ylabel('mean degree')
# ax[0, 1].set_ylabel('standard deviation from mean degree')
# ax[1, 0].set_ylabel('mean shortest path length')
# ax[1, 1].set_ylabel('standard deviation from mean shortest path length')

# ax[0, 0].set_xlabel('s') ; ax[0, 1].set_xlabel('s') ; ax[1, 0].set_xlabel('s') ; ax[1, 1].set_xlabel('s')
# plt.show()
    
#     # g.add_interactions_from(G.edges(), t=nn)

# plt.figure()
# plt.plot(np.linspace(0, 1, n_points - 1), (std_deg[1:] - std_deg[:-1])/(dt*100))
# plt.show()

# exit()  
# # paths = time_respecting_paths(g, 0, 3)#, start=1, end=9)
#######################################################
# for n in range(N_qubits):
#     for j in range(N_qubits-1-i)
#     plt.plot(np.linspace(0, 1, n_points), ent_1to1_gs)




print('min gap = ', min(gaps))
gp = gaps[0]
kkk = 0
while gp != min(gaps):
    kkk += 1
    gp = gaps[kkk]
print('minimum gap located at s =', kkk/n_points)




A = 0.05

fig, ax1 = plt.subplots()
eses = np.linspace(0, 1, n_points)
combinationofspins = [0, 1, 2, 3, 4]
markers = ['o', 'o', 'o', 'o', 'o']
ax1.set_xlabel('s')
ax1.set_ylabel('entanglement entropy')

for i in range(N_qubits):
    ax1.plot(eses, ent_1to1_gs[:,i], marker = markers[i], linestyle = '', ms = 3, label = str(combinationofspins[i]), alpha = 0.7)
ax1.tick_params(axis='y',)
ax1.legend(loc = 'center right', title = 'traced out spin')
ax1.set_title('ground state')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'brown' ### '#ff7f0e'
ax2.set_ylabel('$\\Delta$', color=color)  # we already handled the x-label with ax1
ax2.plot(eses, gaps, '--', color=color, alpha = 0.5)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()





fig, ax1 = plt.subplots()
eses = np.linspace(0, 1, n_points)
ax1.set_xlabel('s')
ax1.set_ylabel('entanglement entropy')

for i in range(N_qubits):
    ax1.plot(eses, ent_1to1_1exc[:,i], marker = markers[i], linestyle = '', ms = 3, label = str(combinationofspins[i]), alpha = 0.7)
ax1.tick_params(axis='y',)
ax1.legend(loc = 'center right', title = 'traced out spin')
ax1.set_title('first excited state')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'brown' ### '#ff7f0e'
ax2.set_ylabel('$\\Delta$', color=color)  # we already handled the x-label with ax1
ax2.plot(eses, gaps, '--', color=color, alpha = 0.5)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()




fig, ax1 = plt.subplots()
eses = np.linspace(0, 1, n_points)
combinationofspins = [0, 1, 2, 3, 4]
c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 'o', 'o', 'o', 'o']
ax1.set_xlabel('s')
ax1.set_ylabel('entanglement entropy')

for i in range(N_qubits):
    ax1.plot(eses, ent_1to1_gs[:,i], color = c[i], linestyle = '-', label = 'gs, ' + str(combinationofspins[i]), alpha = 0.7)
    ax1.plot(eses, ent_1to1_1exc[:,i], color = c[i], linestyle = '--', label = '1exc, ' + str(combinationofspins[i]), alpha = 0.7)
ax1.tick_params(axis='y',)
ax1.legend(loc = 'center right', title = 'traced out spin')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = '#17becf' ### '#ff7f0e'
ax2.set_ylabel('$\\Delta$', color=color)  # we already handled the x-label with ax1
ax2.plot(eses[0:-1:10], gaps[0:-1:10], 'o', ms = 3, color=color, alpha = 0.5)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()





fig, ax1 = plt.subplots()
eses = np.linspace(0, 1, n_points)
combinationofspins = [(0, 1), (0, 2), (0, 3), (0,4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

ax1.set_xlabel('s')
ax1.set_ylabel('entanglement entropy')

for i in range(combinatorial_coefficient):
    ax1.plot(eses, ent_1to2_gs[:,i], color = c[i], linestyle = '-', label = 'gs, ' + str(combinationofspins[i]), alpha = 0.7)
    ax1.plot(eses, ent_1to2_1exc[:,i], color = c[i], linestyle = '--', label = '1exc, ' + str(combinationofspins[i]), alpha = 0.7)
ax1.tick_params(axis='y',)
ax1.legend(loc = 'center right', title = 'traced out spin')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'r' ### '#ff7f0e'
ax2.set_ylabel('$\\Delta$', color=color)  # we already handled the x-label with ax1
ax2.plot(eses[0:-1:10], gaps[0:-1:10], 'o', ms = 3, color=color, alpha = 0.5)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
#################################################################################

fig, ax = plt.subplots(1, 10, figsize=(20, 6))
for i in range(10):
    ax[i].plot(eses, ent_1to1_gs[:,i], 'o')
    ax[i].set_title(str(combinationofspins[i]))
plt.show()


plt.figure()
for i in range(10): ### actually the combinatorial number (5 2)
    plt.plot(np.linspace(0, 1, n_points)[0:-1:10], ent_1to1_gs[:,i][0:-1:10], 'o', ms = 5 - 0.25 * i, label = str(i), alpha = 0.2)

plt.plot(np.linspace(0, 1, n_points), A * gaps/max(np.ndarray.flatten(ent_1to1_gs)), 'k--')
plt.xlabel('s')
plt.ylabel('bipartite entanglement of qubit 0 with qubit i')
plt.title('gs')
plt.legend()
plt.show()

plt.figure()
for i in range(10): ### actually the combinatorial number (5 2)
    plt.plot(np.linspace(0, 1, n_points)[0:-1:10], ent_1to1_1exc[:, i][0:-1:10], 'o', ms = 5 - 0.25 * i, label = str(i), alpha = 0.2)

plt.plot(np.linspace(0, 1, n_points), A * gaps/max(np.ndarray.flatten(ent_1to1_1exc)), 'k--')
plt.xlabel('s')
plt.ylabel('bipartite entanglement of qubit 0 with qubit i')
plt.title('1 exc')
plt.legend()
plt.show()


mean_ent1to1_gs = np.mean(ent_1to1_gs, axis = 1)
mean_ent1to1_1exc = np.mean(ent_1to1_1exc, axis = 1)

var_ent1to1_gs = np.var(ent_1to1_gs, axis = 1)
var_ent1to1_1exc = np.var(ent_1to1_1exc, axis = 1)

plt.figure()
plt.plot(np.linspace(0, 1, n_points)[0:-1:10], mean_ent1to1_gs[0:-1:10])
plt.plot(np.linspace(0, 1, n_points)[0:-1:10], mean_ent1to1_1exc[0:-1:10])
plt.plot(np.linspace(0, 1, n_points), A * gaps/max(mean_ent1to1_1exc), 'k--')
plt.xlabel('s')
plt.ylabel('mean bipartite entanglement')
plt.show()

plt.figure()
plt.plot(np.linspace(0, 1, n_points)[0:-1:10], var_ent1to1_gs[0:-1:10])
plt.plot(np.linspace(0, 1, n_points)[0:-1:10], var_ent1to1_1exc[0:-1:10])
plt.plot(np.linspace(0, 1, n_points), A * gaps/max(var_ent1to1_1exc), 'k--')
plt.xlabel('s')
plt.ylabel('variace of the bipartite entanglement')
plt.show()

exit()

# 

exit()
plt.figure()
for i in range(N_qubits - 1):
    plt.plot(np.linspace(0, 1, n_points)[0:-1:10], ent_1to1_gs[:, 0, i][0:-1:10], 'o', ms = 5 - 0.5 * i, label = str(i), alpha = 0.2)
plt.xlabel('s')
plt.ylabel('bipartite entanglement of qubit 0 with qubit i')
plt.title('gs')
plt.legend()
plt.show()


plt.figure()
for i in range(N_qubits - 1):
    plt.plot(np.linspace(0, 1, n_points)[0:-1:10], ent_1to1_1exc[:, 0, i][0:-1:10], 's', ms = 5 - 0.5 * i, label = str(i), alpha = 0.2)
plt.xlabel('s')
plt.ylabel('bipartite entanglement of qubit 0 with qubit i')
plt.title('1 exc')
plt.legend()
plt.show()
