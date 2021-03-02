import matplotlib.pyplot as plt 
import numpy as np

from scipy.linalg import eigh
from timeit import default_timer as timer

import sys
sys.path.append('/home/ana/Documents/PhD')
from basics_pkg import Sigma, define_Hs, get_mingap_proper, simple_plot, spectral_norm, l2_norm, vN_entropy###, fast_dot




#### SIMULATION BASIC PARAMETERES
N_qubits = 5
dims_matrices = 2**N_qubits
Sigma_dict = {} ### Nested dictionary, first set coord (0--> 'z', 1 --> 'x', 2 --> 'y') and inside a list/array containing all the corresponding
                ### matrices, ordered by qubit number
for sp in range(N_qubits):
    Sigma_dict[sp] = np.empty((3, dims_matrices, dims_matrices), 'complex128')
    for coord in range(3):
        Sigma_dict[sp][coord] = Sigma(coord, sp, N_qubits)

W = 0
risingW = randomspinnetX_W = 1
h_mean = 1 #1#

print('h = {}'.format(h_mean))

def Ising_factory(Jseed, step, Wseed = 0): 
    if step == 2:
        ######### FINAL HAMILTONIAN: CLASSICAL ISING
        ## Generate random couplings
        Js = 1
        np.random.seed(Jseed)
        J = Js * (2 * np.random.rand(N_qubits, N_qubits) - 1)
        ## Generate random fields
        np.random.seed(Wseed)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## Generate Hamiltonian
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H += J[i, j] * np.dot(Sigma_dict[i][0], Sigma_dict[j][0])
            H += h[i] * Sigma_dict[i][0]
        return H


    elif step == 1:  #### MEDIATING HAMILTONIAN IS THE A DIFFERENT RANDOM INSTANCE
        Js = 1
        np.random.seed(Jseed)
        J = Js * (2 * np.random.rand(N_qubits, N_qubits) - 1)
        ## Generate random fields
        np.random.seed(Wseed)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## Generate Hamiltonian
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H += J[i, j] * np.dot(Sigma_dict[i][1], Sigma_dict[j][1])
            H += h[i] * Sigma_dict[i][1]
        return H


    elif step == 0: #### INITIAL HAMILTONIAN, Z-FIELDS
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            H += Sigma_dict[i][0]
        return H

    elif step == 'None': #### PROBLEM HAMILTONIAN ON THE X DIRECTION, SO AS TO RUN THE ORIGINAL PROCESS
        Js = 1
        np.random.seed(Jseed)
        J = Js * (2 * np.random.rand(N_qubits, N_qubits) - 1)
        ## Generate random fields
        np.random.seed(Wseed)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## Generate Hamiltonian
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H += J[i, j] * np.dot(Sigma_dict[i][1], Sigma_dict[j][1])
            H += h[i] * Sigma_dict[i][1]
        return H


    elif step == '1Q':  #### MEDIATING HAMILTONIAN IS A DIFFERENT RANDOM ISING
        Js = 1
        np.random.seed(Jseed)
        J = Js * (2 * np.random.rand(N_qubits, N_qubits) - 1)
        ## Generate random fields
        np.random.seed(Wseed)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## Generate Hamiltonian
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H += J[i, j] * np.dot(Sigma_dict[i][1], Sigma_dict[j][1])
            H += h[i] * Sigma_dict[i][0]
        return H

    elif step == 'complementary plus noise':  #### MEDIATING HAMILTONIAN IS THE COMPLEMENTARY ISING
        Js = 1
        np.random.seed(Jseed)
        jotas = (2 * np.random.rand(N_qubits, N_qubits) - 1)
        jotas = np.sign(jotas) * (Js - np.abs(jotas))
        noise = 0.1 * (2 * np.random.rand(N_qubits, N_qubits) - 1)
        J = jotas + noise
        ## Generate random fields
        np.random.seed(Wseed)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## Generate Hamiltonian
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H += J[i, j] * np.dot(Sigma_dict[i][1], Sigma_dict[j][1])
            H += h[i] * Sigma_dict[i][1]
        return H

    
    elif step == '1QX':  #### MEDIATING HAMILTONIAN IS A DIFFERENT RANDOM ISING
        Js = 1
        np.random.seed(Jseed)
        J = Js * (2 * np.random.rand(N_qubits, N_qubits) - 1)
        ## Generate random fields
        np.random.seed(Wseed)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## Generate Hamiltonian
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H += J[i, j] * np.dot(Sigma_dict[i][0], Sigma_dict[j][0])
            H += h[i] * Sigma_dict[i][1]
        return H





def Hamiltonian_factory(H_i, H_f):
    def H_s(s):
        return (1-s) * H_i + s * H_f 
    return H_s


#### SIMULATION TIME SCALES
T_anneal = 100#100 ## usual values to try out: T=100, dt = 0.01
dt = 0.1
ds = dt / (T_anneal/2) ######################## we set ds to be half the total anneal time so that running the two anneals takes T_anneal
ds0 = dt/ T_anneal
n_points = 2 * int(1/ds)


#### SIMULATION MAGNITUDE
n_samples = 100 ### number of different samples

basicseed = 35#7535#1234
np.random.seed(basicseed)
seeds_ising = np.random.randint(1, 1000000, n_samples)


vN = np.empty(100)
for iii, j in enumerate(seeds_ising):
    np.random.seed(j)
    Jz = 2*np.random.rand(N_qubits**2) - 1
    Jz = np.abs(Jz)
    vN[iii] = vN_entropy(Jz)

np.random.seed(15)#(977805)
Jx = 2*np.random.rand(N_qubits**2) - 1
Jx = np.abs(Jz)
vN_Jx = vN_entropy(Jx) * np.ones(n_samples)
# seeds_isingX = np.random.randint(1, 1000000, n_samples)


middlestep = 'complementary plus noise' #'1QX'
print('middle Hamiltonian', middlestep)


mingaps0 = [] ### in direct anneal
mingaps1 = []
mingaps2 = []
mingaps = [] ## between first and second step

Tad0 = []
Tad1 = []
Tad2 = []

t0 = timer()
for n in range(n_samples):

    risingJ = seeds_ising[n] #15#2#
    randomspinnetX_J = seeds_ising[n]#977805#seeds_isingX[n]#67 #seeds_ising[n]# #### WE WILL START WITH THE STATISTICS OF INVERTING COUPLINGS

    H_f = Ising_factory(risingJ, step = 2)
    # H_m = Ising_factory(randomspinnetX_J, step = 1)
    H_m = Ising_factory(randomspinnetX_J, step = middlestep)
    H_i = Ising_factory('None', step = 0)

    H_F = Ising_factory(risingJ, step = 'None')



    H_s0 = Hamiltonian_factory(H_i, H_F)
    H_s1 = Hamiltonian_factory(H_i, H_m)
    H_s2 = Hamiltonian_factory(H_m, H_f)


    gaps_01_0 = []
    gaps_01_1 = []
    gaps_01_2 = []

    s_2stp = 0 - ds
    s0 = 0 - ds0

    for ns in range(n_points//2):
        s0 += ds0
        s_2stp += ds

        H0 = H_s0(s0)
        eigvals, P = eigh(H0) 
        gaps_01_0 += [eigvals[1] - eigvals[0]]

        H1 = H_s1(s_2stp)
        eigvals, P = eigh(H1) 
        gaps_01_1 += [eigvals[1] - eigvals[0]]

        H2 = H_s2(s_2stp)
        eigvals, P = eigh(H2) 
        gaps_01_2 += [eigvals[1] - eigvals[0]]

    for ns in range(n_points//2):
        s0 += ds0
        H0 = H_s0(s0)
        eigvals, P = eigh(H0) 
        gaps_01_0 += [eigvals[1] - eigvals[0]]


    # for ns in range(n_points):
    #     s0 = ds0 * ns
    #     H0 = H_s0(s0)

    #     eigvals, P = eigh(H0) 
    #     gaps_01_0 += [eigvals[1] - eigvals[0]]

    #     if ns <n_points//2:
    #         s_2stp += ds
    #         H1 = H_s1(s_2stp)

    #         eigvals, P = eigh(H1) 
    #         gaps_01_1 += [eigvals[1] - eigvals[0]]

    #         H2 = H_s2(s_2stp)

    #         eigvals, P = eigh(H2) 
    #         gaps_01_2 += [eigvals[1] - eigvals[0]]

    ming = min(gaps_01_0)                                    ################ SEE ABOUT LOOKING AT THE AVERAGE LOCATION AS WELL
    mingaps0 += [ming]
    Hss = [spectral_norm(H_s0(s)) for s in np.linspace(0, 1, 1000)]
    max_Hss = max(Hss)
    T0 = spectral_norm(H_F - H_i) * max_Hss /(ming**2)
    Tad0 += [T0]

    ming1 = min(gaps_01_1)                                    ################ SEE ABOUT LOOKING AT THE AVERAGE LOCATION AS WELL
    mingaps1 += [ming1]
    Hss = [spectral_norm(H_s1(s)) for s in np.linspace(0, 1, 1000)]
    max_Hss = max(Hss)
    T1 = spectral_norm(H_m - H_i) * max_Hss /(ming1**2)
    Tad1 += [T1]

    ming2 = min(gaps_01_2)                                    ################ SEE ABOUT LOOKING AT THE AVERAGE LOCATION AS WELL
    mingaps2 += [ming2]
    Hss = [spectral_norm(H_s2(s)) for s in np.linspace(0, 1, 1000)]
    max_Hss = max(Hss)
    T2 = spectral_norm(H_f - H_m) * max_Hss /(ming2**2)
    Tad2 += [T2]






    # ### DIRECT ANNEAL
    # gaps_01 = []
    # H_s = Hamiltonian_factory(H_i, H_F)
    # for ns in range(n_points):
    #     s = ds0 * ns
    #     H = H_s(s)

    #     eigvals, P = eigh(H) 
    #     gaps_01 += [eigvals[1] - eigvals[0]]
    # ming = min(gaps_01)                                    ################ SEE ABOUT LOOKING AT THE AVERAGE LOCATION AS WELL
    # mingaps0 += [ming]

    # Hss = [spectral_norm(H_s(s)) for s in np.linspace(0, 1, 1000)]
    # max_Hss = max(Hss)
    # T0 = spectral_norm(H_F - H_i) * max_Hss /(ming**2)
    # Tad0 += [T0]

    # ###### TWO-STEP
    # ### FIRST ANNEAL
    # gaps_01 = []
    # H_s = Hamiltonian_factory(H_i, H_m)
    # for ns in range(n_points//2):
    #     s = ds * ns
    #     H = H_s(s)

    #     eigvals, P = eigh(H) 
    #     gaps_01 += [eigvals[1] - eigvals[0]]
    # ming1 = min(gaps_01)                                    ################ SEE ABOUT LOOKING AT THE AVERAGE LOCATION AS WELL
    # mingaps1 += [ming1]

    # Hss1 = [spectral_norm(H_s(s)) for s in np.linspace(0, 1, 1000)]
    # max_Hss1 = max(Hss1)
    # T1 = spectral_norm(H_m - H_i) * max_Hss1 /(ming1**2)
    # Tad1 += [T1]

    # ### SECOND ANNEAL
    # gaps_01 = []
    # H_s = Hamiltonian_factory(H_m, H_f)
    # for ns in range(n_points//2):
    #     s = ds * ns
    #     H = H_s(s)

    #     eigvals, P = eigh(H) 
    #     gaps_01 += [eigvals[1] - eigvals[0]]
    # ming2 = min(gaps_01)
    # mingaps2 += [ming2]

    # Hss2 = [spectral_norm(H_s(s)) for s in np.linspace(0, 1, 1000)]
    # max_Hss2 = max(Hss2)
    # T2 = spectral_norm(H_f - H_m) * max_Hss2 /(ming2**2)
    # Tad2 += [T2]

    mingaps += [min(ming1, ming2)]


t1 = timer()
print('it took {} s to calculate {} samples'.format(t1-t0, n_samples))

Tad0 = np.asarray(Tad0)

Tad1 = np.asarray(Tad1)
Tad2 = np.asarray(Tad2)
Tad_tot = Tad1 + Tad2

print('mean T0, mean T_tot', np.mean(Tad0), np.mean(Tad_tot))



figsz = (12, 16)
xaxis = np.arange(n_samples) ; xlabel = '# sample'

################################# PLOTTING

fig, ax = plt.subplots( figsize = figsz)
ylabel = '$T_{0} - (T_{1} + T_{2})$'

fig, ax = simple_plot(fig, ax, xaxis, (Tad0 - Tad_tot) , xlabel, ylabel, markr = 'o', alpha = 1, reduce = False)
fig, ax = simple_plot(fig, ax, xaxis, np.zeros(n_samples), xlabel, ylabel, markr = '--', alpha = 0.4, reduce = False)
# plt.show()

fig, ax = plt.subplots( figsize = figsz)
ylabel = '$T_{ad}$'

fig, ax = simple_plot(fig, ax, xaxis, Tad0 , xlabel, ylabel, markr = '-', alpha = 0.4, reduce = False)
fig, ax = simple_plot(fig, ax, xaxis, Tad1, xlabel, ylabel, markr = '--', alpha = 0.4, reduce = False)
# plt.show()


figsz = (16, 12)
fig, ax = plt.subplots(figsize = figsz)
ax.plot((vN_Jx- vN)**2, (Tad0 - Tad_tot), 'o', ms = 3)#, np.var(gaps, axis = 1))
ax.plot((vN_Jx- vN)**2, np.zeros(n_samples), '--', alpha =0.4)
ax.set_xlabel('$(vNJx(fixed) - vNJz)^2$')
ax.set_ylabel('(Tad0 - Tad_tot)')

plt.show()