import numpy as np
from qutip import Qobj

import sys
sys.path.append('/home/ana/Documents/PhD/my-projects/')
from basics_pkg.basics_pauli import Sigma, arbitrary_rotation_spins
from basics_pkg.basics_molecular_Hamiltonian import get_molecular_Hamiltonian
from basics_pkg.basics_manage_data import retrieve_instances

from scipy.linalg import eigh





def Hamiltonian_factory(Hi, Hf, A, B, Hcat = 0, C = 0, Hcat1 = 0, C1 = 0):
    if C == 0:
        def H_t(s):
            return A(s) * Hi + B(s) * Hf
    else:
        if C1 == 0:
            def H_t(s):
                return A(s) * Hi + B(s) * Hf + C(s) * Hcat
        else:
            def H_t(s):
                return A(s) * Hi + B(s) * Hf + C(s) * Hcat + C1(s) * Hcat1 
    return H_t


def get_measurement_probabilities(rho, eigvals_op, eigvects_op):
    dims_matrices = rho.shape[0] ## recall that we have to reshape after the kroenecker product
    ##In order to account for degeneracies we must be careful when projecting to the whole eigenspace
    projector = np.zeros(( len(np.unique(eigvals_op)) , dims_matrices, dims_matrices), dtype = 'complex128') #OJO# NOT SURE ABOUT THE +1 HERE
    ## No problem with using np.unique() because the eigenvalues are already ordered
    k = 0
    egval_previous = 'NaN' #initialise index k and previous eigval
    for i, egval in enumerate(eigvals_op):
        projector[k] += np.kron(eigvects_op[i], eigvects_op[i]).reshape(dims_matrices, dims_matrices)
        
        k += egval != egval_previous ## If true, k=1, and if false, k=0
        egval_previous = egval #update previous eigenvalue

    probs = []
    for pr in projector:
        # projection = np.dot(pr, np.dot(rho, pr))
        probs += [np.trace(np.real(np.dot(rho, pr)))]  #[np.sqrt( np.real(np.trace(np.dot(np.conj(projection), projection))))]
    return probs, projector


def build_probability_distr(probs, nondegenerate_eigvals, r): ##returns the resulting random eigenvalue of the measurement 
                                                                ## nondegenerate_eigals is np.unique(eigvals) and r is a random number
    cumulative_prob = probs[0]
    i = 0
    while r - cumulative_prob > 0 and i <= len(probs) - 1:
        cumulative_prob += probs[i+1]
        i += 1
    return nondegenerate_eigvals[i]



def np2qt(mat, N): ### N is the number of spins
    return Qobj(mat, dims = [[2 for i in range(N)], [2 for i in range(N)]] )






##################################################################


def define_Hs(N_qubits, final_hamiltonian, initial_hamiltonian, annealing_schedule, catalyst = 'None', catalyst_interaction = 'None', \
     coord_catalyst = 1, rotate=True, h_mean_f = 1, W_f = 0, h_mean_i = 1, W_i = 0, seedJ_f = 1334, seedW_f = 954, seedJ_i = 16234, seedW_i = 954, \
     seedJ_cat = 124, seedW_cat = 6543, h_mean_cat = 1, W_cat = 0, mxdf = 3, number_of_ancillas = 0, return_HiHf = False):

    ############### BASIC DEFINITIONS
    dims_matrices = 2**N_qubits
    Sigma_dict = {} ### Nested dictionary, first set coord (0--> 'z', 1 --> 'x', 2 --> 'y') and inside a list/array containing all the corresponding
                ### matrices, ordered by qubit number
    for sp in range(N_qubits):
        Sigma_dict[sp] = np.empty((3, dims_matrices, dims_matrices), 'complex128')
        for coord in range(3):
            Sigma_dict[sp][coord] = Sigma(coord, sp, N_qubits)


    ############################## TRIVIAL HAMILTONIANS
    def staggered(): ### from "Speedup of the Quantum Adiabatic Algorithm using Delocalization Catalysis", C. Cao et al (Dec. 2020)
        H = np.zeros((dims_matrices, dims_matrices), dtype = 'complex128')
        for i in range(N_qubits):
            H += (-1)**i * Sigma_dict[i][1] 
        return H

    def bit_struct():
        H = np.zeros((dims_matrices, dims_matrices), dtype = 'complex128')
        I = np.eye(dims_matrices)
        for i in range(N_qubits):
            H += 0.5 * (I - Sigma_dict[i][1])
        return H

    def disrespect_bit_struct():
        phi0 = np.zeros(dims_matrices) 
        phi0[-1] = np.sqrt(dims_matrices/3)
        phi0[0] = np.sqrt(dims_matrices/3) 
        phi0[4] = np.sqrt(dims_matrices/3)
        ### write phi0 as matrix, see ptraces ############################################
        H = np.eye(dims_matrices) - 1/dims_matrices *(np.kron(phi0, phi0).reshape(dims_matrices, dims_matrices))
        return H

    def transverse_field():
        I = np.eye(dims_matrices)
        H = np.zeros((dims_matrices, dims_matrices), dtype = 'complex128')
        for i in range(N_qubits):
            H += 0.5 * (I - Sigma_dict[i][1])
        return H
    
    def bitflipnoise():
        phi0 = np.ones(dims_matrices)
        return np.eye(dims_matrices) - 1/dims_matrices *(np.kron(phi0, phi0).reshape(dims_matrices, dims_matrices))

    def all_spins_up():
        I = np.eye(dims_matrices)
        H = np.zeros((dims_matrices, dims_matrices), dtype = 'complex128')
        for i in range(N_qubits):
            H += 0.5 * (I - Sigma_dict[i][0])
        return H

    def phaseflipnoise():
        phi0 = np.zeros(dims_matrices)  ## all spins up
        phi0[0] = np.sqrt(dims_matrices)
        H = np.eye(dims_matrices) - 1/dims_matrices *(np.kron(phi0, phi0).reshape(dims_matrices, dims_matrices))
        return H


    def Sigmas():
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            H += Sigma_dict[i][coord_catalyst] #0.5 * Sigma_dict[i][coord_catalyst]
        return H

    def Sigmas_Z():
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            H += Sigma_dict[i][0] # 0.5 * 
        return H


    ############################## COMPLEX HAMILTONIANS
    def molecular():
        H = get_molecular_Hamiltonian() ## this gets me H2, N_qubits = 5
        dims_matrices = H.shape[0]
        return np.asarray(H)

    def Grover():
        # Set target state
        s0 = np.zeros(2**N_qubits)
        s0[0] = 1
        if rotate == True:
            print('We rotate the target state')
            np.random.seed(1234)
            R = arbitrary_rotation_spins(N_qubits)
            s0 = np.dot(R, s0)
            print('target state', s0)
        return np.eye(2**N_qubits) - np.kron(s0, s0).reshape(2**N_qubits, 2**N_qubits)

    def kSAT(mxdf):
        k = 3; n = N_qubits - number_of_ancillas
        filename = '/home/ana/Documents/PhD/kSAT/instances/{}sat_n{}_seed1234.txt'.format(k, n) #'/home/ana/Documents/PhD/kSAT/instances/{}sat_n{}_seed1234_mixeddiff.txt'.format(k, n)
        insts, sols = retrieve_instances(filename, k)
        I = np.eye(dims_matrices)
        H = np.zeros((dims_matrices, dims_matrices), dtype='complex128')
        Sigmasz = [Sigma(0, i, N_qubits) for i in range(N_qubits) ]

        for clause in insts[mxdf]: ## even if it's just one, we still need to unpack from the list format # 70
            i, j, k = clause - 1 ## because spins are numbered from 1 to n
            sz_i, sz_j, sz_k = Sigmasz[i], Sigmasz[j], Sigmasz[k]
            i_p, j_p, k_p = I + sz_i, I + sz_j, I + sz_k
            i_m, j_m, k_m = I - sz_i, I - sz_j, I - sz_k
            sumsigmas = np.dot(i_p, np.dot(j_p, k_p) ) + np.dot(i_m, np.dot(j_m, k_m) ) + np.dot(i_p, np.dot(j_m, k_m) ) +  \
                    np.dot(i_m, np.dot(j_p, k_m) ) + np.dot(i_m, np.dot(j_m, k_p) )
            H += 0.125 * sumsigmas

        if rotate == True:
            print('We rotate the whole H_P')
            np.random.seed(1234)
            R = arbitrary_rotation_spins(N_qubits)
            H = np.dot( np.conj(R.T), np.dot(H, R) )
        
        return H

    def ising_classical(seedJ, seedW, h_mean, W, coord = 0): ## coord = 0 is in z, coord = 1 is in x
        ## Generate random couplings
        Js = 1
        np.random.seed(seedJ)
        J = Js * (2 * np.random.rand(N_qubits, N_qubits) - 1)
        ## Generate random fields
        np.random.seed(seedW)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## Generate Hamiltonian
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H += J[i, j] * np.dot(Sigma_dict[i][coord], Sigma_dict[j][coord])
            H += h[i] * Sigma_dict[i][coord]
        return H

    def ising_quantum(seedJ, seedW, h_mean, W, coord = 0): ### if coord = 0 the transverse field is in x and the couplings in x, if coord = 1 viceversa
        ## Generate random couplings
        Js = 1
        np.random.seed(seedJ)
        J = Js * (2 * np.random.rand(N_qubits, N_qubits) - 1)
        ## Generate random fields
        np.random.seed(seedW)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## Generate Hamiltonian
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H += J[i, j] * np.dot(Sigma_dict[i][(coord+1)%2], Sigma_dict[j][(coord+1)%2])
            H += h[i] * Sigma_dict[i][coord]
        return H

     


        ##################################################3



    ## Final Hamiltonian --> molecular / Grover / spin network / kSAT

    if final_hamiltonian == 'molecular':
        if N_qubits != 4:
            print('The current molecule produces N_qubits = 4. Please make these two parameters consistent')
            exit()
        H_f = molecular()

    elif final_hamiltonian == 'Sigmas Z':
        H_f = Sigmas_Z()
        
    elif final_hamiltonian == 'Grover':
        H_f = Grover()
    
    elif final_hamiltonian == 'kSAT':
        H_f = kSAT(mxdf)

    elif final_hamiltonian == 'Ising':
        H_f = ising_classical(seedJ_f, seedW_f, h_mean_f, W_f)

    elif final_hamiltonian == 'Ising X':
        H_f = ising_classical(seedJ_f, seedW_f, h_mean_f, W_f, coord = 1)

    elif final_hamiltonian == 'Ising quantum':
        H_f = ising_quantum(seedJ_f, seedW_f,  h_mean_f, W_f)
        print('Quantum Ising parameters (final Hamiltonian) h=', h_mean_f, ', W=', W_f, 'couplings in z and transverse field in x')
    
    elif final_hamiltonian == 'spin network X':
        H_f = ising_quantum(seedJ_f, seedW_f,  h_mean_f, W_f, coord = 1)
        print('Quantum Ising parameters (final Hamiltonian) h=', h_mean_f, ', W=', W_f, 'couplings in x and transverse field in z')

    else:
        print('You have not  selected any of the available final Hamiltonians')
        exit()




    ########################### Initial Hamiltonain

    if initial_hamiltonian == 'Sigmas Z':
        H_i = Sigmas_Z()

    elif initial_hamiltonian == 'Ising':
        H_i = ising_classical(seedJ_i, seedW_i, h_mean_i, W_i)

    elif initial_hamiltonian == 'Ising X':
        H_i = ising_classical(seedJ_i, seedW_i, h_mean_i, W_i, coord=1)

    elif initial_hamiltonian == 'Ising quantum':
        H_i = ising_quantum(seedJ_i, seedW_i, h_mean_i, W_i)
        print('Quantum Ising parameters (initial Hamiltonian) h=', h_mean_i, ', W=', W_i, 'couplings in z, transverse field in x')

    elif initial_hamiltonian == 'Ising quantum X':
        H_i = ising_quantum(seedJ_i, seedW_i, h_mean_i, W_i, coord = 1)
        print('Quantum Ising parameters (initial Hamiltonian) h=', h_mean_i, ', W=', W_i, 'couplings in x, transverse field in z')

    elif initial_hamiltonian == 'Sigmas':
        H_i = Sigmas()

    


    ########### NOT REALLY USED
    elif initial_hamiltonian == 'transverse field':
        H_i = transverse_field()

    elif initial_hamiltonian == 'all spins up': ## parallel magnetic field plus some energy shift so that energies are contained within [0, N_qubits]
        H_i = all_spins_up()

    elif initial_hamiltonian == 'bit flip noisy': # what happens when we designate directly an equal superposition of all states in the z basis as gs
        H_i = bitflipnoise()

    elif initial_hamiltonian == 'phase flip noisy': # what happens when we designate directly the state with all spins pointing up as gs
        H_i = phaseflipnoise()

    elif initial_hamiltonian == 'disrespect bit structure':
        H_i = disrespect_bit_struct()
        
    elif initial_hamiltonian == 'bit structure':
        H_i = bit_struct()

    elif initial_hamiltonian == 'staggered': ### from "Speedup of the Quantum Adiabatic Algorithm using Delocalization Catalysis", C. Cao et al (Dec. 2020)
        H_i = staggered()
        
    else:
        print('You have not  selected any of the available initial Hamiltonians')
        exit()


    ##################################################################################### SCHEDULES
    def linear_i(s):
        return 1-s

    def linear_f(s):
        return s

    def parabola(s):
        return s*(1-s)

    def linear_i_onehalf(s):
        if s <= 0.5:
            return 1 - 2*s
        else:
            return 0

    def linear_f_onehalf(s):
        if s <= 0.5:
            return 0
        else:
            return 2*s - 1

    def linear_i_onethird(s):
        if s <= 2/3:
            return 1 - 1.5*s
        else:
            return 0

    def linear_f_onethird(s):
        if s <= 1/3:
                return 0
        else:
            return 1.5*s - 0.5

    def optimised_Grover(s):
        b = np.sqrt(N_qubits - 1)
        a = (2*s - 1) * np.arctan(b)
        return 1 - (0.5 + 0.5 / np.sqrt(N_qubits - 1) * np.tan(a) )

    def null(s):
        return 0

    ############## annealin_schedule
    if annealing_schedule == 'linear':
        A = linear_i
        B = linear_f

    elif annealing_schedule == 'smooth two-step':
        A = linear_i_onehalf
        B = linear_f_onehalf

    elif annealing_schedule == 'smooth two-step 2':
        A = linear_i_onethird
        B = linear_f_onethird

    elif annealing_schedule == 'optimised Grover':
        A = optimised_Grover
        def B(s):
            return 1 - A(s)

    elif annealing_schedule == 'force Landau': ### we never use this one anymore either
        def A(s):
            return 1 - 0.5 * (np.tanh(5*(s-0.5)) + 1) ### Forcing Landau levels toghether
        def B(s):
            return 0.5 * (np.tanh(5*(s-0.5)) + 1)
    else:
        print('You have not  selected any of the available anealing schedules')
        exit()


    if catalyst == 'None':
        C = null
        
    elif catalyst == 'parabola':
        C = parabola

    elif catalyst == 'smooth two-step':
        def C(s):
            return 4 * parabola(s)

    else:
        print('You have not  selected any of the available catalysts')
        exit()



    ############################################ Catalyst Hamiltonian 
    if catalyst_interaction == 'Sigmas':
        H_catalyst = Sigmas()

    
    elif catalyst_interaction == 'bit flip noise':
        H_catalyst = bitflipnoise()

    elif catalyst_interaction == 'Ising quantum':
        H_catalyst = ising_quantum(seedJ_cat, seedW_cat, h_mean_cat, W_cat)

    elif catalyst_interaction == 'None':
        H_catalyst = np.zeros((dims_matrices, dims_matrices), 'complex128')


    if return_HiHf == True:
        return Hamiltonian_factory(H_i, H_f, A, B, H_catalyst, C), H_i, H_f, H_catalyst
    else:
        return Hamiltonian_factory(H_i, H_f, A, B, H_catalyst, C)
















def get_corrections(H0, H1, H2, dims_matrices, precision):
    eigvals, P = eigh(H0) 
    eigvals = np.round(eigvals, precision)
    eigvals_unique = np.unique(eigvals)
    number_distinct_eigvals = len(eigvals_unique)

    E0 = eigvals

    projector = np.zeros(( number_distinct_eigvals , dims_matrices, dims_matrices), dtype = 'complex128') 
    k = 0 ## initialise index k
    egval_previous = eigvals[0] ## initialise previous eigval
    projector[0] += np.kron(P[:,0], P[:,0]).reshape(dims_matrices, dims_matrices) ## initialise gs projector
    list_of_degeneracies = np.zeros(number_distinct_eigvals, dtype = int)
    for ee, egval in enumerate(eigvals):
        if ee == 0:
            pass
        else:
            projector[k] += np.kron(P[:,ee], P[:,ee]).reshape(dims_matrices, dims_matrices)
        k += egval != egval_previous ## If true, k+=1, and if false, k+=0
        list_of_degeneracies[k] += 1
        egval_previous = egval #update previous eigenvalue
    #############################################

    print('list of degeneracies', list_of_degeneracies)

    P0VP0 = np.array([np.dot(projector[i], np.dot(H1, projector[i])) for i in range(number_distinct_eigvals)])

    first_correction_E = []#np.zeros(dims_matrices)
    zeroth_correction_state = np.zeros((dims_matrices, dims_matrices), dtype = 'complex128')
    kkk = 0
    for iii, pvp in enumerate(P0VP0):
        E1, l0s = eigh(pvp)
        E1 = np.round(E1, precision)
        [first_correction_E.append(x) for x in E1[-list_of_degeneracies[iii]:] ]
        for x in l0s[:, -list_of_degeneracies[iii]:].T :
            zeroth_correction_state[kkk, :] += x
            kkk += 1

    E1 = np.asarray(first_correction_E)

    E2 = np.zeros(dims_matrices)
    for i in range(dims_matrices): ## the part coming from the expected value of the zeroth-order correction on the second-order Hamiltonian correction
        l0 = zeroth_correction_state[:, i]
        rho_l0 = np.kron(l0, l0).reshape(dims_matrices, dims_matrices)
        E2[i] += np.trace(np.real(np.dot(H2, rho_l0)) )

        for j, e in enumerate(E0):
            if e != E0[i]:
                E2[i] += np.real(np.conj(H1[j, i]) * H1[j, i]) / (E0[i] - e)

    E2 = np.round(E2, precision)
    return E0, E1, E2

