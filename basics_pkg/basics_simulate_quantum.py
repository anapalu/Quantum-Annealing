import numpy as np
from qutip import Qobj

import sys
sys.path.append('/home/ana/Documents/PhD')
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
     coord_catalyst = 1, rotate=True, h_mean = 1, W = 0, risingJ = 1334, risingW = 954, randomspinnetX_J = 16234, randomspinnetX_W = 954, \
     mxdf = 3, number_of_ancillas = 0, return_HiHf = False):

    dims_matrices = 2**N_qubits
    Sigma_dict = {} ### Nested dictionary, first set coord (0--> 'z', 1 --> 'x', 2 --> 'y') and inside a list/array containing all the corresponding
                ### matrices, ordered by qubit number
    for sp in range(N_qubits):
        Sigma_dict[sp] = np.empty((3, dims_matrices, dims_matrices), 'complex128')
        for coord in range(3):
            Sigma_dict[sp][coord] = Sigma(coord, sp, N_qubits)


    ############################## TRIVIAL HAMILTONIANS
    def transverse_field():
        I = np.eye(dims_matrices)
        H = np.zeros((dims_matrices, dims_matrices), dtype = 'complex128')
        for i in range(N_qubits):
            H += 0.5 * (I - Sigma_dict[i][1])
        return H

    def all_spins_up():
        I = np.eye(dims_matrices)
        H = np.zeros((dims_matrices, dims_matrices), dtype = 'complex128')
        for i in range(N_qubits):
            H += 0.5 * (I - Sigma_dict[i][0])
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

    def kSAT():
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

    def ising_classical():
        ## Generate random couplings
        Js = 1
        np.random.seed(risingJ)
        J = Js * (2 * np.random.rand(N_qubits, N_qubits) - 1)
        ## Generate random fields
        np.random.seed(risingW)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## Generate Hamiltonian
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H += J[i, j] * np.dot(Sigma_dict[i][0], Sigma_dict[j][0])
            H += h[i] * Sigma_dict[i][0]
        return H


    def ising_classical_Xseed():
        ## Generate random couplings
        Js = 1
        np.random.seed(randomspinnetX_J)
        J = Js * (2 * np.random.rand(N_qubits, N_qubits) - 1)
        ## Generate random fields
        np.random.seed(randomspinnetX_W)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## Generate Hamiltonian
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H += J[i, j] * np.dot(Sigma_dict[i][0], Sigma_dict[j][0])
            H += h[i] * Sigma_dict[i][0]
        return H

    def ising_classical_X():
        ## Generate random couplings
        Js = 1
        np.random.seed(randomspinnetX_J)
        J = Js * (2 * np.random.rand(N_qubits, N_qubits) - 1)
        ## Generate random fields
        np.random.seed(randomspinnetX_W)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## Generate Hamiltonian
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H += J[i, j] * np.dot(Sigma_dict[i][1], Sigma_dict[j][1])
            H += h[i] * Sigma_dict[i][1]
        return H

    def ising_classical_X_inverted():
        ## Generate random couplings
        Js = 1
        np.random.seed(randomspinnetX_J)
        J = - Js * (2 * np.random.rand(N_qubits, N_qubits) - 1)
        ## Generate random fields
        np.random.seed(randomspinnetX_W)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## Generate Hamiltonian
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H += J[i, j] * np.dot(Sigma_dict[i][1], Sigma_dict[j][1])
            H += h[i] * Sigma_dict[i][1]
        return H

    def ising_classical_X_complementary():
        ## Generate random couplings
        Js = 1
        np.random.seed(risingJ)
        jotas = (2 * np.random.rand(N_qubits, N_qubits) - 1)
        jotas = np.sign(jotas) * (Js - np.abs(jotas))
        J = jotas
        ## Generate random fields
        np.random.seed(randomspinnetX_W)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## Generate Hamiltonian
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H += J[i, j] * np.dot(Sigma_dict[i][1], Sigma_dict[j][1])
            H += h[i] * Sigma_dict[i][1]
        return H
    
    def ising_classical_X_complementary_plus_noise():
        ## Generate random couplings
        Js = 1
        np.random.seed(risingJ)
        jotas = (2 * np.random.rand(N_qubits, N_qubits) - 1)
        jotas = -np.sign(jotas) * (Js - np.abs(jotas))
        noise = 1*np.random.randint(-1, 2, size=(N_qubits, N_qubits))
        J = jotas + noise
        ## Generate random fields
        np.random.seed(randomspinnetX_W)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## Generate Hamiltonian
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H += J[i, j] * np.dot(Sigma_dict[i][1], Sigma_dict[j][1])
            H += h[i] * Sigma_dict[i][1]
        return H

    def ising_classical_X_adhoc(): ## targeting Ising for seed 2 (N=5) (we want to get close to its inverse here)
                                    ### THE RESULT WAS NEGATIVE 
        ## Generate random couplings
        J = np.array([[0.1, 0.94, -0.1, 0.13, 0.16], 
        [0.34, 0.63, -0.24,0.38, 0.42],
        [-0.24, -0.07, 0.7, -0.01, 0.6],
        [-0.55, -0.68, 0.011, -0.69, 0.81],
        [-0.03, 0.83, 0.17, 0.82, 0.72]])
        ## Generate random fields
        np.random.seed(randomspinnetX_W)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## Generate Hamiltonian
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H += J[i, j] * np.dot(Sigma_dict[i][1], Sigma_dict[j][1])
            H += h[i] * Sigma_dict[i][1]
        return H


    def spin_network():
        ## Generate random couplings
        Js = 1
        np.random.seed(risingJ)
        J = Js * (2 * np.random.rand(N_qubits, N_qubits) - 1)
        ## Generate random fields
        np.random.seed(risingW)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## Generate Hamiltonian
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H += J[i, j] * np.dot(Sigma_dict[i][1], Sigma_dict[j][1])
            H += h[i] * Sigma_dict[i][0]
        return H


    def spin_network_X_nosmessedscale():
        Js = 1
        np.random.seed(randomspinnetX_J)
        J = Js * (2 * np.random.rand(N_qubits, N_qubits) - 1)
        np.random.seed(randomspinnetX_W)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## SPIN NETWORK HAMILTONIAN
        I = np.eye(dims_matrices)
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H += J[i, j] * np.dot(Sigma_dict[i][1], Sigma_dict[j][1]) 
            H += h[i] * Sigma_dict[i][1]
        return H




    def spin_network_X():
        Js = 1
        np.random.seed(randomspinnetX_J)
        J = Js * (2 * np.random.rand(N_qubits, N_qubits) - 1)
        np.random.seed(randomspinnetX_W)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## SPIN NETWORK HAMILTONIAN
        I = np.eye(dims_matrices)
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H += J[i, j] * 0.25 * np.dot((Sigma_dict[i][1] + I), (Sigma_dict[j][1] - I)) ##### OJO CON ESTO, WHY SIGNS?
            H += h[i] * Sigma_dict[i][1]
        amplification_spnX = 3
        H += amplification_spnX*I
        print('Spin network X parameters h=', h_mean, ', W=', W, ', amplification (FOR NOW, THIS IS AN INTERNAL PARAMETER): ', amplification_spnX)
        return H

    def spinnetworkX_perp(): ## completely perpendicular to the one in z
        ## Generate random couplings
        Js = 1
        np.random.seed(randomspinnetX_J)
        J = Js * (2 * np.random.rand(N_qubits, N_qubits) - 1)
        ## Generate random fields
        np.random.seed(randomspinnetX_W)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## Generate Hamiltonian
        H = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H += J[i, j] * np.dot(Sigma_dict[i][0], Sigma_dict[j][0])
            H += h[i] * Sigma_dict[i][1]
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

    def bitflipnoise():
        phi0 = np.ones(dims_matrices)
        return np.eye(dims_matrices) - 1/dims_matrices *(np.kron(phi0, phi0).reshape(dims_matrices, dims_matrices)) 


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
        H_f = kSAT()

    elif final_hamiltonian == 'Ising':
        H_f = ising_classical()

    elif final_hamiltonian == 'Ising Z Xseed':
        H_f = ising_classical_Xseed() 

    elif final_hamiltonian == 'Ising X':
        H_f = ising_classical_X()

    elif final_hamiltonian == 'Ising X inverted':
        H_f = ising_classical_X_inverted()

    elif final_hamiltonian == 'Ising X complementary':
        H_f = ising_classical_X_complementary()

    elif final_hamiltonian == 'Ising X complementary plus noise':
        H_f = ising_classical_X_complementary_plus_noise()

    elif final_hamiltonian == 'Ising X ad hoc':
        H_f = ising_classical_X_adhoc()

    elif final_hamiltonian == 'spin network':
        H_f = spin_network()
        print('Spin network parameters h=', h_mean, ', W=', W)
    
    elif final_hamiltonian == 'spin network X':
        # H_f = spin_network_X()
        H_f = spin_network_X_nosmessedscale()
        # H_f = spinnetworkX_perp()
        print('Spin network X parameters h=', h_mean, ', W=', W)

    else:
        print('You have not  selected any of the available final Hamiltonians')
        exit()




    ## Initial Hamiltonain --> transverse field / all spins up
    if initial_hamiltonian == 'transverse field':
        H_i = transverse_field()


    elif initial_hamiltonian == 'all spins up': ## parallel magnetic field plus some energy shift so that energies are contained within [0, N_qubits]
        H_i = all_spins_up()

    elif initial_hamiltonian == 'Sigmas Z':
        H_i = Sigmas_Z()

    elif initial_hamiltonian == 'all spins down': ## parallel magnetic field plus some energy shift so that energies are contained within [0, N_qubits]
        H_i = all_spins_up()
        for i in range(N_qubits):
            H_i += Sigma_dict[i][0]

    elif initial_hamiltonian == 'Ising X':
        H_i = ising_classical_X()

    elif initial_hamiltonian == 'Ising Z Xseed':
        H_i = ising_classical_Xseed() 

    elif initial_hamiltonian == 'Ising X inverted':
        H_i = ising_classical_X_inverted()

    elif initial_hamiltonian == 'Ising X complementary':
        H_i = ising_classical_X_complementary()

    elif initial_hamiltonian == 'Ising X complementary plus noise':
        H_i = ising_classical_X_complementary_plus_noise()

    elif initial_hamiltonian == 'Ising X ad hoc':
        H_i = ising_classical_X_adhoc()

    elif initial_hamiltonian == 'spin network X':
        # H_i = spin_network_X()
        H_i = spin_network_X_nosmessedscale()
        # H_f = spinnetworkX_perp()
        print('Spin network X parameters h=', h_mean, ', W=', W)

    elif initial_hamiltonian == 'Sigmas':
        H_i = Sigmas()

    elif initial_hamiltonian == 'bit flip noisy': # what happens when we designate directly an equal superposition of all states in the z basis as gs
        H_i = bitflipnoise()


    ########### NOT REALLY USED
    elif initial_hamiltonian == 'phase flip noisy': # what happens when we designate directly the state with all spins pointing up as gs
        phi0 = np.zeros(dims_matrices)  ## all spins up
        phi0[0] = np.sqrt(dims_matrices)
        H_i = np.eye(dims_matrices) - 1/dims_matrices *(np.kron(phi0, phi0).reshape(dims_matrices, dims_matrices))

    elif initial_hamiltonian == 'disrespect bit structure':
        phi0 = np.zeros(dims_matrices) 
        phi0[-1] = np.sqrt(dims_matrices/3)
        phi0[0] = np.sqrt(dims_matrices/3) 
        phi0[4] = np.sqrt(dims_matrices/3)
        ### write phi0 as matrix, see ptraces ############################################
        H_i = np.eye(dims_matrices) - 1/dims_matrices *(np.kron(phi0, phi0).reshape(dims_matrices, dims_matrices))

    elif initial_hamiltonian == 'bit structure':
        H_i = np.zeros((dims_matrices, dims_matrices), dtype = 'complex128')
        I = np.eye(dims_matrices)
        for i in range(N_qubits):
            H_i += 0.5 * (I - Sigma_dict[i][1])

    elif initial_hamiltonian == 'staggered': ### from "Speedup of the Quantum Adiabatic Algorithm using Delocalization Catalysis", C. Cao et al (Dec. 2020)
        H_i = np.zeros((dims_matrices, dims_matrices), dtype = 'complex128')
        for i in range(N_qubits):
            H_i += (-1)**i * Sigma_dict[i][1] 
        
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




    ## Annealing schedule --> linear / optimised Grover / force Landau 
    ## catalyst --> parabola
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

    ############## we never use this one anymore either
    elif annealing_schedule == 'force Landau':
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





    ## Catalyst Hamiltonian 
    if catalyst_interaction == 'Sigmas':
        H_catalyst = Sigmas()

    
    elif catalyst_interaction == 'bit flip noise':
        H_catalyst = bitflipnoise()

    elif catalyst_interaction == 'spin network X':
        H_catalyst = spin_network_X()
        # H_catalyst = spinnetworkX_perp()

    elif catalyst_interaction == 'None':
        H_catalyst = np.zeros((dims_matrices, dims_matrices), 'complex128')


    if return_HiHf == True:
        return Hamiltonian_factory(H_i, H_f, A, B, H_catalyst, C), H_i, H_f, H_catalyst ## break deg, one ancilla per clause
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



### EQUIVALENT WAYS OF BUILDING THEM
# N_qubits = 5
# dims_matrices = 2**N_qubits

# phi0 = np.zeros(dims_matrices)  ## all spins up
# phi0[0] = np.sqrt(dims_matrices)
# H_i = np.eye(dims_matrices) - 1/dims_matrices *(np.kron(phi0, phi0).reshape(dims_matrices, dims_matrices))
# print('gs ', eigh(H_i)[1][:,0])
# print('initial_hamiltonian all spins up (previous) \n', H_i)

# sigx = Sz
# prod = np.kron(0.5 *(np.eye(2) + sigx), 0.5 *(np.eye(2) + sigx)).reshape(4, 4)
# for i in range(2, N_qubits):
#     prod = np.kron(prod, 0.5 *(np.eye(2) + sigx)).reshape(2**(i+1), 2**(i+1))
# I = np.eye(dims_matrices)
# Hi = I - prod

# print('gs prev all spins up (Sz construction) \n', eigh(Hi)[1][:,0])
# print('prev all spins up (Sz construction) \n', Hi)


# I = np.eye(dims_matrices)
# Hii = np.zeros((dims_matrices, dims_matrices), dtype = 'complex128')
# for i in range(N_qubits):
#     Hii += 0.5 * (I - Sigma(0, i, N_qubits))
# print('gs true all spins up \n', eigh(Hii)[1][:,0])
# print('all spins up \n', Hii)

# I = np.eye(dims_matrices)
# Hii = np.zeros((dims_matrices, dims_matrices), dtype = 'complex128')
# for i in range(N_qubits):
#     Hii +=  - Sigma(0, i, N_qubits)
# print('gs z field to everyone \n', eigh(Hii)[1][:,0])
# print('z field to everyone \n', Hii)

# exit()

# H_catalyst = np.zeros((dims_matrices, dims_matrices), 'complex128')
# for i in range(N_qubits):
#     H_catalyst += Sigma(1, i, N_qubits)
# print('gs Sigmasx \n', eigh(H_catalyst)[1][:,0])
# print('Sigmas \n ', H_catalyst)


### IN CONCLUSION: the 'transverse field initial Hamiltonian we have been using and the one defined as in Hii, which we find in the literature, 
#### have the same ground state (of course, we built our transverse field by directly aiming to the equal superposition of all spins as gs), but
##### the higher energy levels differ. This, of course, has a strong impact on the structure of the whole annealing process. The 'transverse field'
###### Hamiltonian as we have defined it actually corresponds to to random, uncorrelated bit flip noise acting on each of the spins individually.
# exit()

# N_qubits = 5
# dims_matrices = 2**N_qubits
# I = np.eye(dims_matrices)
# Hii = np.zeros((dims_matrices, dims_matrices), dtype = 'complex128')
# for i in range(N_qubits):
#     Hii += 0.5 * (I - Sigma(0, i, N_qubits))
# gs = eigh(Hii)[1][:, 0]
# rhogs = np.kron(gs, gs).reshape(dims_matrices, dims_matrices)
# # rhogs_qt = np2qt(rhogs, N_qubits)
# S = np.zeros((dims_matrices, dims_matrices), dtype = 'complex128')
# for i in range(N_qubits):
#     S += Sigma(0, i, N_qubits)
# print(np.trace(np.dot(S, rhogs)))

# exit()
