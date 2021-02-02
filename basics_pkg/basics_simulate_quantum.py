import numpy as np
from qutip import Qobj
from basics_pauli import *
from basics_molecular_Hamiltonian import get_molecular_Hamiltonian
from basics_manage_data import retrieve_instances

from scipy.linalg import eigh



def Hamiltonian_factory(Hi, Hf, A, B, Hcat = 0, C = 0):
    if C == 0:
        def H_t(s):
            return A(s) * Hi + B(s) * Hf
    else:
        def H_t(s):
            return A(s) * Hi + B(s) * Hf + C(s) * Hcat
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


def define_Hs(N_qubits, final_hamiltonian, initial_hamiltonian, annealing_schedule, catalyst, coord_catalyst = 1, rotate=True, h_mean = 1, W = 0, mxdf = 3, number_of_ancillas = 1, break_degeneracy = 1, return_HiHf = False):

    if final_hamiltonian == 'molecular' and N_qubits != 4:
        print('The current molecule produces N_qubits = 4. Please make these two parameters consistent')

    dims_matrices = 2**N_qubits
    
    Sigma_dict = {} ### Nested dictionary, first set coord (0--> 'z', 1 --> 'x', 2 --> 'y') and inside a list/array containing all the corresponding
                ### matrices, ordered by qubit number
    for sp in range(N_qubits):
        Sigma_dict[sp] = np.empty((3, dims_matrices, dims_matrices), 'complex128')
        for coord in range(3):
            Sigma_dict[sp][coord] = Sigma(coord, sp, N_qubits)

    ## Final Hamiltonian --> molecular / Grover / spin network / kSAT
    if final_hamiltonian == 'molecular':
        H_f = get_molecular_Hamiltonian() ## this gets me H2, N_qubits = 5
        dims_matrices = H_f.shape[0]
        N_qubits = int(np.log2(dims_matrices))
        H_f = np.asarray(H_f)

    elif final_hamiltonian == 'Grover':
        # Set target state
        s0 = np.zeros(2**N_qubits)
        s0[0] = 1
        if rotate == True:
            print('We rotate the target state')
            np.random.seed(1234)
            R = arbitrary_rotation_spins(N_qubits)
            s0 = np.dot(R, s0)
            print('target state', s0)
        H_f = np.eye(2**N_qubits) - np.kron(s0, s0).reshape(2**N_qubits, 2**N_qubits)



    elif final_hamiltonian == 'kSAT':
        k = 3; n = N_qubits
        
        filename = '/home/ana/Documents/PhD/kSAT/instances/{}sat_n{}_seed1234.txt'.format(k, n) #'/home/ana/Documents/PhD/kSAT/instances/{}sat_n{}_seed1234_mixeddiff.txt'.format(k, n)

        insts, sols = retrieve_instances(filename, k)
        I = np.eye(dims_matrices)
        H_f = np.zeros((dims_matrices, dims_matrices), dtype='complex128')
        Sigmasz = [Sigma(0, i, N_qubits) for i in range(N_qubits) ]

        for clause in insts[mxdf]: ## even if it's just one, we still need to unpack from the list format # 70
            i, j, k = clause - 1 ## because spins are numbered from 1 to n
            sz_i, sz_j, sz_k = Sigmasz[i], Sigmasz[j], Sigmasz[k]
            i_p, j_p, k_p = I + sz_i, I + sz_j, I + sz_k
            i_m, j_m, k_m = I - sz_i, I - sz_j, I - sz_k
            sumsigmas = np.dot(i_p, np.dot(j_p, k_p) ) + np.dot(i_m, np.dot(j_m, k_m) ) + np.dot(i_p, np.dot(j_m, k_m) ) +  \
                    np.dot(i_m, np.dot(j_p, k_m) ) + np.dot(i_m, np.dot(j_m, k_p) )
            H_f += 0.125 * sumsigmas

        
        if rotate == True:
            print('We rotate the whole H_P')
            np.random.seed(1234)
            R = arbitrary_rotation_spins(N_qubits)
            H_f = np.dot( np.conj(R.T), np.dot(H_f, R) )


    elif final_hamiltonian == 'kSAT Dickson':
        k = 3; n = N_qubits - number_of_ancillas
        print('number of ancillas:', number_of_ancillas, '\nstrength of disruption:', break_degeneracy)
        
        filename = '/home/ana/Documents/PhD/kSAT/instances/{}sat_n{}_seed1234.txt'.format(k, n) 
        #'/home/ana/Documents/PhD/kSAT/instances/{}sat_n{}_seed1234_mixeddiff.txt'.format(k, n)

        insts, sols = retrieve_instances(filename, k)
        I = np.eye(dims_matrices)
        H_f = np.zeros((dims_matrices, dims_matrices), dtype='complex128')
        Sigmasz = [Sigma(0, i, N_qubits) for i in range(N_qubits) ]

        kkk = 0 ## for selecting how many degeneracies we want to break
        for clause in insts[mxdf]: ## even if it's just one, we still need to unpack from the list format # 70
            i, j, k = clause - 1 ## because spins are numbered from 1 to n
            sz_i, sz_j, sz_k = Sigmasz[i], Sigmasz[j], Sigmasz[k]
            i_p, j_p, k_p = I + sz_i, I + sz_j, I + sz_k
            i_m, j_m, k_m = I - sz_i, I - sz_j, I - sz_k
            sumsigmas = np.dot(i_p, np.dot(j_p, k_p) ) + np.dot(i_m, np.dot(j_m, k_m) ) + np.dot(i_p, np.dot(j_m, k_m) ) +  \
                    np.dot(i_m, np.dot(j_p, k_m) ) + np.dot(i_m, np.dot(j_m, k_p) )
            H_f += 0.125 * sumsigmas ## np.dot(sumsigmas, 0.5 * (I + Sigmasz[-1])) ### only acting on the last ancilla, did not work
            
        ## break last clause
        for anc in range(number_of_ancillas):
            print('We break deg from last clause')
            H_f += break_degeneracy * 0.125 * 0.5 * np.dot( sumsigmas, (Sigmasz[N_qubits - number_of_ancillas + anc] + I) )
            
            # for anc in range(number_of_ancillas):
            #     if kkk != anc:
            #         H_f += break_degeneracy * 0.125 * 0.5 * np.dot( sumsigmas, (Sigmasz[N_qubits - number_of_ancillas + anc] + I) )
                
            # kkk += 1

        ###############################
        # H_f = H_f / eigh(H_f)[0][-1] ## DOESN'T GIVE GOOD RESULTS EITHER
        ###############################

        if rotate == True:
            print('We rotate the whole H_P')
            np.random.seed(1234)
            R = arbitrary_rotation_spins(N_qubits)
            H_f = np.dot( np.conj(R.T), np.dot(H_f, R) )




    elif final_hamiltonian == 'kSAT Dickson PAPER': ### BUILD THE EXACT SAME 
        k = 3; n = N_qubits - number_of_ancillas
        print('number of ancillas:', number_of_ancillas, '\nstrength of disruption:', break_degeneracy)
        
        filename = '/home/ana/Documents/PhD/kSAT/instances/{}sat_n{}_seed1234.txt'.format(k, n) 
        #'/home/ana/Documents/PhD/kSAT/instances/{}sat_n{}_seed1234_mixeddiff.txt'.format(k, n)

        insts, sols = retrieve_instances(filename, k)
        I = np.eye(dims_matrices)
        H_f = np.zeros((dims_matrices, dims_matrices), dtype='complex128')
        Sigmasz = [Sigma(0, i, N_qubits) for i in range(N_qubits) ]

        kkk = 0 ## for selecting how many degeneracies we want to break
        for clause in insts[mxdf]: ## even if it's just one, we still need to unpack from the list format # 70
            i, j, k = clause - 1 ## because spins are numbered from 1 to n
            sz_i, sz_j, sz_k = Sigmasz[i], Sigmasz[j], Sigmasz[k]

            ## let's apply the translation to Ising only to the first clause
            if kkk == 0:
                sz_anc = Sigmasz[-1]
                H_f += 0.125 * ( 3* (  np.dot(sz_i, sz_anc) + np.dot(sz_j, sz_anc) + np.dot(sz_k, sz_anc)  ) + 24*sz_anc +  \
                        4*np.dot(sz_i, sz_j) + 4*np.dot(sz_i, sz_j) + 4*np.dot(sz_j, sz_k) + 5 * (sz_i + sz_j + sz_k) ) # 77*I  + (inside 0.125 parenthesis) 
            else: 
                i_p, j_p, k_p = I + sz_i, I + sz_j, I + sz_k
                i_m, j_m, k_m = I - sz_i, I - sz_j, I - sz_k
                sumsigmas = np.dot(i_p, np.dot(j_p, k_p) ) + np.dot(i_m, np.dot(j_m, k_m) ) + np.dot(i_p, np.dot(j_m, k_m) ) +  \
                        np.dot(i_m, np.dot(j_p, k_m) ) + np.dot(i_m, np.dot(j_m, k_p) )
                H_f += 0.125 * sumsigmas


            

            
            # for anc in range(number_of_ancillas):
            #     if kkk != anc:
            #         H_f += break_degeneracy * 0.125 * 0.5 * np.dot( sumsigmas, (Sigmasz[N_qubits - number_of_ancillas + anc] + I) )
                
            kkk += 1

        
        if rotate == True:
            print('We rotate the whole H_P')
            np.random.seed(1234)
            R = arbitrary_rotation_spins(N_qubits)
            H_f = np.dot( np.conj(R.T), np.dot(H_f, R) )





    elif final_hamiltonian == 'spin network':
        Js = 1
        np.random.seed(1334)
        J = Js * (2 * np.random.rand(N_qubits, N_qubits) - 1)
        np.random.seed(954)
        hi = W * (2 * np.random.rand(N_qubits) - 1)
        h = h_mean * np.ones(N_qubits) + hi

        ## SPIN NETWORK HAMILTONIAN
        H_f = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H_f += J[i, j] * np.dot(Sigma_dict[i][1], Sigma_dict[j][1])
            H_f += h[i] * Sigma_dict[i][0]
        print('Spin network parameters h=', h_mean, ', W=', W)


    elif final_hamiltonian == 'simple Ising Dickson': ## N_qubits = 6, N_q = 3
        Js = 1
        N_q = N_qubits - number_of_ancillas ### number of operational qubits
        # np.random.seed(1334)
        # J = Js * (2 * np.random.rand(N_q, N_q) - 1)
        J = np.array([[0, 0.5, 0], [0.5, 0, -0.3], [0, -0.3, 0]])
        # np.random.seed(954)
        # hi = W * (2 * np.random.rand(N_q) - 1)
        # h = h_mean * np.ones(N_q) + hi
        h = np.array([-0.75, 0, 0])
        Sigmasz = [Sigma(0, i, N_qubits) for i in range(N_qubits) ]
        I = np.eye(dims_matrices)

        ## ISING HAMILTONIAN
        nn_anc = 0 ## ancilla counter
        H_f = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_q):
            for j in range(i):
                H_f += J[i, j] * np.dot(Sigmasz[i], Sigmasz[j]) 
                if J[i, j] != 0:
                    H_f += break_degeneracy *0.5 * np.dot( (J[i, j] * np.dot(Sigmasz[i], Sigmasz[j]) + I) , (Sigmasz[-1-nn_anc] + I)  )
                    nn_anc += 1
            H_f += h[i] * Sigmasz[i]
            if h[i] != 0:
                H_f += break_degeneracy * 0.5 * np.dot( (h[i] * Sigmasz[i] + I) , (Sigmasz[-1-nn_anc] + I)  )
                nn_anc += 1



        print('Simple Ising with couplings J = {} and h = {}'.format(J, h))


    elif final_hamiltonian == 'simple Ising': ## N_qubits = 3
        J = np.array([[0, 0.5, 0], [0.5, 0, -0.3], [0, -0.3, 0]])

        h = np.array([-0.75, 0, 0])
        Sigmasz = [Sigma(0, i, N_qubits) for i in range(N_qubits) ]

        ## ISING HAMILTONIAN
        H_f = np.zeros((dims_matrices, dims_matrices), 'complex128')
        for i in range(N_qubits):
            for j in range(i):
                H_f += J[i, j] * np.dot(Sigmasz[i], Sigmasz[j]) 
            H_f += h[i] * Sigmasz[i]
            
        print('Simple Ising with couplings J = {} and h = {}'.format(J, h))

    elif final_hamiltonian == 'simple Ising PAPER': ## N_qubits = 3
        Sigmasz = [Sigma(0, i, N_qubits) for i in range(N_qubits) ]
        I = np.eye(dims_matrices)

        ## ISING HAMILTONIAN
        H_f = -Sigmasz[0] - np.dot(Sigmasz[0], Sigmasz[1]) - np.dot(Sigmasz[0], Sigmasz[2]) - \
            2 * - np.dot(Sigmasz[1], Sigmasz[1]) + 5*I

    elif final_hamiltonian == 'simple Ising PAPER anticross': ## N_qubits = 3
        Sigmasz = [Sigma(0, i, N_qubits) for i in range(N_qubits) ]
        I = np.eye(dims_matrices)

        ## ISING HAMILTONIAN
        H_f = -Sigmasz[0] - np.dot(Sigmasz[0], Sigmasz[1]) - np.dot(Sigmasz[0], Sigmasz[2]) - \
            2 * - np.dot(Sigmasz[1], Sigmasz[1]) + 5*I + \
                np.dot( (Sigmasz[0] + I), (-Sigmasz[3] + I) ) + \
                    np.dot( (Sigmasz[1] + I), (-Sigmasz[4] + I) ) + \
                        np.dot( (Sigmasz[2] + I), (-Sigmasz[5] + I) )

    elif final_hamiltonian == 'simple Ising PAPER now CROSS': ## N_qubits = 3
        Sigmasz = [Sigma(0, i, N_qubits) for i in range(N_qubits) ]
        I = np.eye(dims_matrices)

        ## ISING HAMILTONIAN
        H_f = -Sigmasz[0] - np.dot(Sigmasz[0], Sigmasz[1]) - np.dot(Sigmasz[0], Sigmasz[2]) - \
            2 * - np.dot(Sigmasz[1], Sigmasz[1]) + 5*I + \
                np.dot( (Sigmasz[0] + I), (-Sigmasz[3] + I) ) + \
                    np.dot( (Sigmasz[1] + I), (-Sigmasz[4] + I) ) + \
                        np.dot( (Sigmasz[2] + I), (-Sigmasz[5] + I) ) + \
                            np.dot( (-Sigmasz[0] + I), (-Sigmasz[6] + I) ) + \
                                np.dot( (-Sigmasz[1] + I), (-Sigmasz[7] + I) ) + \
                                    np.dot( (-Sigmasz[2] + I), (-Sigmasz[8] + I) )







    else:
        print('You have not  selected any of the available final Hamiltonians')
        exit()



    ## Initial Hamiltonain --> transverse field / all spins up

    if initial_hamiltonian == 'transverse field':
        phi0 = np.ones(dims_matrices)
        H_i = (np.eye(dims_matrices) - 1/dims_matrices *(np.kron(phi0, phi0).reshape(dims_matrices, dims_matrices)) )

    elif initial_hamiltonian == 'transverse field Dickson': ## transverse field for operational qubits, ancilla pointing up
        rho0 = 0.5 * np.array([[1, 1], [1, 1]])
        rhoanc = np.array([[0, 0], [0, 1]])
        state = np.kron(rho0, rho0)
        for i in range(N_qubits - number_of_ancillas - 2 ):
            state = np.kron(state, rho0)
        for i in range(number_of_ancillas):
            state = np.kron(state, rhoanc)
        H_i = np.eye(dims_matrices) - state

    elif initial_hamiltonian == 'all spins up': ## it does NOT correspond to all spins up
        phi0 = np.zeros(dims_matrices)  ## all spins up
        phi0[0] = np.sqrt(dims_matrices)
        H_i = np.eye(dims_matrices) - 1/dims_matrices *(np.kron(phi0, phi0).reshape(dims_matrices, dims_matrices))

    elif initial_hamiltonian == 'entangled':
        plussup = 1/np.sqrt(2) * np.array([1, 1])
        bell = 1/np.sqrt(2) * (np.kron(np.array([1, 0]), np.array([1, 0])) + np.kron(np.array([0, 1]), np.array([0, 1]) ) )
        if N_qubits == 4:
            full5 = np.kron(bell, bell) 
        elif N_qubits == 5:
            full5 = np.kron(plussup, np.kron(bell, bell) )
        else:
            print('That number of qubits is not contemplated for this initial condition')
            exit()
        H_i = np.eye(dims_matrices) - np.kron(full5, full5).reshape(dims_matrices, dims_matrices)

    elif initial_hamiltonian == 'entangled 2':
        s0 = 1/np.sqrt(N_qubits) * np.zeros(dims_matrices)
        s1 = 1/np.sqrt(N_qubits) * np.ones(dims_matrices)
        full5 = 1/np.sqrt(2) * (s0 + s1)
        H_i = np.eye(dims_matrices) - np.kron(full5, full5).reshape(dims_matrices, dims_matrices)


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

    ## Annealing schedule --> linear / optimised Grover / force Landau 
    ## catalyst --> parabola
    if annealing_schedule == 'linear':
        def A(s):
            return 1-s
        def B(s):
            return s

    elif annealing_schedule == 'optimised Grover':
        def A(s):
            b = np.sqrt(N_qubits - 1)
            a = (2*s - 1) * np.arctan(b)
            return 1 - (0.5 + 0.5 / np.sqrt(N_qubits - 1) * np.tan(a) )
        def B(A):
            return 1 - A

    elif annealing_schedule == 'force Landau':
        def A(s):
            return 1 - 0.5 * (np.tanh(5*(s-0.5)) + 1) ### Forcing Landau levels toghether
        def B(s):
            return 0.5 * (np.tanh(5*(s-0.5)) + 1)
    else:
        print('You have not  selected any of the available anealing schedules')
        exit()


    if catalyst == 'None':
        def C(s): # * (1-s) * np.log(1+s)
            return 0#1 * np.exp(-25*(s-0.5)**2)# s*(1-s) ## a partir de 1/2*25 está bien como desviación ##(1-s)*np.log(1+s)
    elif catalyst == 'parabola':
        def C(s):
            return s*(1-s)
    else:
        print('You have not  selected any of the available catalysts')
        exit()




    ## Catalyst Hamiltonian 
    H_catalyst = np.zeros((dims_matrices, dims_matrices), 'complex128')
    for i in range(N_qubits):
        H_catalyst += 1 * Sigma_dict[i][coord_catalyst] # h[i] *

    if return_HiHf == True:
        return Hamiltonian_factory(H_i, H_f, A, B, H_catalyst, C), H_i, H_f
    else:
        return Hamiltonian_factory(H_i, H_f, A, B, H_catalyst, C)
