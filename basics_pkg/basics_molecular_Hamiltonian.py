import numpy as np

import openfermion as of
from openfermion.utils import geometry_from_pubchem
import openfermionpyscf as ofpyscf
from openfermion.transforms import get_sparse_operator, get_fermion_operator, bravyi_kitaev #jordan_wigner


# Set molecule parameters
#testing geometries:
## H2 --> [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7414))] --> results in 5 spins
## He --> [('He', (2, 0, 0))] --> results in 2 spins
## LiH --> [('Li', (3, 0, 0)), ('H', (2, 0, 0))] --> results in 12 spins
### any other --> #geometry = geometry_from_pubchem('name_of_molecule')


def get_molecular_Hamiltonian(geometry = [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7414))], basis = 'sto-3g', multiplicity = 1, charge = 0):
    # Perform electronic structure calculations and obtain Hamiltonian as an InteractionOperator
    hamiltonian = ofpyscf.generate_molecular_hamiltonian(geometry, basis, multiplicity, charge)

    # Convert to a FermionOperator
    hamiltonian_ferm_op = of.get_fermion_operator(hamiltonian)

    mol_qubit_hamiltonianBK = bravyi_kitaev(hamiltonian_ferm_op) # JW is exponential in the maximum locality of the original FermionOperator.
    mol_matrix = get_sparse_operator(mol_qubit_hamiltonianBK).todense()
    return mol_matrix
