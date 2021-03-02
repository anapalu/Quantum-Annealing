import sys
sys.path.append('/home/ana/Documents/PhD/basics_pkg')

sys.path.append('/home/ana/Documents/PhD/basics_pkg/stubs') ### For some reason importing the .pyi files does not fix the alarm 
                                ### raising on the following statements, but since I no longer see them on the main scipts I honestly don't care

from basics_simulate_quantum import define_Hs, np2qt
from basics_measures import spectral_norm, l2_norm,  get_mingap_proper, vN_entropy, Hamming_distance
from basics_pauli import Sigma
from basics_plot import simple_plot







# def main():
#     print('I got in')
#     # from basics_pkg import Sigma, define_Hs, get_mingap_proper, simple_plot, spectral_norm, l2_norm
#     from basics_simulate_quantum import define_Hs
#     from basics_measures import spectral_norm, l2_norm,  get_mingap_proper
#     from basics_pauli import Sigma
#     from basics_plot import simple_plot
#     # from numba_njits import *
#     return define_Hs, get_mingap_proper, spectral_norm, l2_norm, Sigma, simple_plot


# if __name__== "__main__":
#     main()
#     print('I did not get in :(')
