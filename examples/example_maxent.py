from __future__ import print_function

import dit
import numpy as np

try:
    # The functions will import this for you...just make sure you have it.
    import cvxopt
except ImportError:
    print("Module cvxopt is required")
    exit()

def print_output(d, maxent_dists):
    # Calculate the entropy for each.
    entropies = np.asarray(map(dit.shannon.entropy, maxent_dists))
    print()
    print("Entropies:")
    print(entropies)

    # Differences are what we learn at each step.
    netinfo = -1 * np.diff(entropies)
    print()
    print("Network Informations:")
    print(netinfo)

    # Total correlation is what is learned at ith from (i-1)th starting at i=2.
    total_corr = netinfo[1:].sum()
    total_corr_true = dit.multivariate.total_correlation(d)
    print()
    print("Total correlation: {0} (numerically)\t {1} (true)".format(total_corr, total_corr_true))
    print()

def example_A():
    """
    Calculate network information using marginal maxentropy.

    """
    d = dit.example_dists.Xor()

    # Calculate marginal maximum entropy distributions up to order 3.
    maxent_dists = dit.algorithms.marginal_maxent_dists(d, 3)

    print_output(d, maxent_dists)

    return maxent_dists

def example_B():
    """
    Calculate network information using moment-based maxentropy.

    """
    d = dit.example_dists.Xor()

    # Calculate moment maximum entropy distributions up to order 3.
    mapping = [-1, 1]
    maxent_dists = dit.algorithms.moment_maxent_dists(d, mapping, 3, with_replacement=False)

    print_output(d, maxent_dists)

    return maxent_dists

def example_C():
    # Giant bit, perfect correlation.

    # Note, doesn't converge if we do this with n=4. e.g.: '1111', '0000'. Lol!
    outcomes = ['111', '000']
    d = dit.Distribution(outcomes, [.5, .5])
    maxent_dists = dit.algorithms.marginal_maxent_dists(d)
    print_output(d, maxent_dists)

if __name__ == '__main__':
    example_A()

