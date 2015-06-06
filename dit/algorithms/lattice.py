# -*- coding: utf-8 -*-
"""
Some algorithms related to lattices.

Here we are concerned with determining random variable induced sigma-algebras.
That is, we want to know the subsigma-algebra (of the underlying sigma-algebra)
that corresponds to a random variable. However, we consider only those random
variables that map outcomes in the underlying sigma-algebra to a projection of
that outcome. For example, if an outcome in the underlying sigma-algebra is
(1,2,3), then random variable `X_1` would take on the value (2,) and the joint
random variable `X_0,X_2` has the value (1,2). By considering all possible
induced sigma-algebras, we are equivalently considering all possible partitions
of the sample space.

Relationship to Intersection Information based on Common Randomness
--------------------------------------------------------------------
In general, the meet and join operators are defined with respect to information
equivalence, which in turn, derives from the notions of informationally
richer/poorer. In this module, richer and poorer correspond to refinements and
coarsenings of the sample space and thus, depend explicitly on the structure
of the underlying sigma-algebra.

As an example, consider the r.v.s. X and Y, where X = 1 and Y = X with p=1 and
Y = 0 with p=0. There is a sense in X and Y have the same distribution. However,
X and Y are not informationally equivalent as there is a function f such that
X = f(Y) where f maps 0 and 1 to 1, but not the other way around. One imagines
X as the coarsest possible partition of the sample space, whereas Y is a
refinement. To wit, the sample space is {10,11} with p(10) = 0 and p(11) = 1.
Then X corresponds to the partition: {(10, 11)} while Y is {(10,), (11,)}.

In [1], richer and poorer are defined in terms of probability almost surely.
That is, X is informationally poorer than Y if X = f(Y) almost surely.  This
means that the definition is not as sensitive to the structure of the
underlying sigma-algebra. In the above example, X and Y are now informationally
equivalent, even though Y is a refinement of the partition corresponding to X.

In general, there is a trend that coarser partitions correspond to
informationally poorer random variables (but due to the partial ordering, not
all partitions are comparable to one another).  In [1], a (comparable) coarser
partition is only poorer if the coarsening involves outcomes with nonzero
probability. Said another way, a r.v. that corresponds to a refinement of
another r.v. is informationally richer only if it refines outcomes which have
nonzero probability.

If the behavior of [1] is desired, one must make sure to prune the sample space
of any outcomes that have zero probability. Then, the implementation here will
give the same results as [1].

[1] "Intersection Information based on Common Randomness"
    http://arxiv.org/abs/1310.1538

"""
from collections import defaultdict

from six.moves import map, range, zip # pylint: disable=redefined-builtin

import dit
from ..helpers import parse_rvs, RV_MODES
from ..math import sigma_algebra, atom_set
from ..utils import quasilexico_key

def sigma_algebra_sort(sigalg):
    """
    Put the sigma algebra in quasi-lexicographical order.
    """
    sigalg = [tuple(sorted(cet)) for cet in sigalg]
    sigalg.sort(key=quasilexico_key)
    return sigalg

def induced_sigalg(dist, rvs, rv_mode=None):
    """
    Returns the induced sigma-algebra of the random variable defined by `rvs`.

    Parameters
    ----------
    dist : Distribution
        The distribution which defines the base sigma-algebra.
    rvs : list
        The indexes of the random variable used to calculate the induced
        sigma algebra.
    rv_mode : str, None
        Specifies how to interpret the elements of `rvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `rvs` are interpreted as random variable indices. If equal to 'names',
        the the elements are interpreted as random variable names. If `None`,
        then the value of `dist._rv_mode` is consulted.

    Returns
    -------
    F : frozenset of frozensets
        The induced sigma-algebra.

    """
    # This is brute force and ugly.
    #
    # Implementation:
    #   1) Find induced atoms from atoms of new sigma-algebra:
    #           X^{-1}(A) = { w : X(w) \in A }
    #       where A = \{a\} and a is a nonzero outcome in the marginal.
    #   2) Generate sigma algebra from induced atoms.
    #
    # Step 2 may not be necessary.
    #
    indexes = parse_rvs(dist, rvs, rv_mode=rv_mode, unique=True, sort=True)[1]

    # This creates a mapping from new outcomes (defined by rvs) to the
    # original outcomes which map to those new outcomes. This defines a
    # partition of the original outcomes.
    d = defaultdict(list)
    ctor = dist._outcome_ctor
    for outcome, _ in dist.zipped(mode='atoms'):
        # Build a list of inner outcomes. "c" stands for "constructed".
        # We need to iterate over all atoms, not just those in pmf since
        # we are trying to partition the sample space.
        c_outcome = ctor([outcome[i] for i in indexes])
        d[c_outcome].append(outcome)

    atoms = frozenset(map(frozenset, d.values()))
    F = sigma_algebra(atoms)
    return F

def join_sigalg(dist, rvs, rv_mode=None):
    """
    Returns the sigma-algebra of the join of random variables defined by `rvs`.

    Parameters
    ----------
    dist : Distribution
        The distribution which defines the base sigma-algebra.
    rvs : list
        A list of lists.  Each list specifies a random variable to be
        joined with the other lists.  Each random variable can defined as a
        series of unique indexes.  Multiple random variables can use the same
        index. For example, [[0,1],[1,2]].
    rv_mode : str, None
        Specifies how to interpret the elements of `rvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `rvs` are interpreted as random variable indices. If equal to 'names',
        the the elements are interpreted as random variable names. If `None`,
        then the value of `dist._rv_mode` is consulted.

    Returns
    -------
    jsa : frozenset of frozensets
        The induced sigma-algebra of the join.

    """
    # We require unique indexes within each random variable and want the
    # indexes in distribution order. We don't need the names.
    parse = lambda rv: parse_rvs(dist, rv, rv_mode=rv_mode,
                                 unique=False, sort=True)[1]
    indexes = [parse(rv) for rv in rvs]

    sigalgs = [induced_sigalg(dist, rv, rv_mode=RV_MODES.INDICES)
               for rv in indexes]

    # \sigma( X join Y ) = \sigma( \sigma(X) \cup \sigma(Y) )
    # Union all the sigma algebras.
    union_sa = frozenset().union(*sigalgs)
    jsa = sigma_algebra(union_sa)
    return jsa

def meet_sigalg(dist, rvs, rv_mode=None):
    """
    Returns the sigma-algebra of the meet of random variables defined by `rvs`.

    Parameters
    ----------
    dist : Distribution
        The distribution which defines the base sigma-algebra.
    rvs : list
        A list of lists.  Each list specifies a random variable to be
        met with the other lists.  Each random variable can defined as a
        series of unique indexes.  Multiple random variables can use the same
        index. For example, [[0,1],[1,2]].
    rv_mode : str, None
        Specifies how to interpret the elements of `rvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `rvs` are interpreted as random variable indices. If equal to 'names',
        the the elements are interpreted as random variable names. If `None`,
        then the value of `dist._rv_mode` is consulted.

    Returns
    -------
    msa : frozenset of frozensets
        The induced sigma-algebra of the meet.

    """
    # We require unique indexes within each random variable and want the
    # indexes in distribution order. We don't need the names.
    parse = lambda rv: parse_rvs(dist, rv, rv_mode=rv_mode,
                                           unique=False, sort=True)[1]
    indexes = [parse(rv) for rv in rvs]

    sigalgs = [induced_sigalg(dist, rv, rv_mode=RV_MODES.INDICES)
               for rv in indexes]

    # \sigma( X meet Y ) = \sigma(X) \cap \sigma(Y) )
    # Intersect all the sigma algebras.
    first_sa = sigalgs[0]
    msa = first_sa.intersection(*sigalgs[1:])
    return msa

def dist_from_induced_sigalg(dist, sigalg, int_outcomes=True):
    """
    Returns the distribution associated with an induced sigma algebra.

    The sigma algebra is induced by a random variable from a probability
    space defined by `dist`.

    Parameters
    ----------
    dist : Distribution
        The distribution which defines the base sigma-algebra.
    sigalg : frozenset
        A sigma-algebra induced by a random variable from `dist`.
    int_outcomes : bool
        If `True`, then the outcomes of the induced distribution are relabeled
        as integers instead of the atoms of the induced sigma-algebra.

    Returns
    -------
    d : ScalarDistribution
        The distribution of the induced sigma algebra.

    """
    from dit import ScalarDistribution

    atoms = atom_set(sigalg)
    if int_outcomes:
        atoms = [sorted(atom) for atom in atoms]
        atoms.sort(key=quasilexico_key)

    pmf = [dist.event_probability(atom) for atom in atoms]
    if int_outcomes:
        outcomes = range(len(atoms))
    else:
        # Outcomes must be sequences.
        outcomes = [tuple(sorted(atom)) for atom in atoms]

    d = ScalarDistribution(outcomes, pmf, base=dist.get_base())
    return d

def join(dist, rvs, rv_mode=None, int_outcomes=True):
    """
    Returns the distribution of the join of random variables defined by `rvs`.

    Parameters
    ----------
    dist : Distribution
        The distribution which defines the base sigma-algebra.
    rvs : list
        A list of lists.  Each list specifies a random variable to be
        joined with the other lists.  Each random variable can defined as a
        series of unique indexes.  Multiple random variables can use the same
        index. For example, [[0,1],[1,2]].
    rv_mode : str, None
        Specifies how to interpret the elements of `rvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `rvs` are interpreted as random variable indices. If equal to 'names',
        the the elements are interpreted as random variable names. If `None`,
        then the value of `dist._rv_mode` is consulted.
    int_outcomes : bool
        If `True`, then the outcomes of the join are relabeled as integers
        instead of as the atoms of the induced sigma-algebra.

    Returns
    -------
    d : ScalarDistribution
        The distribution of the join.

    """
    join_sa = join_sigalg(dist, rvs, rv_mode)
    d = dist_from_induced_sigalg(dist, join_sa, int_outcomes)
    return d

def meet(dist, rvs, rv_mode=None, int_outcomes=True):
    """
    Returns the distribution of the meet of random variables defined by `rvs`.

    Parameters
    ----------
    dist : Distribution
        The distribution which defines the base sigma-algebra.
    rvs : list
        A list of lists.  Each list specifies a random variable to be
        met with the other lists.  Each random variable can defined as a
        series of unique indexes.  Multiple random variables can use the same
        index. For example, [[0,1],[1,2]].
    rv_mode : str, None
        Specifies how to interpret the elements of `rvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `rvs` are interpreted as random variable indices. If equal to 'names',
        the the elements are interpreted as random variable names. If `None`,
        then the value of `dist._rv_mode` is consulted.
    int_outcomes : bool
        If `True`, then the outcomes of the meet are relabeled as integers
        instead of as the atoms of the induced sigma-algebra.

    Returns
    -------
    d : ScalarDistribution
        The distribution of the meet.

    """
    meet_sa = meet_sigalg(dist, rvs, rv_mode)
    d = dist_from_induced_sigalg(dist, meet_sa, int_outcomes)
    return d

def insert_rv(dist, idx, sigalg):
    """
    Returns a new distribution with a random variable inserted at index `idx`.

    The random variable is constructed according to its induced sigma-algebra.

    Parameters
    ----------
    dist : Distribution
        The distribution which defines the base sigma-algebra.
    idx : int
        The index at which to insert the random variable. To append, set `idx`
        to be equal to -1 or dist.outcome_length().
    sigalg : frozenset
        The sigma-algebra induced by the random variable.

    Returns
    -------
    d : Distribution
        The new distribution.

    """
    from itertools import chain

    if idx == -1:
        idx = dist.outcome_length()

    if not 0 <= idx <= dist.outcome_length():
        raise IndexError('Invalid insertion index.')

    # Provide sane sorting of atoms
    atoms = atom_set(sigalg)
    atoms = [sorted(atom) for atom in atoms]
    atoms.sort(key=quasilexico_key)
    labels = range(len(atoms))
    if dist._outcome_class == str:
        # Then the labels for the new random variable must be strings.
        labels = map(str, labels)

    # Create an index from outcomes to atoms.
    atom_of = {}
    for label, atom in zip(labels, atoms):
        for outcome in atom:
            atom_of[outcome] = label

    if idx == dist.outcome_length():
        def new_outcome_ctor(outcome, ctor=dist._outcome_ctor):
            """The end of the outcome"""
            new_outcome = [outcome, [atom_of[outcome]]]
            return ctor(chain.from_iterable(new_outcome))
    elif idx == 0:
        def new_outcome_ctor(outcome, ctor=dist._outcome_ctor):
            """The beginning of the outcome"""
            new_outcome = [[atom_of[outcome]], outcome]
            return ctor(chain.from_iterable(new_outcome))
    else:
        def new_outcome_ctor(outcome, ctor=dist._outcome_ctor):
            """In the middle of the outcome"""
            new_outcome = [outcome[:idx], [atom_of[outcome]], outcome[idx:]]
            return ctor(chain.from_iterable(new_outcome))

    d = dit.modify_outcomes(dist, new_outcome_ctor)
    return d

def insert_join(dist, idx, rvs, rv_mode=None):
    """
    Returns a new distribution with the join inserted at index `idx`.

    The join of the random variables in `rvs` is constructed and then inserted
    into at index `idx`.

    Parameters
    ----------
    dist : Distribution
        The distribution which defines the base sigma-algebra.
    idx : int
        The index at which to insert the join. To append the join, set `idx`
        to be equal to -1 or dist.outcome_length().
    rvs : list
        A list of lists.  Each list specifies a random variable to be
        met with the other lists.  Each random variable can defined as a
        series of unique indexes.  Multiple random variables can use the same
        index. For example, [[0,1],[1,2]].
    rv_mode : str, None
        Specifies how to interpret the elements of `rvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `rvs` are interpreted as random variable indices. If equal to 'names',
        the the elements are interpreted as random variable names. If `None`,
        then the value of `dist._rv_mode` is consulted.

    Returns
    -------
    d : Distribution
        The new distribution with the join at index `idx`.

    """
    jsa = join_sigalg(dist, rvs, rv_mode)
    d = insert_rv(dist, idx, jsa)
    return d

def insert_meet(dist, idx, rvs, rv_mode=None):
    """
    Returns a new distribution with the meet inserted at index `idx`.

    The meet of the random variables in `rvs` is constructed and then inserted
    into at index `idx`.

    Parameters
    ----------
    dist : Distribution
        The distribution which defines the base sigma-algebra.
    idx : int
        The index at which to insert the meet. To append the meet, set `idx`
        to be equal to -1 or dist.outcome_length().
    rvs : list
        A list of lists.  Each list specifies a random variable to be
        met with the other lists.  Each random variable can defined as a
        series of unique indexes.  Multiple random variables can use the same
        index. For example, [[0,1],[1,2]].
    rv_mode : str, None
        Specifies how to interpret the elements of `rvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `rvs` are interpreted as random variable indices. If equal to 'names',
        the the elements are interpreted as random variable names. If `None`,
        then the value of `dist._rv_mode` is consulted.

    Returns
    -------
    d : Distribution
        The new distribution with the meet at index `idx`.

    """
    msa = meet_sigalg(dist, rvs, rv_mode)
    d = insert_rv(dist, idx, msa)
    return d
