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
-------------------------------------------------------------------
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

import dit

from ..helpers import parse_rvs
from ..math import atom_set, sigma_algebra
from ..utils import quasilexico_key

__all__ = (
    "dist_from_induced_sigalg",
    "induced_sigalg",
    "insert_join",
    "insert_meet",
    "insert_rv",
    "join",
    "join_sigalg",
    "meet",
    "meet_sigalg",
    "sigma_algebra_sort",
)


def sigma_algebra_sort(sigalg):
    """
    Put the sigma algebra in quasi-lexicographical order.
    """
    sigalg = [tuple(sorted(cet)) for cet in sigalg]
    sigalg.sort(key=quasilexico_key)
    return sigalg


def induced_sigalg(dist, rvs, support_only=False):
    """
    Returns the induced sigma-algebra of the random variable defined by `rvs`.

    Parameters
    ----------
    dist : Distribution
        The distribution which defines the base sigma-algebra.
    rvs : list
        The indexes of the random variable used to calculate the induced
        sigma algebra.
    support_only : bool
        If True, only partition outcomes in the support (non-zero probability).
        This is needed for algorithms like GK common information where the
        sample space should be restricted to the support.

    Returns
    -------
    F : frozenset of frozensets
        The induced sigma-algebra.

    """
    indexes = parse_rvs(dist, rvs, unique=True, sort=True)[1]

    d = defaultdict(list)
    ctor = dist._outcome_ctor
    if support_only:
        iterator = ((o, p) for o, p in dist.zipped(mode="atoms")
                    if not dist.ops.is_null_exact(p))
    else:
        iterator = dist.zipped(mode="atoms")
    for outcome, _ in iterator:
        c_outcome = ctor([outcome[i] for i in indexes])
        d[c_outcome].append(outcome)

    atoms = frozenset(map(frozenset, d.values()))
    F = sigma_algebra(atoms)
    return F


def join_sigalg(dist, rvs, support_only=False):
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
        index. For example, [[0, 1], [1, 2]].
    support_only : bool
        If True, only consider outcomes with non-zero probability.

    Returns
    -------
    jsa : frozenset of frozensets
        The induced sigma-algebra of the join.

    """
    parse = lambda rv: parse_rvs(dist, rv, unique=False, sort=True)[1]
    indexes = [parse(rv) for rv in rvs]

    sigalgs = [induced_sigalg(dist, rv, support_only=support_only) for rv in indexes]

    # \sigma( X join Y ) = \sigma( \sigma(X) \cup \sigma(Y) )
    union_sa = frozenset().union(*sigalgs)
    jsa = sigma_algebra(union_sa)
    return jsa


def meet_sigalg(dist, rvs, support_only=False):
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
    support_only : bool
        If True, only consider outcomes with non-zero probability.

    Returns
    -------
    msa : frozenset of frozensets
        The induced sigma-algebra of the meet.

    """
    parse = lambda rv: parse_rvs(dist, rv, unique=False, sort=True)[1]
    indexes = [parse(rv) for rv in rvs]

    sigalgs = [induced_sigalg(dist, rv, support_only=support_only) for rv in indexes]

    # \sigma( X meet Y ) = \sigma(X) \cap \sigma(Y) )
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
    d : Distribution
        The distribution of the induced sigma algebra.

    """
    from dit.distribution import Distribution

    atoms = atom_set(sigalg)
    if int_outcomes:
        atoms = [sorted(atom) for atom in atoms]
        atoms.sort(key=quasilexico_key)

    pmf = [dist.event_probability(atom) for atom in atoms]
    outcomes = range(len(atoms)) if int_outcomes else [tuple(sorted(atom)) for atom in atoms]

    d = Distribution(outcomes, pmf, base=dist.get_base())
    return d


def join(dist, rvs, int_outcomes=True):
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
        index. For example, [[0, 1], [1, 2]].
    int_outcomes : bool
        If `True`, then the outcomes of the join are relabeled as integers
        instead of as the atoms of the induced sigma-algebra.

    Returns
    -------
    d : Distribution
        The distribution of the join.

    """
    join_sa = join_sigalg(dist, rvs)
    d = dist_from_induced_sigalg(dist, join_sa, int_outcomes)
    return d


def meet(dist, rvs, int_outcomes=True):
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
        index. For example, [[0, 1], [1, 2]].
    int_outcomes : bool
        If `True`, then the outcomes of the meet are relabeled as integers
        instead of as the atoms of the induced sigma-algebra.

    Returns
    -------
    d : Distribution
        The distribution of the meet.

    """
    meet_sa = meet_sigalg(dist, rvs)
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
        raise IndexError("Invalid insertion index.")

    # Provide sane sorting of atoms
    atoms = atom_set(sigalg)
    atoms = [sorted(atom) for atom in atoms]
    atoms.sort(key=quasilexico_key)
    if dist._outcome_class is str:
        # Then the labels for the new random variable must be strings.
        from string import ascii_letters, digits

        labels = (digits + ascii_letters)[: len(atoms)]
    else:
        labels = range(len(atoms))

    # Create an index from outcomes to atoms.
    atom_of = {}
    for label, atom in zip(labels, atoms, strict=True):
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


def insert_join(dist, idx, rvs, support_only=False):
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
        index. For example, [[0, 1], [1, 2]].
    support_only : bool
        If True, only consider outcomes with non-zero probability.

    Returns
    -------
    d : Distribution
        The new distribution with the join at index `idx`.

    """
    jsa = join_sigalg(dist, rvs, support_only=support_only)
    d = insert_rv(dist, idx, jsa)
    return d


def insert_meet(dist, idx, rvs, support_only=False):
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
    support_only : bool
        If True, only consider outcomes with non-zero probability.

    Returns
    -------
    d : Distribution
        The new distribution with the meet at index `idx`.

    """
    msa = meet_sigalg(dist, rvs, support_only=support_only)
    d = insert_rv(dist, idx, msa)
    return d
