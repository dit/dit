"""
Joint distributions from Bayesian graphical models, aka Bayesian networks.

Currently, this does not support factorings which include more than one
random variable, such as P(X) P(Y,Z|X). Effectively, we assume each random
variable can be factored so that it is "alone", as in P(X) P(Y|X,Z) P(Z|X)
or P(X) P(Z|X,Y) P(Y|X). While one can always do this in probability, it may
not be the case that this corresponds to how it is causally generated.

An example of this is an edge-emitting hidden Markov model: P(X_0, S_1 | S_0).
There is no causal relationship between X_0 and S_1...they are jointly
generated from S_0.

"""
import six

import dit

__all__ = ['distribution_from_bayesnet']

def sanitize_inputs(digraph, nodes, attr):
    """
    Quick sanity checks on the input.

    """
    all_nodes = set(digraph.nodes())

    def validate_names(node, dist):
        names = dist.get_rv_names()
        if names is None:
            names = set(range(dist.outcome_length()))
        else:
            names = set(names)

        if not names.issubset(all_nodes):
            msg = "Node {} has invalid rv names: {}".format(node, names)
            raise ValueError(msg)

    ops = None
    for rv in digraph:
        # Make sure we have dists for each node.
        try:
            val = digraph.node[rv][attr]
        except KeyError:
            msg = "Node {} is missing its distributions.".format(rv)
            raise ValueError(msg)

        # Make sure the rv names are appropriate.
        if digraph.in_degree(rv) == 0:
            # No parents
            dists = [val]
        else:
            # A distribution for each value of the parents.
            try:
                dists = val.values()
            except AttributeError:
                outcomes, dists = val

        for dist in dists:
            validate_names(rv, dist)

    else:
        # Use the last dist to get the base.
        ops = dist.ops

    # Get a good set of random variable names.
    if nodes is None:
        rv_names = sorted(digraph.nodes())
    else:
        if len(nodes) != len(all_nodes):
            msg = "`nodes` is missing required nodes: {}".format(nodes)
            raise ValueError(msg)
        if set(nodes) != all_nodes:
            msg = "`set(nodes)` does not contain all required nodes."
            raise ValueError(msg)
        rv_names = nodes

    return rv_names, ops


def build_samplespace(digraph, rv_names, attr):
    """
    Builds the sample space for the joint distribution.

    """
    sample_spaces = []
    product = None
    for rv in rv_names:
        if digraph.in_degree(rv) == 0:
            # No parents
            dist = digraph.node[rv][attr]
        else:
            # Grab the first distribution.
            val = digraph.node[rv][attr]
            try:
                dist = next(six.itervalues(val))
            except AttributeError:
                dist = val[1][0]

        # Since we are assuming each random is completely alone and factored.
        # We can just take the alphabet. This will need to change eventually.
        try:
            alphabet = dist._sample_space.alphabets[0]
        except AttributeError:
            alphabet = list(dist._sample_space)
        sample_spaces.append(alphabet)
    else:
        # Use the last dist to get a product function.
        # We'll assume they are all the same.
        product = dist._product

    ss = dit.CartesianProduct(sample_spaces, product=product)
    return ss


def build_pfuncs(digraph, rv_names, attr, outcome_ctor):
    """
    Build probability functions for each rv.

    The function takes a random variable and the joint outcome and returns
    the probability contribution to the joint probability.


    """
    rv_index = dict(zip(rv_names, range(len(rv_names))))
    pfuncs = {}
    for rv in rv_names:
        parents = list(digraph.predecessors(rv))
        parents.sort(key=rv_index.__getitem__)

        if not parents:
            dist = digraph.node[rv][attr]
            # Bind the distribution to dist, immediately.
            # http://docs.python-guide.org/en/latest/writing/gotchas/#late-binding-closures
            def prob(outcome, dist=dist):
                rv_outcome = outcome_ctor([ outcome[rv_index[rv]] ])
                return dist[rv_outcome]

        else:
            val = digraph.node[rv][attr]
            try:
                val.values()
            except AttributeError:
                outcomes, dists = val
                dists = dict(zip(outcomes, dists))
            else:
                dists = val

            def prob(outcome, dist=dist):
                rv_outcome = outcome_ctor([ outcome[rv_index[rv]] ])
                parent_vals = [outcome[rv_index[parent]] for parent in parents]
                parent_outcome = outcome_ctor(parent_vals)
                dist = dists[parent_outcome]
                return dist[rv_outcome]

        pfuncs[rv] = prob

    return pfuncs


def distribution_from_bayesnet(digraph, nodes=None, attr='dist'):
    """
    Returns a distribution built from a Bayesian network.

    Each node represents a random variable ``X_i``. Every node must store its
    conditional probability distribution ``P(X_i | Y_i)`` where ``Y_i``
    represents the parents of ``X_i``. If a node has no in-degree, then it
    must store the probability distribution ``P(X_i)``.


    Parameters
    ----------
    digraph : NetworkX digraph
        A directed graph, representing the Bayesian graphical model.
    nodes : list, None
        The order of the nodes that will determine the random variable order.
        If `None`, then we use `sorted(digraph.nodes())`, which assumes the
        nodes are sortable. The reason we assume they are sortable is because
        the parent values must correspond to the node order and thus, we need
        an unambiguous ordering that the user could have known ahead of time.
    attr : str
        The attribute for each node that holds the conditional distributions.
        The attribute value can take a variety of forms. It can be a list of
        two lists, such as (parents, dists) which holds the parents and the
        conditional distributions: `dists[i] = P(X_i | Y_i = parents[i])`.
        It can also be an dict-like structure `dists[y] = P(X | Y = y)`.
        If the node has no in-degree, then it should store the distribution.
        All distributions should have random variable names assigned that match
        the nodes in the graph, or alternatively, all nodes in the graph should
        be integers. The order of these lists (or dict) does not matter, but
        for nodes that have parents, the order of the random variables that
        specify the parents, must match the order of ``nodes``. So for example,
        if the node order is [2, 1, 0] and node 1 has parents 0 and 2. Then the
        parents for node 1 will be such that the first element corresponds to
        node 2 and the second to node 0, since 2 precedes 0 in the node order.
        All distributions will be assumed to have the same base, and this base
        will determine the base of the constructed distribution.

    Returns
    -------
    dist : Distribution
        The joint distribution.

    """
    rv_names, ops = sanitize_inputs(digraph, nodes, attr)
    sample_space = build_samplespace(digraph, rv_names, attr)
    pfuncs = build_pfuncs(digraph, rv_names, attr, sample_space._outcome_ctor)

    outcomes = list(sample_space)
    pmf = [ops.mult_reduce([pfuncs[rv](outcome) for rv in rv_names])
           for outcome in outcomes]

    dist = dit.Distribution(outcomes, pmf,
                            sample_space=sample_space, base=ops.get_base())
    dist.set_rv_names(rv_names)

    return dist
