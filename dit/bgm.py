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

import numpy as np
import dit

__all__ = ['distribution_from_bayesnet']

def sanitize_inputs(digraph, nodes, attr):
    """
    Quick sanity checks on the input.

    """
    all_nodes = set(digraph.nodes())
    ops = dit.math.get_ops('linear')
    is_callable = []
    for rv in digraph:
        # Make sure we have dists for each node.

        try:
            val = digraph.node[rv][attr]
        except KeyError:
            msg = "Node {} is missing its distributions.".format(rv)
            raise ValueError(msg)

        if callable(val):
            is_callable.append(1)
            continue
        else:
            is_callable.append(0)

        # Make sure the rv names are appropriate.
        if digraph.in_degree(rv) == 0:
            # No parents
            dists = [val]
        else:
            # A distribution for each value of the parents.

            # This helps find mistakes more easily!
            # We need [parents, dists] rather than a single distribution.
            if isinstance(val, dit.Distribution):
                msg = 'Node {} has an invalid dist specification.'
                raise Exception(msg.format(rv))

            if isinstance(val, dict):
                dists = val.values()
            else:
                outcomes, dists = val
                if len(outcomes) != len(dists):
                    msg = 'Node {} has an invalid dist specification.'
                    raise Exception(msg.format(rv))

        # No worries if this gets overwritten with each rv, as it had better
        # be the same for each rv.
        ops = next(iter(dists)).ops

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

    n_callable = sum(is_callable)
    if 0 < n_callable < len(rv_names):
        # Then some distributions were callable while others were not.
        msg = "All distributions must be callable if any are."
        raise Exception(msg)

    all_callable = bool(n_callable)

    return rv_names, ops, all_callable


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

        # Since we are assuming each rv is completely alone and factored, we
        # can just take the alphabet. This will need to change eventually.
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
    parents_index = {}

    for rv in rv_names:
        parents = list(digraph.predecessors(rv))
        parents.sort(key=rv_index.__getitem__)
        parents_index[rv] = parents

        val = digraph.node[rv][attr]
        if callable(val):
            pfuncs[rv] = val
            continue

        if not parents:
            # Immediately bind variables since we are in a for loop.
            # http://docs.python-guide.org/en/latest/writing/gotchas/#late-binding-closures
            def prob(outcome, dist=val, rv=rv):
                rv_outcome = outcome_ctor([outcome[rv_index[rv]]])
                return dist[rv_outcome]

        else:
            if isinstance(val, dict):
                val.values()
                dists = val
            else:
                outcomes, dists = val
                dists = dict(zip(outcomes, dists))

            def prob(outcome, dists=dists, parents=parents, rv=rv):
                node_outcome = outcome_ctor([outcome[rv_index[rv]]])
                parent_vals = [outcome[rv_index[parent]] for parent in parents]
                parent_outcome = outcome_ctor(parent_vals)
                dist = dists[parent_outcome]
                return dist[node_outcome]

        pfuncs[rv] = prob

    # Create a function for callable dists that returns the node value and
    # the parents via a dict.
    def get_values(rv, outcome):
        node_val = outcome[rv_index[rv]]
        parents = parents_index[rv]
        parent_vals = [outcome[rv_index[parent]] for parent in parents]
        parents = dict(zip(parents, parent_vals))
        return node_val, parents

    return pfuncs, get_values


def distribution_from_bayesnet(digraph, nodes=None, sample_space=None, attr='dist'):
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
    sample_space : SampleSpace
        If provided, this specifies the outcomes of the distribution to be
        constructed. The distributions stored on the nodes are assumed to be
        compatible with this space. If functions are stored on the nodes, then
        this parameter must be provided.
    attr : str
        The attribute for each node that holds the conditional distributions.
        The attribute value can take a variety of forms.

        It can be a function. The function must take two arguments. The first
        is the value of random variable for the current node. The second is
        a dictionary of keyed by parents whose values are the values of the
        random variables corresponding to the parents. The function should
        return the probability P(node_val|parent_vals).

        It can be a list, such as [parents, dists], that holds the parents and
        the conditional distributions: `dists[i] = P(X | Y_i = parents[i])`.
        It can also be a dict-like structure so that `dists[y]` is a
        distribution representing P(X | Y = y)`. If the node has no in-degree,
        then the attribute value should store the distribution only. When using
        distributions, each should have random variable names assigned that
        match the nodes in the graph, or alternatively, all nodes in the graph
        should be integers and then random variable names are not necessary.
        The order of elements within these lists (or the dict) does not matter,
        but for nodes that have parents, the order of the random variables that
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

    Examples
    --------
    The Xor logic gate has the following structure:

        >>> g = nx.DiGraph()
        >>> g.add_edge(0, 2)
        >>> g.add_edge(1, 2)

    Let's add distributions to it using functions.

        >>> uniform = lambda node_val, parents: 0.5
        >>> def xor(node_val, parents):
        ...     if '1' == parents[0] == parents[1]:
        ...         desired_output = '1'
        ...     else:
        ...         desired_output = '0'
        ...     return int(node_val == desired_output)
        ...
        >>> g.node[0]['dist'] = uniform
        >>> g.node[1]['dist'] = uniform
        >>> g.node[2]['dist'] = xor
        >>> ss = ['000', '001', '010', '011', '100', '101', '110', '111']
        >>> d = dit.distribution_from_bayesnet(g, sample_space=ss)

    Alternatively, we could add distributions using Distribution objects.

        >>> uniform = dit.uniform_distribution(1, 2)
        >>> sample_space1 = [(0,), (1,)]
        >>> one = dit.Distribution(sample_space1, [0, 1])
        >>> zero = dit.Distribution(sample_space1, [1, 0])
        >>> sample_space2 = [(0, 0), (0, 1), (1, 0), (1, 1)]
        >>> xor = [ sample_space2, [zero, one, one, zero]]
        >>> g.node[0]['dist'] = uniform
        >>> g.node[1]['dist'] = uniform
        >>> g.node[2]['dist'] = xor
        >>> d = dit.distribution_from_bayesnet(g)

    We can add noise whenever the output would normally be 1.

        >>> noisy = dit.Distribution(sample_space, [.1, .9])
        >>> dists = [zero, noisy, noisy, zero]
        >>> g.node[2]['dist'][1] = dists
        >>> d = dit.distribution_from_bayesnet(g)

    """
    rv_names, ops, callables = sanitize_inputs(digraph, nodes, attr)
    if callables:
        if sample_space is None:
            msg = 'sample_space must be specified since the '
            msg += 'distributions were callable.'
            raise ValueError(msg)
        if not isinstance(sample_space, dit.SampleSpace):
            sample_space = dit.SampleSpace(sample_space)
    else:
        sample_space = build_samplespace(digraph, rv_names, attr)

    ctor = sample_space._outcome_ctor
    pfuncs, get_values = build_pfuncs(digraph, rv_names, attr, ctor)

    outcomes = list(sample_space)
    mult = ops.mult_reduce
    if callables:
        pmf = [mult(np.asarray([pfuncs[rv](*get_values(rv, outcome)) for rv in rv_names]))
               for outcome in outcomes]
    else:
        pmf = [mult(np.asarray([pfuncs[rv](outcome) for rv in rv_names]))
               for outcome in outcomes]

    # Technically, we shouldn't need this but some values must be underflowing.
    pmf = ops.normalize(np.asarray(pmf))

    dist = dit.Distribution(outcomes, pmf,
                            sample_space=sample_space, base=ops.get_base())
    dist.set_rv_names(rv_names)

    return dist
