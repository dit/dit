"""
Joint distributions from a Bayesian graphical model.

"""

def sanitize_inputs(digraph, attr, nodes):
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

    # Get a good set of random variable names.
    if nodes is None:
        rvs_names = list(digraph.nodes())
    else:
        if len(nodes) != all_nodes:
            raise ValueError("`nodes` is missing required nodes")
        if set(nodes) != all_nodes:
            msg = "`set(nodes)` does not contain all required nodes."
            raise ValueError(msg)
        rvs_names = nodes

    return rvs_names

def build_distribution(digraph, attr='dist', nodes=None):
    """
    Returns a distribution built from a Bayesian graphical model.

    Each node represents a random variable ``X_i``. Every node must store its
    conditional probability distribution ``P(X_i | Y_i)`` where ``Y_i``
    represents the parents of ``X_i``. If a node has no in-degree, then it
    must store the probability distribution ``P(X_i)``.


    Parameters
    ----------
    digraph : NetworkX digraph
        A directed graph, representing the Bayesian graphical model.
    attr : str
        The attribute for each node that holds the conditional distributions.
        The attribute value can take a variety of forms. It can be a list of
        two lists, such as (parents, dists) which holds the parents and the
        conditional distributions: `dists[i] = P(X_i | Y_i = parents[i])`.
        It can also be an dict-like structure `dists[y] = P(X | Y = y)`.
        If the node has no in-degree, then it should store the distribution.
        All distributions should have random variable names assigned that match
        the nodes in the graph, or alternatively, all nodes in the graph should
        be integers.
    nodes : list, None
        The order of the nodes. If `None`, then we use `digraph.nodes()`.

    """
    rvs_names = sanitize_inputs(digraph, attr, nodes)


import dit
import networkx as nx

x = nx.DiGraph()
d = dit.example_dists.Xor()
#d.set_rv_names('XYZ')
rv_mode = 'indices'
cdist, dists = d.condition_on([0, 1], rv_mode=rv_mode)
x.add_edge(0, 2)
x.add_edge(1, 2)
x.node[2]['dist'] = (cdist.outcomes, dists)
x.node[0]['dist'] = cdist.marginal([0], rv_mode=rv_mode)
x.node[1]['dist'] = cdist.marginal([1], rv_mode=rv_mode)
build_distribution(x)
