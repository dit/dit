from nose.tools import *

import dit
import networkx as nx

def test_distribution_from_bayesnet_nonames():
    """
    Smoke test without rv names.

    """
    x = nx.DiGraph()
    d = dit.example_dists.Xor()
    cdist, dists = d.condition_on([0, 1])
    x.add_edge(0, 2)
    x.add_edge(1, 2)
    x.node[2]['dist'] = (cdist.outcomes, dists)
    x.node[0]['dist'] = cdist.marginal([0])
    x.node[1]['dist'] = cdist.marginal([1])
    d2 = dit.distribution_from_bayesnet(x)
    d3 = dit.distribution_from_bayesnet(x, [0, 1, 2])
    assert_true( d.is_approx_equal(d2) )
    assert_true( d.is_approx_equal(d3) )

    # Use a dictionary too
    x.node[2]['dist'] = dict(zip(cdist.outcomes, dists))
    d4 = dit.distribution_from_bayesnet(x)
    assert_true( d.is_approx_equal(d4) )

def test_distribution_from_bayesnet_names():
    """
    Smoke test with rv names.

    """
    x = nx.DiGraph()
    d = dit.example_dists.Xor()
    d.set_rv_names(['A', 'B', 'C'])
    cdist, dists = d.condition_on(['A', 'B'])
    x.add_edge('A', 'C')
    x.add_edge('B', 'C')
    x.node['C']['dist'] = (cdist.outcomes, dists)
    x.node['A']['dist'] = cdist.marginal(['A'])
    x.node['B']['dist'] = cdist.marginal(['B'])
    d2 = dit.distribution_from_bayesnet(x)
    assert_true( d.is_approx_equal(d2) )

    # Specify names
    d3 = dit.distribution_from_bayesnet(x, ['A', 'B', 'C'])
    assert_true( d.is_approx_equal(d2) )

    # Test with a non-Cartesian product distribution
    dd = x.node['B']['dist']
    dd._sample_space = dit.SampleSpace(list(dd.sample_space()))
    d3 = dit.distribution_from_bayesnet(x)
    assert_true( d.is_approx_equal(d3) )


def test_bad_names1():
    x = nx.DiGraph()
    d = dit.example_dists.Xor()
    cdist, dists = d.condition_on([0, 1])
    x.add_edge(0, 2)
    x.add_edge(1, 2)
    x.node[2]['dist'] = (cdist.outcomes, dists)
    x.node[0]['dist'] = cdist.marginal([0])
    x.node[1]['dist'] = cdist.marginal([1])
    assert_raises(ValueError, dit.distribution_from_bayesnet, x, [0, 1])
    assert_raises(ValueError, dit.distribution_from_bayesnet, x, [0, 1, 1])
    assert_raises(ValueError, dit.distribution_from_bayesnet, x, [0, 1, 2, 3])
    assert_raises(ValueError, dit.distribution_from_bayesnet, x, [0, 1, 2, 2])

def test_bad_names2():
    """
    Now the distributions have bad names.

    """
    x = nx.DiGraph()
    d = dit.example_dists.Xor()
    cdist, dists = d.condition_on([0, 1])
    x.add_edge(0, 2)
    x.add_edge(1, 2)
    x.node[2]['dist'] = (cdist.outcomes, dists)
    x.node[0]['dist'] = cdist.marginal([0])
    dd = cdist.marginal([1])
    dd.set_rv_names(['A'])
    x.node[1]['dist'] = dd
    assert_raises(ValueError, dit.distribution_from_bayesnet, x)
    del x.node[1]['dist']
    assert_raises(ValueError, dit.distribution_from_bayesnet, x)
