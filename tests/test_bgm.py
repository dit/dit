"""
Tests for dit.bgm.
"""

import pytest

import numpy as np
import dit
import networkx as nx


def test_distribution_from_bayesnet_nonames():
    # Smoke test without rv names.
    x = nx.DiGraph()
    d = dit.example_dists.Xor()
    cdist, dists = d.condition_on([0, 1])
    x.add_edge(0, 2)
    x.add_edge(1, 2)
    x.nodes[2]['dist'] = (cdist.outcomes, dists)
    x.nodes[0]['dist'] = cdist.marginal([0])
    x.nodes[1]['dist'] = cdist.marginal([1])
    d2 = dit.distribution_from_bayesnet(x)
    d3 = dit.distribution_from_bayesnet(x, [0, 1, 2])
    assert d.is_approx_equal(d2)
    assert d.is_approx_equal(d3)

    # Use a dictionary too
    x.nodes[2]['dist'] = dict(zip(cdist.outcomes, dists))
    d4 = dit.distribution_from_bayesnet(x)
    assert d.is_approx_equal(d4)

    del x.nodes[1]['dist']
    with pytest.raises(ValueError, match="Node 1 is missing its distributions."):
        dit.distribution_from_bayesnet(x)


def test_distribution_from_bayesnet_names():
    # Smoke test with rv names.
    x = nx.DiGraph()
    d = dit.example_dists.Xor()
    d.set_rv_names(['A', 'B', 'C'])
    cdist, dists = d.condition_on(['A', 'B'])
    x.add_edge('A', 'C')
    x.add_edge('B', 'C')
    x.nodes['C']['dist'] = (cdist.outcomes, dists)
    x.nodes['A']['dist'] = cdist.marginal(['A'])
    x.nodes['B']['dist'] = cdist.marginal(['B'])
    d2 = dit.distribution_from_bayesnet(x)
    assert d.is_approx_equal(d2)

    # Specify names
    d3 = dit.distribution_from_bayesnet(x, ['A', 'B', 'C'])
    assert d.is_approx_equal(d2)

    # Test with a non-Cartesian product distribution
    dd = x.nodes['B']['dist']
    dd._sample_space = dit.SampleSpace(list(dd.sample_space()))
    d3 = dit.distribution_from_bayesnet(x)
    assert d.is_approx_equal(d3)


def test_distribution_from_bayesnet_func():
    # Smoke test with distributions as functions.
    x = nx.DiGraph()
    x.add_edge('A', 'C')
    x.add_edge('B', 'C')

    d = dit.example_dists.Xor()
    sample_space = d._sample_space

    def uniform(node_val, parents):
        return 0.5

    def xor(node_val, parents):
        if parents['A'] != parents['B']:
            output = '1'
        else:
            output = '0'

        # If output agrees with passed in output, p = 1
        p = int(output == node_val)

        return p

    x.nodes['C']['dist'] = xor
    x.nodes['A']['dist'] = uniform
    x.nodes['B']['dist'] = uniform
    # Samplespace is required when functions are callable.
    with pytest.raises(ValueError, match="sample_space must be specified since the distributions were callable."):
        dit.distribution_from_bayesnet(x)

    d2 = dit.distribution_from_bayesnet(x, sample_space=sample_space)
    assert d.is_approx_equal(d2)

    ss = ['000', '001', '010', '011', '100', '101', '110', '111']
    d3 = dit.distribution_from_bayesnet(x, sample_space=ss)
    # Can't test using is_approx_equal, since one has SampleSpace and the
    # other has CartesianProduct as sample spaces. So is_approx_equal would
    # always be False.
    assert np.allclose(d.pmf, d3.pmf)
    assert list(d.sample_space()) == list(d3.sample_space())


def test_distribution_from_bayesnet_error():
    # Test distribution_from_bayesnet with functions and distributions.
    # This is not allowed and should fail.

    x = nx.DiGraph()
    x.add_edge('A', 'C')
    x.add_edge('B', 'C')

    d = dit.example_dists.Xor()
    sample_space = d._sample_space

    def uniform(node_val, parents):
        return 0.5

    unif = dit.Distribution('01', [.5, .5])
    unif.set_rv_names('A')

    x.nodes['C']['dist'] = uniform
    x.nodes['A']['dist'] = unif
    x.nodes['B']['dist'] = uniform

    with pytest.raises(Exception, match="All distributions must be callable if any are."):
        dit.distribution_from_bayesnet(x, sample_space=sample_space)


def test_bad_names1():
    x = nx.DiGraph()
    d = dit.example_dists.Xor()
    cdist, dists = d.condition_on([0, 1])
    x.add_edge(0, 2)
    x.add_edge(1, 2)
    x.nodes[2]['dist'] = (cdist.outcomes, dists)
    x.nodes[0]['dist'] = cdist.marginal([0])
    x.nodes[1]['dist'] = cdist.marginal([1])
    with pytest.raises(ValueError, match="`nodes` is missing required nodes:"):
        dit.distribution_from_bayesnet(x, [0, 1])
    with pytest.raises(ValueError, match=r"`set\(nodes\)` does not contain all required nodes."):  # noqa: W605
        dit.distribution_from_bayesnet(x, [0, 1, 1])
    with pytest.raises(ValueError, match="`nodes` is missing required nodes:"):
        dit.distribution_from_bayesnet(x, [0, 1, 2, 3])
    with pytest.raises(ValueError, match="`nodes` is missing required nodes:"):
        dit.distribution_from_bayesnet(x, [0, 1, 2, 2])


def test_bad_names2():
    """
    Now the distributions have bad names.

    """
    x = nx.DiGraph()
    d = dit.example_dists.Xor()
    cdist, dists = d.condition_on([0, 1])
    x.add_edge(0, 2)
    x.add_edge(1, 2)

    # Node 2 should have more than one dist. If we pass just a distribution in,
    # as if it had no parents, then an exception should raise.

    # x.nodes[2]['dist'] = (cdist.outcomes, dists)
    x.nodes[2]['dist'] = cdist.marginal([0])
    x.nodes[0]['dist'] = cdist.marginal([0])
    x.nodes[1]['dist'] = cdist.marginal([1])
    with pytest.raises(Exception, match="Node 2 has an invalid dist specification."):
        dit.distribution_from_bayesnet(x)

    # If they don't have the same length, it's invalid too.
    x.nodes[2]['dist'] = (cdist.outcomes, dists[:0])
    with pytest.raises(Exception, match="Node 2 has an invalid dist specification."):
        dit.distribution_from_bayesnet(x)
