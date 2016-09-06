
import pytest

from dit.samplespace import SampleSpace, CartesianProduct

import dit

class TestSampleSpace(object):
    def setup_class(self):
        product = dit.helpers.get_product_func(str)
        self.samplespace = ['00', '01', '10', '12']
        self.ss = SampleSpace(self.samplespace, product)

    def test_samplespace_auto(self):
        samplespace = ['00', '01', '10', '12']
        ss = SampleSpace(samplespace)
        assert list(ss) == list(self.ss)

    def test_samplespace(self):
        assert list(self.ss) == self.samplespace
        assert len(self.ss) == 4
        assert self.ss.outcome_length() == 2
        assert '00' in self.ss
        assert not '22' in self.ss

    def test_marginalize(self):
        ss0 = self.ss.marginalize([1])
        assert list(ss0) == ['0', '1']

    def test_marginal(self):
        ss1 = self.ss.marginal([1])
        assert list(ss1) == ['0', '1', '2']

    def test_coalesce(self):
        ss2 = self.ss.coalesce([[0,1,1],[1,0]])
        ss2_ = [('000', '00'), ('011', '10'), ('100', '01'), ('122', '21')]
        assert list(ss2) == ss2_

    def test_sort(self):
        product = dit.helpers.get_product_func(str)
        samplespace = ['00', '01', '12', '10']
        ss = SampleSpace(samplespace, product)
        assert list(ss) == samplespace
        indexes = [ss.index(i) for i in samplespace]
        assert indexes == list(range(len(samplespace)))

        ss.sort()
        assert list(ss) == sorted(samplespace)
        indexes = [ss.index(i) for i in samplespace]
        assert indexes == [0, 1, 3, 2]

class TestCartesianProduct(object):
    def setup_class(self):
        product = dit.helpers.get_product_func(str)
        alphabets = [['0', '1'], ['0', '1', '2']]
        self.ss = CartesianProduct(alphabets, product)

    def test_stralphabets(self):
        # Test that str product is inferred.
        x = CartesianProduct([['0', '1']]*2)
        assert list(x) == ['00', '01', '10', '11']

    def test_samplespace(self):
        assert list(self.ss) == ['00', '01', '02', '10', '11', '12']
        assert len(self.ss) == 6
        assert self.ss.outcome_length() == 2
        assert '00' in self.ss
        assert not '22' in self.ss

    def test_marginalize(self):
        ss0 = self.ss.marginalize([1])
        assert list(ss0) == ['0', '1']

    def test_marginal(self):
        ss1 = self.ss.marginal([1])
        assert list(ss1) == ['0', '1', '2']

    def test_coalesce(self):
        ss2 = self.ss.coalesce([[1,0],[1]])
        ss2_ =  [('00', '0'), ('00', '1'), ('00', '2'),
                 ('01', '0'), ('01', '1'), ('01', '2'),
                 ('10', '0'), ('10', '1'), ('10', '2'),
                 ('11', '0'), ('11', '1'), ('11', '2'),
                 ('20', '0'), ('20', '1'), ('20', '2'),
                 ('21', '0'), ('21', '1'), ('21', '2')]
        assert list(ss2) == ss2_

    def test_sort(self):
        alphabets = [[0,1], [2,1]]
        ss = CartesianProduct(alphabets)
        indexes = [ss.index(i) for i in list(ss)]
        assert indexes == list(range(len(ss)))

        samplespace = list(ss)
        ss.sort()
        indexes = [ss.index(i) for i in samplespace]
        assert indexes == [1, 0, 3, 2]

    def test_from_outcomes(self):
        outcomes = ['00', '11']
        ss = CartesianProduct.from_outcomes(outcomes)
        assert list(ss) == ['00', '01', '10', '11']
        with pytest.raises(ValueError):
            ss.index('22')

def test_nested():
    outcomes = ['11', '00']
    ss = CartesianProduct.from_outcomes(outcomes)
    ss2 = CartesianProduct([[0], ss])
    assert list(ss2) == [(0,'11'), (0,'10'), (0,'01'), (0,'00')]
    ss2.sort()
    assert list(ss2) == [(0,'00'), (0,'01'), (0,'10'), (0,'11')]
