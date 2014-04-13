from nose.tools import *

from ..samplespace import SampleSpace, CartesianProduct

import dit

class TestSampleSpace(object):
    def setUp(self):
        product = dit.helpers.get_product_func(str)
        self.samplespace = ['00', '01', '10', '12']
        self.ss = SampleSpace(self.samplespace, product)

    def test_samplespace(self):
        assert_equal(list(self.ss), self.samplespace)
        assert_equal(len(self.ss), 4)
        assert_equal(self.ss.outcome_length(), 2)
        assert_true('00' in self.ss)
        assert_false('22' in self.ss)

    def test_marginalize(self):
        ss0 = self.ss.marginalize([1])
        assert_equal(list(ss0), ['0', '1'])

    def test_marginal(self):
        ss1 = self.ss.marginal([1])
        assert_equal(list(ss1), ['0', '1', '2'])

    def test_coalesce(self):
        ss2 = self.ss.coalesce([[0,1,1],[1,0]])
        ss2_ = [('000', '00'), ('011', '10'), ('100', '01'), ('122', '21')]
        assert_equal(list(ss2), ss2_)

class TestCartesianProduct(object):
    def setUp(self):
        product = dit.helpers.get_product_func(str)
        alphabets = [['0', '1'], ['0', '1', '2']]
        self.ss = CartesianProduct(alphabets, product)

    def test_samplespace(self):
        assert_equal(list(self.ss), ['00', '01', '02', '10', '11', '12'])
        assert_equal(len(self.ss), 6)
        assert_equal(self.ss.outcome_length(), 2)
        assert_true('00' in self.ss)
        assert_false('22' in self.ss)

    def test_marginalize(self):
        ss0 = self.ss.marginalize([1])
        assert_equal(list(ss0), ['0', '1'])

    def test_marginal(self):
        ss1 = self.ss.marginal([1])
        assert_equal(list(ss1), ['0', '1', '2'])

    def test_coalesce(self):
        ss2 = self.ss.coalesce([[1,0],[1]])
        ss2_ =  [('00', '0'), ('00', '1'), ('00', '2'),
                 ('01', '0'), ('01', '1'), ('01', '2'),
                 ('10', '0'), ('10', '1'), ('10', '2'),
                 ('11', '0'), ('11', '1'), ('11', '2'),
                 ('20', '0'), ('20', '1'), ('20', '2'),
                 ('21', '0'), ('21', '1'), ('21', '2')]
        assert_equal(list(ss2), ss2_)

