"""
"""

from __future__ import division

from abc import ABCMeta, abstractmethod

from operator import itemgetter

import numpy as np

from dit import Distribution, ditParams, insert_rvf, modify_outcomes
from dit.algorithms import channel_capacity
from dit.exceptions import ditException
from dit.helpers import flatten, normalize_rvs
from dit.multivariate import caekl_mutual_information
from dit.shannon import entropy_pmf as h
from dit.utils import partitions
from dit.utils.optimization import BasinHoppingCallBack, Uniquifier, accept_test

class BaseIntrinsicMutualInformation(object):
    """

    """
    __metaclass__ = ABCMeta

    def __init__(self, dist, rvs=None, crvs=None, rv_mode=None):
        """
        Parameters
        ----------
        dist : Distribution
        rvs : list of lists
        crvs : list
        rv_mode : str or None
        """
        self._dist = dist.copy()
        self._alphabet = self._dist.alphabet
        self._dist = modify_outcomes(self._dist, lambda x: tuple(x))
        self._rvs, self._crvs, self._rv_mode = normalize_rvs(self._dist, rvs, crvs, rv_mode)
        if not self._crvs:
            msg = "Intrinsic mutual informations require a conditional variable."
            raise ditException(msg)

        self._unq = Uniquifier()
        self._dist = insert_rvf(self._dist, lambda x: (self._unq(tuple(x[i] for i in self._crvs)),))

        sizes = list(map(len, self._dist.alphabet))
        self._dist.make_dense()
        self._pmf = self._dist.pmf.reshape(sizes)

        self._crv = self._dist.outcome_length() - 1
        self._crv_size = sizes[-1]

        all_vars = set(range(len(sizes)))
        keepers = set(flatten(self._rvs)) | {self._crv+1}
        self._others = tuple(all_vars - keepers)

    def construct_random_initial(self):
        """
        Construct a random optimization vector.

        Returns
        -------
        x : ndarray
            A random optimization vector.
        """
        n = self._crv_size
        x = np.random.random((n, n))
        return x

    def construct_joint(self, x):
        """
        Construct the joint distribution.

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        joint : float
            The joint distribution resulting from the distribution passed
            in and the optimization vector.
        """
        n = self._crv_size
        channel = x.reshape((n, n))
        channel = channel / channel.sum(axis=1, keepdims=True)
        channel[np.isnan(channel)] = 0
        slc = (len(self._pmf.shape) - 1)*[np.newaxis] + 2*[slice(None, None)]
        joint = self._pmf[..., np.newaxis] * channel[slc]

        return joint

    def channel_capacity(self, x):
        """
        Compute the channel capacity of the mapping z -> z_bar.

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        cc : float
            The channel capacity.

        Todo
        ----
        write this to not use dit.
        """
        cc = channel_capacity(x.reshape(2*(self._crv_size,)).copy())[0]
        return cc

    @abstractmethod
    def objective(self, x): # pragma: no cover
        """
        The multivariate mutual information to minimize.
        """
        pass

    def optimize(self, x0=None, nhops=None, callback=False):
        """
        Perform the optimization.

        Parameters
        ----------
        x0 : ndarray, None
            Initial optimization vector. If None, use a random vector.
        nhops : int, None
            The number of basin hops to perform while optimizing. If None,
            hop a number of times equal to the dimension of the conditioning
            variable(s).
        callback : bool
            If true, collect each basin in a callback.

        Returns
        -------
        callback : BasinHoppingCallBack
            If `callback` is True, then the callback is returned.
        """
        from scipy.optimize import basinhopping

        if x0 is not None:
            x = x0
        else:
            x = self.construct_random_initial()

        if nhops is None:
            nhops = self._crv_size

        minimizer_kwargs = {'method': 'L-BFGS-B',
                            'bounds': [(0, 1)]*x.size,
                           }

        if callback:
            cb = BasinHoppingCallBack([])
        else:
            cb = None

        res = basinhopping(func=self.objective,
                           x0=x,
                           minimizer_kwargs=minimizer_kwargs,
                           niter=nhops,
                           accept_test=accept_test,
                           callback=cb,
                          )

        self._optima = res.x

        return cb

    def _post_process(self, nhops=10, style='entropy', minmax='min', callback=False):
        """
        Find a solution to the minimization with a secondary property.

        Paramters
        ---------
        nhops : int
        style : 'entropy', 'channel'
        minmax : 'min', 'max'
        callback : bool
        """
        from scipy.optimize import basinhopping

        sign = +1 if minmax == 'min' else -1

        if style == 'channel':
            def objective(x):
                return sign*self.channel_capacity(x)
        elif style == 'entropy':
            def objective(x):
                var = self.construct_joint(x).sum(axis=tuple(range(self._crv+1)))
                return sign*h(var)

        true_objective = self.objective(self._optima)

        def constraint_match_objective(x):
            obj = (self.objective(x) - true_objective)**2
            return obj

        def constraint_normalized(x):
            channel = x.reshape(self._crv_size, self._crv_size)
            return (channel.sum(axis=1) - 1).sum()**2

        minimizer_kwargs = {'method': 'SLSQP',
                            'bounds': [(0, 1)]*self._optima.size,
                            'constraints': [{'type': 'eq',
                                             'fun': constraint_match_objective,
                                            },
                                            {'type': 'eq',
                                             'fun': constraint_normalized,
                                            },
                                           ],
                           }

        cb = BasinHoppingCallBack(minimizer_kwargs['constraints'])
        cb(self._optima, true_objective, True)

        res = basinhopping(func=objective,
                           x0=self._optima.copy(),
                           minimizer_kwargs=minimizer_kwargs,
                           niter=nhops,
                           callback=cb,
                           accept_test=accept_test,
                          )

        minimum = cb.minimum()
        if minimum is not None:
            self._optima = minimum

        if callback:
            return cb

    def construct_distribution(self, x=None, cutoff=1e-5):
        """
        Construct the distribution.

        Parameters
        ----------
        x : ndarray
            An optimization vector.
        cutoff : float
            Ignore probabilities smaller than this.

        Returns
        -------
        d : Distribution
            The original distribution, plus `z_bar'.
        """
        if x is None:
            x = self._optima

        alphabets = list(self._alphabet)

        try:
            # if all outcomes are strings, make new variable strings too.
            ''.join(flatten(alphabets))
            alphabets += [self._unq.chars]
            string = True
        except:
            string = False

        joint = self.construct_joint(x)
        joint = joint.sum(axis=self._crv)
        outcomes, pmf = zip(*[(o, p) for o, p in np.ndenumerate(joint) if p > cutoff])
        outcomes = [ tuple(a[i] for i, a in zip(o, alphabets)) for o in outcomes ]

        if string:
            outcomes = [ ''.join(o) for o in outcomes ]

        pmf = np.asarray(pmf)
        pmf /= pmf.sum()

        d = Distribution(outcomes, pmf)
        return d


class IntrinisicTotalCorrelation(BaseIntrinsicMutualInformation):
    """
    """

    def objective(self, x):
        """
        The total correlation, also known as the binding information.

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        tc : float
            The total correlation.
        """
        rvs = self._rvs
        crv = [self._crv+1]
        joint = self.construct_joint(x)
        joint = joint.sum(axis=self._others, keepdims=True)
        margs = [ joint.sum(axis=tuple(flatten(rvs[:i]+rvs[i+1:]))) for i, _ in enumerate(rvs) ]
        crv = joint.sum(axis=tuple(flatten(rvs)))

        a = sum(h(marg.ravel()) for marg in margs)
        b = h(joint.ravel())
        c = h(crv.ravel())

        tc = a - b - (len(rvs) - 1)*c

        return tc


class IntrinisicDualTotalCorrelation(BaseIntrinsicMutualInformation):
    """
    """

    def objective(self, x):
        """
        The dual total correlation, also known as the binding information.

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        dtc : float
            The dual total correlation.
        """
        rvs = self._rvs
        crv = [self._crv+1]
        joint = self.construct_joint(x)
        joint = joint.sum(axis=self._others, keepdims=True)
        margs = [ joint.sum(axis=tuple(rv)) for rv in rvs ]
        crv = joint.sum(axis=tuple(flatten(rvs)))

        a = sum(h(marg.ravel()) for marg in margs)
        b = h(joint.ravel())
        c = h(crv.ravel())

        dtc = a - (len(rvs) - 1)*b - c

        return dtc


class IntrinisicCAEKLMutualInformation(BaseIntrinsicMutualInformation):
    """
    """

    def objective(self, x):
        """
        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        caekl : float
            The CAEKL mutual information.

        Todo
        ----
        rewrite to not use dit.
        """
        rvs = self._rvs
        crv = [self._crv+1]
        joint = self.construct_joint(x)
        joint = joint.sum(axis=self._others, keepdims=True)
        crv = joint.sum(axis=tuple(flatten(rvs)))

        H = h(joint.ravel())

        def I_P(part):
            margs = [ joint.sum() ]

        parts = [p for p in partitions(map(tuple, rvs)) if len(p) > 1]
        for p in parts:
            print(p)

        margs = [ joint.sum(axis=tuple(flatten(rvs[:i]+rvs[i+1:]))) for i, _ in enumerate(rvs) ]


        #return caekl


def intrinsic_mutual_information(func):
    """
    Given a measure of shared information, construct an optimizer which computes
    its ``intrinsic'' form.

    Parameters
    ----------
    func : function
        A function which computes the information shared by a set of variables.
        It must accept the arguments `rvs' and `crvs'.
    """
    class IntrinsicMutualInformation(BaseIntrinsicMutualInformation):
        """
        """
        def objective(self, x):
            """
            """
            d = self.construct_distribution(x)
            mi = func(d, rvs=self._rvs, crvs=[self._crv])
            return mi

    return IntrinsicMutualInformation
