.. channelorder.rst
.. py:module:: dit.channelorder

**************
Channel Order
**************

:mod:`dit.channelorder` compares discrete memoryless channels. The preorders
are Blackwell (output-degraded), input-degraded, less noisy, more capable,
and Shannon inclusion :cite:`Cover2006`. Le Cam and KL deficiencies
quantify how far a pair of channels is from comparability.

These comparisons are the layer used by the channel-order PIDs
:class:`~dit.pid.PID_Deg`, :class:`~dit.pid.PID_MC`, and
:class:`~dit.pid.PID_Prec` (see :doc:`measures/pid`).

A noiseless identity channel Blackwell-dominates a binary symmetric
channel of the same alphabet, but not conversely:

.. ipython::

   In [1]: import numpy as np

   In [2]: from dit.channelorder import is_output_degraded, is_less_noisy

   In [3]: identity = np.eye(2)

   In [4]: bsc = np.array([[0.9, 0.1], [0.1, 0.9]])

   @doctest
   In [5]: is_output_degraded(identity, bsc)
   Out[5]: True

   @doctest
   In [6]: is_output_degraded(bsc, identity)
   Out[6]: False

   @doctest
   In [7]: is_less_noisy(identity, bsc)
   Out[7]: True

Arguments may be channel matrices (rows = inputs) or any format accepted
by the module's channel helpers, including lists of conditional
:class:`~dit.Distribution` objects.

Preorders
=========

.. autofunction:: is_output_degraded

.. autofunction:: is_blackwell_sufficient

.. autofunction:: is_input_degraded

.. autofunction:: is_less_noisy

.. autofunction:: is_more_capable

.. autofunction:: is_shannon_included

.. autofunction:: blackwell_order_joint

Deficiencies
============

Le Cam deficiency is zero if and only if the first channel
output-degrades to the second. KL variants weight the gap by an input
distribution.

.. autofunction:: dit.channelorder.le_cam_deficiency

.. autofunction:: dit.channelorder.le_cam_distance

.. autofunction:: dit.channelorder.weighted_le_cam_deficiency

.. autofunction:: dit.channelorder.output_kl_deficiency

.. autofunction:: dit.channelorder.weighted_output_kl_deficiency

.. autofunction:: dit.channelorder.weighted_output_kl_deficiency_joint

.. autofunction:: dit.channelorder.weighted_input_kl_deficiency
