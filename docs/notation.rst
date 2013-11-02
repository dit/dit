.. notation.rst

********
Notation
********

``dit`` is a scientific tool, and so much of this documentation will contain
mathematical expressions. Here we will describe this notation.

Many distributions are *joint* distribution. In the absence of variable names,
we will index each random variable with a subscript. For example, a distribution
over three variables will be written :math:`X_0X_1X_2`. As a shorthand, we will
also denote those random variables as :math:`X_{0:3}`, meaning start with
:math:`X_0` and go through, but not includeing :math:`X_3` -- just like python
slice notation. If we ever need to describe an infinitely long chain of
variables we will drop the index from the side that is infinite. For example,
:math:`X_{:0}` means all random variables :math:`X_{-\infty} \ldots X_{-1}`.
