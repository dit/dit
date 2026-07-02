"""
Visualizations of multivariate distributions.

This subpackage collects plotting tools for exploring the structure of joint
distributions. Currently it provides an UpSet-style plot of the atoms of a
distribution's information diagram, which scales to an arbitrary number of
random variables (unlike Venn/Euler diagrams).
"""

from .upset import *
