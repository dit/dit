"""
Tests for dit.utils.latexarray.
"""

import pytest

import os

import numpy as np
import dit.utils.latexarray as la

def test_to_latex1():
    x = np.array([1, 2, 3, 10])
    y = la.to_latex(x)
    y_ = '\\newcolumntype{X}{D{.}{.}{2,0}}\n\\begin{array}{*{4}{X}}\n  1 & 2 & 3 & 10\n\\end{array}'
    assert y == y_

def test_to_latex2():
    x = np.array([1, 2, -3, 100])
    y = la.to_latex(x)
    y_ = '\\newcolumntype{X}{D{.}{.}{4,0}}\n\\begin{array}{*{4}{X}}\n  1 & 2 & -3 & 100\n\\end{array}'
    assert y == y_

def test_to_latex3():
    x = x = np.array([0.078, 0.480, 0.413, 0.830, 0.776, 0.102, 0.513, 0.462, 0.335, 0.712])
    y = la.to_latex(x)
    y_ = '\\newcolumntype{X}{D{.}{.}{1,3}}\n\\begin{array}{*{10}{X}}\n  0.078 & 0.480 & 0.413 & 0.830 & 0.776 & 0.102 & 0.513 & 0.462 & 0.335 & 0.712\n\\end{array}'
    assert y == y_

def test_to_latex4():
    x = x = np.array([-0.078, 0.480, 0.413, 0.830, 0.776, 0.102, 0.513, 0.462, 0.335, 0.712])
    y = la.to_latex(x)
    y_ = '\\newcolumntype{X}{D{.}{.}{2,3}}\n\\begin{array}{*{10}{X}}\n  -0.078 & 0.480 & 0.413 & 0.830 & 0.776 & 0.102 & 0.513 & 0.462 & 0.335 & 0.712\n\\end{array}'
    assert y == y_

def test_to_latex5():
    np.random.seed(0)
    x = np.random.rand(12).reshape(4, 3) - 0.5
    y = la.to_latex(x, decimals=2)
    y_ = '\\newcolumntype{X}{D{.}{.}{2,2}}\n\\begin{array}{*{3}{X}}\n  0.05 & 0.22 & 0.10 \\\\\n  0.04 & -0.08 & 0.15 \\\\\n  -0.06 & 0.39 & 0.46 \\\\\n  -0.12 & 0.29 & 0.03\n\\end{array}'
    assert y == y_

def test_to_latex_exact1():
    x = np.array([1, 2, 3, 10, .1])
    y = la.to_latex(x, exact=True)
    y_ = '\\begin{array}{*{5}{c}}\n  1 & 2 & 3 & 10 & \\frac{1}{10}\n\\end{array}'
    assert y == y_

def test_to_latex_exact2():
    x = 0.12345
    y = la.to_latex(x, exact=.01)
    y_ = '\\begin{array}{*{1}{c}}\n  \\frac{1}{8}\n\\end{array}'
    assert y == y_

    y = la.to_latex(x, exact=.001)
    y_ = '\\begin{array}{*{1}{c}}\n  \\frac{7}{57}\n\\end{array}'
    assert y == y_

@pytest.mark.xfail
def test_to_pdf():
    import subprocess
    with open(os.devnull, 'w') as fp:
        error = subprocess.call('pdflatex --help', shell=True, stdout=fp, stderr=fp)
        error |= subprocess.call('pdfcrop --help', shell=True, stdout=fp, stderr=fp)

    x = 0.1
    # This generates a temporary file...
    y = la.to_pdf(x, show=False)
