"""
Functions to convert pmfs to latex arrays.
"""

import contextlib
import os
import subprocess
import tempfile

from debtcollector import removals

import numpy as np
import numpy.core.arrayprint as arrayprint

from .context import cd, named_tempfile, tempdir
from .misc import default_opener


__all__ = (
    'printoptions',
    'to_latex',
    'to_pdf',
)


# http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
#
# Includes hack to prevent NumPy from removing trailing zeros.
#
@removals.remove(message="Please use np.core.arrayprint.printoptions",
                 version='1.2.3')
@contextlib.contextmanager
def printoptions(strip_zeros=True, **kwargs):
    if strip_zeros:
        kwargs['trim'] = 'k'
    origcall = arrayprint.FloatingFormat.__call__

    def __call__(self, x):
        return origcall.__call__(self, x)

    arrayprint.FloatingFormat.__call__ = __call__
    original = np.get_printoptions()
    np.set_printoptions(**kwargs)
    yield
    np.set_printoptions(**original)
    arrayprint.FloatingFormat.__call__ = origcall


def to_latex__numerical(a, decimals, tab):
    # The elements of each column are aligned on the decimal, if present.
    # Spacing is automatically adjusted depending on if `a` contains negative
    # numbers or not. For float data, trailing zeros are included so that
    # array output is uniform.

    array = np.atleast_2d(a)
    array = np.around(array, decimals)

    # Determine the number of digits left of the decimal.
    # Grab integral part, convert to string, and get maximum length.
    # This does not handle negative signs appropriately since -0.5 goes to 0.
    # So we make sure it never counts a negative sign by taking the abs().
    integral = np.abs(np.trunc(array.flat).astype(int))
    left_digits = max(map(len, map(str, integral)))

    # Adjust for negative signs.
    if np.any(array < 0):
        left_digits += 1

    # Set decimal digits to 0 if data are not floats.
    try:
        np.finfo(array.dtype)
    except ValueError:
        decimals = 0

    # Align the elements on the decimal, making room for largest element.
    coltype = r"\newcolumntype{{X}}{{D{{.}}{{.}}{{{0},{1}}}}}"
    coltype = coltype.format(left_digits, decimals)

    # Specify that we want all columns to have the same column type.
    nCols = array.shape[1]
    cols = r"*{{{nCols}}}{{X}}".format(nCols=nCols)

    # Build the lines in the array.
    #
    # In general, we could just use the rounded array and map(str, row),
    # but NumPy strips trailing zeros on floats (undesirably). So we make
    # use of the context manager to prevent that.
    options = {
        'precision': decimals,
        'suppress': True,
        'strip_zeros': False,
        'threshold': nCols + 1,
    }
    with printoptions(**options):
        lines = []
        for row in array:
            # Strip [ and ], remove newlines, and split on whitespace
            elements = row.__str__()[1:-1].replace('\n', '').split()
            # hack to fix trailing zeros, really the numpy stuff needs to be updated.
            try:
                elements = [element + '0' * (decimals - len(element.split('.')[1])) for element in elements]
            except:
                pass
            line = [tab, ' & '.join(elements), r' \\']
            lines.append(''.join(line))

    # Remove the \\ on the last line.
    lines[-1] = lines[-1][:-3]

    # Create the LaTeX code
    subs = {'coltype': coltype, 'cols': cols, 'lines': '\n'.join(lines)}
    template = r"""{coltype}
\begin{{array}}{{{cols}}}
{lines}
\end{{array}}"""

    return template.format(**subs)


def to_latex__exact(a, tol, tab):

    from dit.math import approximate_fraction

    array = np.atleast_2d(a)

    to_frac = lambda f: approximate_fraction(f, tol)
    fractions = np.array(list(map(to_frac, array.flat))).reshape(array.shape)

    # Specify that we want all columns to have the same column type.
    nCols = array.shape[1]
    cols = r"*{{{nCols}}}{{c}}".format(nCols=nCols)

    def to_frac(f):
        if f.denominator != 1:
            return r'\frac{{{}}}{{{}}}'.format(f.numerator, f.denominator)
        else:
            return str(f.numerator)

    # Build the lines in the array.
    lines = []
    for row in fractions:
        # Strip [ and ], remove newlines, and split on whitespace
        elements = map(to_frac, row)
        line = [tab, ' & '.join(elements), r' \\']
        lines.append(''.join(line))

    # Remove the \\ on the last line.
    lines[-1] = lines[-1][:-3]

    # Create the LaTeX code
    subs = {'cols': cols, 'lines': '\n'.join(lines)}
    template = r"""\begin{{array}}{{{cols}}}
{lines}
\end{{array}}"""

    return template.format(**subs)


def to_latex(a, exact=False, decimals=3, tab='  '):
    r"""
    Convert an array-like object into a LaTeX array.

    Parameters
    ----------
    a : array-like
        A list, tuple, NumPy array, etc. The elements are written into a
        LaTeX array environment.

    exact : bool
        When `exact` is False, the elements of each column are aligned on the
        decimal, if present. Spacing is automatically adjusted depending on if
        `a` contains negative numbers or not. For float data, trailing zeros
        are included so that array output is uniform.

        When `exact` is `True`, then each float is turned into a fraction and
        placed in a column. One may specify `exact` as a small float to be used
        as the tolerance while converting to a fraction.

    decimals : int
        The number of decimal places to round to before writing to LaTeX.

    tab : str
        The tab character to use for indentation within LaTeX.

    Examples
    --------
    >>> x = [1,2,-3]
    >>> print to_latex(x)
    \newcolumntype{X}{D{.}{.}{2,0}}
    \begin{array}{*{3}{X}}
      1 & 2 & -3
    \end{array}

    >>> import numpy as np
    >>> np.random.seed(0)
    >>> y = np.random.rand(12).reshape(4,3) - 0.5
    >>> print to_latex(y, decimals=2)
    \newcolumntype{X}{D{.}{.}{2,2}}
    \begin{array}{*{3}{X}}
      0.05 & 0.22 & 0.10 \\
      0.04 & -0.08 & 0.15 \\
      -0.06 & 0.39 & 0.46 \\
      -0.12 & 0.29 & 0.03
    \end{array}

    Notes
    -----
    The resultant array environment should be used within LaTeX's mathmode,
    and the following should appear in the preamble:

        \usepackage{array}
        \usepackage{dcolumn}

    """
    if exact:
        if exact is True:
            tol = 1e-6
        else:
            tol = exact
        return to_latex__exact(a, tol, tab)
    else:
        return to_latex__numerical(a, decimals, tab)


def to_pdf(a, exact=False,
              decimals=3,
              line="p(x) = \\left[\n{0}\n\\right]",
              show=True):
    """
    Converts a NumPy array to a LaTeX array, compiles and displays it.

    Examples
    --------
    >>> a = np.array([[.1, .23, -1.523],[2.1, .23, .523]])
    >>> to_pdf(a) # pragma: no cover

    """
    template = r"""\documentclass{{article}}
\usepackage{{amsmath}}
\usepackage{{array}}
\usepackage{{dcolumn}}
\pagestyle{{empty}}
\begin{{document}}
\begin{{displaymath}}
{0}
\end{{displaymath}}
\end{{document}}"""

    fline = line.format(to_latex(a, exact=exact, decimals=decimals))
    latex = template.format(fline)

    with contextlib.ExitStack() as stack:  # pragma: no cover
        EC = stack.enter_context
        tmpdir = EC(tempdir())
        EC(cd(tmpdir))
        latexfobj = EC(named_tempfile(dir=tmpdir, suffix='.tex'))

        # Write the latex file
        latexfobj.write(latex.encode('utf8'))
        latexfobj.close()

        # Compile to PDF
        args = ['pdflatex', '-interaction=batchmode', latexfobj.name]
        with open(os.devnull, 'w') as fp:
            subprocess.call(args, stdout=fp, stderr=fp)  # noqa: S603
            subprocess.call(args, stdout=fp, stderr=fp)  # noqa: S603

        # Create another tempfile which will not be deleted.
        pdffobj = tempfile.NamedTemporaryFile(suffix='_pmf.pdf', delete=False)
        pdffobj.close()

        # Crop the PDF and copy to persistent tempfile.
        pdfpath = latexfobj.name[:-3] + 'pdf'
        # Cannot add &>/dev/null to cmd, as Ghostscript is unable to find the
        # input file. This seems to be some weird interaction between
        # subprocess and pdfcrop. Also, we need to use shell=True since some
        # versions of pdfcrop rely on a hack to determine what perl interpreter
        # to call it with.
        cmd = r'pdfcrop --debug {} {}'.format(pdfpath, pdffobj.name)
        with open(os.devnull, 'w') as fp:
            subprocess.call(cmd, shell=True, stdout=fp)

        # Open the PDF
        if show:
            default_opener(pdffobj.name)

        return pdffobj.name
