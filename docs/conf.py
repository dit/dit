import os

import dit

on_rtd = os.environ.get("READTHEDOCS", None) == "True"

# -- General configuration -----------------------------------------------------

needs_sphinx = "4.0"

primary_domain = "py"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "matplotlib.sphinxext.plot_directive",
]

bibtex_bibfiles = ["references.bib"]

ipython_mplbackend = "agg"
ipython_execlines = [
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    'plt.rcParams["figure.figsize"] = (8, 6)',
    'plt.rcParams["savefig.facecolor"] = (1, 1, 1, 0)',
    "import dit",
]
ipython_savefig_dir = "images/"
ipython_warning_is_error = False

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

project = "dit"
copyright = "2013-2026, dit contributors"  # noqa: A001
version = dit.__version__
release = dit.__version__

exclude_patterns = ["_build"]
add_module_names = False
pygments_style = "sphinx"
modindex_common_prefix = ["dit."]
todo_include_todos = not on_rtd


# -- Math macros (single source of truth for HTML and PDF) ---------------------
#
# Format: name -> [definition, nargs]  or  name -> definition (0 args).
# Used to generate both the MathJax 3 browser config and the LaTeX preamble.

_MACROS = {
    "op": [r"\operatorname{#1}\left[#2\right]", 2],
    # Shannon measures
    "H": [r"\op{H}{#1}", 1],
    "I": [r"\op{I}{#1}", 1],
    "T": [r"\op{T}{#1}", 1],
    "B": [r"\op{B}{#1}", 1],
    "J": [r"\op{J}{#1}", 1],
    "R": [r"\op{R}{#1}", 1],
    "II": [r"\op{II}{#1}", 1],
    "TSE": [r"\op{TSE}{#1}", 1],
    # Common informations
    "K": [r"\op{K}{#1}", 1],
    "C": [r"\op{C}{#1}", 1],
    "G": [r"\op{G}{#1}", 1],
    "F": [r"\op{F}{#1}", 1],
    "M": [r"\op{M}{#1}", 1],
    # Other measures
    "P": [r"\op{P}{#1}", 1],
    "X": [r"\op{X}{#1}", 1],
    "L": [r"\op{L}{#1}", 1],
    "ID": [r"\op{I_D}{#1}", 1],
    "CRE": [r"\op{\mathcal{E}}{#1}", 1],
    "GCRE": [r"\op{\mathcal{E^\prime}}{#1}", 1],
    "RE": [r"\op{H_{#1}}{#2}", 2],
    "TE": [r"\op{S_{#1}}{#2}", 2],
    "xH": [r"\op{xH}{#1}", 1],
    # Divergences
    "DKL": [r"\op{D_{KL}}{#1}", 1],
    "JSD": [r"\op{D_{JS}}{#1}", 1],
    # PID measures
    "Icap": [r"\op{I_{\cap}}{#1}", 1],
    "Ipart": [r"\op{I_{\partial}}{#1}", 1],
    "Imin": [r"\op{I_{min}}{#1}", 1],
    "Immi": [r"\op{I_{MMI}}{#1}", 1],
    "Iwedge": [r"\op{I_{\wedge}}{#1}", 1],
    "Iproj": [r"\op{I_{proj}}{#1}", 1],
    "Ibroja": [r"\op{I_{BROJA}}{#1}", 1],
    "Iccs": [r"\op{I_{ccs}}{#1}", 1],
    "Ipm": [r"\op{I_{\pm}}{#1}", 1],
    "Ida": [r"\op{I_{\downarrow}}{#1}", 1],
    "Idda": [r"\op{I_{\Downarrow}}{#1}", 1],
    "Idep": [r"\op{I_{dep}}{#1}", 1],
    "Irav": [r"\op{I_{RAV}}{#1}", 1],
    "Irr": [r"\op{I_{RR}}{#1}", 1],
    "Ira": [r"\op{I_{RA}}{#1}", 1],
    "Iskara": [r"\op{I_{:}}{#1}", 1],
    "Iskarb": [r"\op{I_{\rightarrow}}{#1}", 1],
    "Iskarc": [r"\op{I_{\leftarrow}}{#1}", 1],
    "Iskard": [r"\op{I_{\leftrightarrow}}{#1}", 1],
    # Entropy decompositions
    "Hpart": [r"\op{H_{\partial}}{#1}", 1],
    "Hcs": [r"\op{H_{cs}}{#1}", 1],
    # Lattice / ordering symbols
    "meet": r"\curlywedge",
    "join": r"\curlyvee",
    "iless": r"\preceq",
    "imore": r"\succeq",
    "ieq": r"\cong",
    "mss": r"\searrow",
    "ind": r"\mathrel{\large\text{$\perp\mkern-10mu\perp$}}",
}


# -- MathJax 3 configuration --------------------------------------------------

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {"tex": {"macros": _MACROS}}


# -- HTML output ---------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
htmlhelp_basename = "ditdoc"


# -- LaTeX output (generate preamble from the same macro dict) -----------------

_LATEX_RENEW = {"H", "P"}


def _macros_to_latex(macros, renew):
    """Convert the shared macro dict into LaTeX \\newcommand lines."""
    lines = []
    for name, defn in macros.items():
        cmd = "renewcommand" if name in renew else "newcommand"
        if isinstance(defn, list):
            tex_def, nargs = defn
            lines.append(rf"\{cmd}{{\{name}}}[{nargs}]{{{tex_def}}}")
        else:
            lines.append(rf"\{cmd}{{\{name}}}{{{defn}}}")
    return "\n".join(lines)


latex_elements = {
    "preamble": "\n".join([
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage{nicefrac}",
        r"\usepackage{scalerel}",
        "",
        _macros_to_latex(_MACROS, _LATEX_RENEW),
        "",
        r"\DeclareMathOperator*{\meetop}{\scalerel*{\meet}{\textstyle\sum}}",
        r"\DeclareMathOperator*{\joinop}{\scalerel*{\join}{\textstyle\sum}}",
    ]),
}

latex_documents = [
    ("index", "dit.tex", "dit Documentation", "dit Contributors", "manual"),
]


# -- Manual page output --------------------------------------------------------

man_pages = [("index", "dit", "dit Documentation", ["dit Contributors"], 1)]


# -- Texinfo output ------------------------------------------------------------

texinfo_documents = [
    (
        "index",
        "dit",
        "dit Documentation",
        "dit Contributors",
        "dit",
        "Discrete Information Theory in Python.",
        "Science",
    ),
]


# -- Intersphinx ---------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
