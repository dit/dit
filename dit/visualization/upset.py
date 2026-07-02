"""
UpSet-style plots of the information diagram of a multivariate distribution.

UpSet plots [Lex2014]_ are a scalable alternative to Venn/Euler diagrams for
visualizing intersections among many sets. Here the "sets" are the random
variables of a joint distribution, and the "intersections" are the atoms of the
Shannon information diagram (I-diagram): each atom is a signed sum of entropies
of the form ``I[A : B : ... | C, D, ...]``, i.e. the information shared among a
subset of the variables *after* conditioning on the rest. There are ``2**n - 1``
such atoms for ``n`` variables, and -- unlike ordinary set cardinalities -- they
may be negative.

.. [Lex2014] Lex, Alexander, et al. "UpSet: visualization of intersecting sets."
   IEEE transactions on visualization and computer graphics 20.12 (2014):
   1983-1992.
"""

from ..profiles.information_partitions import ShannonPartition
from ..shannon import entropy

__all__ = ("InformationUpsetPlot",)


class InformationUpsetPlot:
    """
    An UpSet plot of the atoms of a distribution's information diagram.

    The plot has three coordinated panels:

    - a *matrix* whose rows are random variables and whose columns are the atoms
      of the information diagram; a filled dot means the variable participates in
      that atom (an unconditioned variable), an empty dot means it is conditioned
      upon;
    - an *atom* bar chart (above the matrix) giving each atom's value, colored by
      sign since information atoms may be negative;
    - a *variable* bar chart (beside the matrix) giving each variable's marginal
      entropy ``H[X_i]``.

    Attributes
    ----------
    dist : Distribution
        The distribution being visualized.
    partition : BaseInformationPartition
        The information partition supplying the atoms.
    variables : list
        The random variables, in display order.
    atoms : list of dict
        One entry per atom, each with keys ``members`` (frozenset of the
        participating variables), ``conditions`` (frozenset of the conditioned
        variables), ``degree`` (number of members) and ``value``.
    sizes : dict
        Mapping of variable to its marginal entropy ``H[X_i]``.
    unit : str
        The unit the atom/size values are reported in (e.g. ``"bits"``).
    """

    def __init__(self, dist, *, partition=ShannonPartition):
        """
        Construct an UpSet plot description for `dist`.

        Parameters
        ----------
        dist : Distribution
            The distribution to visualize.
        partition : class
            The information-partition class used to compute the atoms. Defaults
            to :class:`~dit.profiles.information_partitions.ShannonPartition`.
            Any subclass of ``BaseInformationPartition`` (e.g. ``ExtropyPartition``)
            works.
        """
        self.dist = dist
        self.partition = partition(dist)
        self.unit = getattr(self.partition, "unit", "bits")

        names = dist.get_rv_names()
        if names:
            self.variables = list(names)
        else:
            self.variables = list(range(dist.outcome_length()))

        self.atoms = self._build_atoms()
        self.sizes = self._build_sizes()

    def _build_atoms(self):
        """
        Flatten the partition's atoms into per-atom membership records.

        Returns
        -------
        atoms : list of dict
            The atom records, unsorted.
        """
        atoms = []
        for (rvs, crvs), value in self.partition.atoms.items():
            members = frozenset(rv[0] for rv in rvs)
            conditions = frozenset(crvs)
            atoms.append(
                {
                    "members": members,
                    "conditions": conditions,
                    "degree": len(members),
                    "value": value,
                }
            )
        return atoms

    def _build_sizes(self):
        """
        Compute each variable's marginal entropy, used as its "set size".

        Returns
        -------
        sizes : dict
            Mapping of variable to ``H[X_i]``.
        """
        names = self.dist.get_rv_names()
        sizes = {}
        for i, var in enumerate(self.variables):
            rv = var if names else i
            sizes[var] = entropy(self.dist, [rv])
        return sizes

    def _sorted_atoms(self, sort_by="value", min_degree=1):
        """
        Return the atoms filtered by degree and ordered for display.

        Parameters
        ----------
        sort_by : str
            One of ``"value"`` (descending signed value), ``"magnitude"``
            (descending absolute value), or ``"degree"`` (ascending degree, then
            descending value).
        min_degree : int
            Only include atoms whose degree is at least this. Defaults to 1
            (i.e. all atoms, since every atom has at least one member).

        Returns
        -------
        atoms : list of dict
            The filtered, ordered atom records.
        """
        atoms = [a for a in self.atoms if a["degree"] >= min_degree]

        keys = {
            "value": lambda a: -a["value"],
            "magnitude": lambda a: -abs(a["value"]),
            "degree": lambda a: (a["degree"], -a["value"]),
        }
        try:
            key = keys[sort_by]
        except KeyError:
            msg = f"Unknown sort_by={sort_by!r}; choose from {sorted(keys)}."
            raise ValueError(msg) from None

        return sorted(atoms, key=key)

    def draw(
        self,
        ax=None,
        *,
        sort_by="value",
        min_degree=1,
        show_values=True,
        color_positive="C0",
        color_negative="C3",
    ):  # pragma: no cover
        """
        Draw the UpSet plot using matplotlib.

        Parameters
        ----------
        ax : Axis or None
            An existing matplotlib axis whose location is used to host the panels.
            If None, a new figure is created.
        sort_by : str
            How to order the atom columns; see :meth:`_sorted_atoms`.
        min_degree : int
            Only draw atoms of at least this degree.
        show_values : bool
            Annotate each atom bar with its value.
        color_positive : color
            The color for non-negative atoms.
        color_negative : color
            The color for negative atoms.

        Returns
        -------
        axes : dict
            A dictionary with keys ``"atoms"``, ``"matrix"``, and ``"sizes"``
            mapping to the three panel axes.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpecFromSubplotSpec

        atoms = self._sorted_atoms(sort_by=sort_by, min_degree=min_degree)
        variables = self.variables
        n_atoms = len(atoms)
        n_vars = len(variables)

        if ax is None:
            fig = plt.figure(figsize=(max(6, 0.6 * n_atoms + 2), max(4, 0.5 * n_vars + 2)))
            subspec = fig.add_gridspec(1, 1)[0]
        else:
            fig = ax.figure
            subspec = ax.get_subplotspec()
            ax.set_axis_off()

        gs = GridSpecFromSubplotSpec(
            2,
            2,
            subplot_spec=subspec,
            width_ratios=[1, 4],
            height_ratios=[3, 2],
            wspace=0.05,
            hspace=0.05,
        )

        ax_atoms = fig.add_subplot(gs[0, 1])
        ax_matrix = fig.add_subplot(gs[1, 1], sharex=ax_atoms)
        ax_sizes = fig.add_subplot(gs[1, 0], sharey=ax_matrix)

        xs = list(range(n_atoms))

        # ── atom value bar chart ──────────────────────────────────────────
        values = [a["value"] for a in atoms]
        colors = [color_negative if v < 0 else color_positive for v in values]
        ax_atoms.bar(xs, values, color=colors, width=0.6)
        ax_atoms.axhline(0, color="k", linewidth=0.8)
        ax_atoms.set_ylabel(f"information [{self.unit}]")
        ax_atoms.tick_params(axis="x", labelbottom=False, bottom=False)
        ax_atoms.spines[["top", "right"]].set_visible(False)
        if show_values:
            for x, v in zip(xs, values, strict=True):
                offset = 3 if v >= 0 else -3
                va = "bottom" if v >= 0 else "top"
                ax_atoms.annotate(
                    f"{v:.2f}",
                    (x, v),
                    textcoords="offset points",
                    xytext=(0, offset),
                    ha="center",
                    va=va,
                    fontsize=8,
                )

        # ── membership matrix ─────────────────────────────────────────────
        ys = {var: n_vars - 1 - i for i, var in enumerate(variables)}
        # light row striping for readability
        for var in variables:
            y = ys[var]
            if y % 2 == 0:
                ax_matrix.axhspan(y - 0.5, y + 0.5, color="0.95", zorder=0)

        for x, atom in enumerate(atoms):
            members = atom["members"]
            filled_ys = [ys[var] for var in variables if var in members]
            empty_ys = [ys[var] for var in variables if var not in members]
            if empty_ys:
                ax_matrix.scatter([x] * len(empty_ys), empty_ys, s=80, color="0.8", zorder=2)
            if filled_ys:
                ax_matrix.scatter([x] * len(filled_ys), filled_ys, s=80, color="0.15", zorder=3)
                if len(filled_ys) > 1:
                    ax_matrix.plot(
                        [x, x],
                        [min(filled_ys), max(filled_ys)],
                        color="0.15",
                        linewidth=2,
                        zorder=3,
                    )

        ax_matrix.set_ylim(-0.5, n_vars - 0.5)
        ax_matrix.set_xlim(-0.5, n_atoms - 0.5)
        ax_matrix.set_xticks([])
        ax_matrix.set_yticks(range(n_vars))
        ax_matrix.set_yticklabels([str(var) for var in variables][::-1])
        ax_matrix.yaxis.set_ticks_position("right")
        ax_matrix.yaxis.set_label_position("right")
        ax_matrix.tick_params(axis="y", left=False, right=False, pad=4)
        ax_matrix.spines[["top", "right", "bottom", "left"]].set_visible(False)

        # ── variable size bar chart ───────────────────────────────────────
        size_ys = [ys[var] for var in variables]
        size_vals = [self.sizes[var] for var in variables]
        ax_sizes.barh(size_ys, size_vals, color="0.4", height=0.6)
        ax_sizes.invert_xaxis()
        ax_sizes.set_xlabel(f"H [{self.unit}]")
        ax_sizes.tick_params(axis="y", labelleft=False, left=False)
        ax_sizes.spines[["top", "left"]].set_visible(False)

        return {"atoms": ax_atoms, "matrix": ax_matrix, "sizes": ax_sizes}

    def __repr__(self):
        """
        Represent using the underlying partition's table.
        """
        return self.partition.__repr__()

    def __str__(self):
        """
        Render the underlying partition as a table.
        """
        return self.partition.to_string()

    def to_string(self, digits=3):
        """
        Render the underlying partition as a table.

        Parameters
        ----------
        digits : int
            The number of digits to display.
        """
        return self.partition.to_string(digits=digits)
