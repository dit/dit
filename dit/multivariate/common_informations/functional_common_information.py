"""
The functional common information.
"""

import heapq
from itertools import combinations

import numpy as np

from ...distconst import RVFunctions, insert_rvf
from ...helpers import normalize_rvs
from ...utils import partitions, unitful
from ..dual_total_correlation import dual_total_correlation
from ..entropy import entropy
from ._functional_partition import (
    conditional_dtc,
    labels_from_partition,
    partition_entropy,
    partition_from_joint_mss,
    partition_from_meet,
    prepare_functional_search,
    refinements_by_binary_split,
)

__all__ = ("functional_common_information",)

_MAX_PURE_REFINE_SUPPORT = 8


def functional_markov_chain_naive(dist, rvs=None, crvs=None):  # pragma: no cover
    """
    Add the smallest function of `dist` which renders `rvs` independent.

    Parameters
    ----------
    dist : Distribution
        The distribution for which the smallest function will be constructed.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the total correlation. If None, then the
        total correlation is calculated over all random variables, which is
        equivalent to passing `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.

    Returns
    -------
    d : Distribution
        The distribution `dist` with the additional variable added to the end.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    outcomes = dist.outcomes
    bf = RVFunctions(dist)
    f = [len(dist.rvs)]
    parts = partitions(outcomes)
    dists = [insert_rvf(dist, bf.from_partition(part)) for part in parts]
    B = lambda d: dual_total_correlation(d, rvs, crvs + f)
    dists = [d for d in dists if np.isclose(B(d), 0)]
    return min(dists, key=lambda d: entropy(d, rvs=f))


def _partition_metrics(ctx, part, pmf_size):
    """Return (H(W), B(rvs | crvs, W)) for an outcome partition."""
    labels = labels_from_partition(part, ctx.outcome_to_flat, pmf_size)
    h = partition_entropy(ctx.pmf, labels)
    b = conditional_dtc(ctx.pmf, labels, ctx.rvs, ctx.crvs)
    return h, b


def _coarsen_neighbors(part):
    """Partitions obtained by merging two blocks."""
    return [
        frozenset([p for p in part if p not in pair] + [pair[0] | pair[1]]) for pair in combinations(part, 2)
    ]


def _probe_seeds(ctx, dist, rvs, crvs, pmf_size, *, _use_mss_warmstart=True):
    """
    Compute meet / MSS (or finest) partition seeds and their H, B metrics.
    """
    finest_part = frozenset(frozenset([o]) for o in ctx.dist.outcomes)
    h_finest, b_finest = _partition_metrics(ctx, finest_part, pmf_size)

    meet_part = None
    h_meet = b_meet = None
    try:
        meet_part = partition_from_meet(dist, rvs=rvs, crvs=crvs)
        h_meet, b_meet = _partition_metrics(ctx, meet_part, pmf_size)
    except Exception:  # pragma: no cover
        pass

    mss_part = None
    h_mss = b_mss = None
    mss_valid = False
    if _use_mss_warmstart:
        try:
            mss_part = partition_from_joint_mss(dist, rvs=rvs)
            h_mss, b_mss = _partition_metrics(ctx, mss_part, pmf_size)
            mss_valid = np.isclose(b_mss, 0)
        except Exception:  # pragma: no cover
            pass

    if mss_valid:
        upper_part = mss_part
        h_upper = h_mss
        b_upper = b_mss
    else:
        upper_part = finest_part
        h_upper = h_finest
        b_upper = b_finest

    return {
        "finest_part": finest_part,
        "meet_part": meet_part,
        "h_meet": h_meet,
        "b_meet": b_meet,
        "mss_part": mss_part,
        "h_mss": h_mss,
        "b_mss": b_mss,
        "mss_valid": mss_valid,
        "upper_part": upper_part,
        "h_upper": h_upper,
        "b_upper": b_upper,
    }


def _route_auto_strategy(probe, optimal_b, n_outcomes):
    """
    Pick coarsen, refine, or bidirectional from cheap seed probes.
    """
    h_meet = probe["h_meet"]
    b_meet = probe["b_meet"]
    h_upper = probe["h_upper"]

    if (
        probe["meet_part"] is not None
        and b_meet is not None
        and np.isclose(b_meet, 0)
        and h_meet is not None
        and np.isclose(h_meet, optimal_b)
    ):
        return "refine"

    if probe["mss_valid"] and np.isclose(h_upper, optimal_b):
        return "coarsen"

    if probe["meet_part"] is None:
        return "coarsen"

    n_meet = len(probe["meet_part"])
    n_upper = len(probe["upper_part"])

    if n_meet == 1 and n_upper >= 4:
        return "coarsen"

    if n_upper <= n_meet + 1:
        return "refine"

    if n_outcomes > _MAX_PURE_REFINE_SUPPORT:
        return "bidirectional"

    gap_h = h_upper - h_meet if h_meet is not None else h_upper
    if gap_h > 1.5 * max(b_meet or 0.0, 1e-12):
        return "coarsen"

    if n_meet >= n_upper // 2:
        return "refine"

    return "bidirectional"


def _search_coarsen(
    ctx,
    dist,
    rvs,
    optimal_b,
    pmf_size,
    *,
    _use_mss_warmstart=True,
    probe=None,
):
    """Best-first coarsening from the finest (or MSS) partition."""
    if probe is None:
        probe = _probe_seeds(ctx, dist, rvs, None, pmf_size, _use_mss_warmstart=_use_mss_warmstart)

    finest_part = probe["finest_part"]
    finest_labels = labels_from_partition(finest_part, ctx.outcome_to_flat, pmf_size)
    used_mss_warmstart = False

    if probe["mss_valid"]:
        mss_part = probe["mss_part"]
        h_mss = probe["h_mss"]
        optimal_h = h_mss
        heap: list[tuple[float, int, frozenset]] = [(h_mss, 0, mss_part)]
        used_mss_warmstart = True
        if np.isclose(h_mss, optimal_b):
            return optimal_h, {
                "visited": 0,
                "mss_warmstart": True,
                "meet_warmstart": False,
                "direction": "coarsen",
            }
    else:
        optimal_h = partition_entropy(ctx.pmf, finest_labels)
        heap = [(optimal_h, 0, finest_part)]

    checked: set[frozenset] = set()
    seq = 1

    while heap:  # pragma: no branch
        _, _, part = heapq.heappop(heap)

        if part in checked:
            continue
        checked.add(part)

        labels = labels_from_partition(part, ctx.outcome_to_flat, pmf_size)

        if not np.isclose(conditional_dtc(ctx.pmf, labels, ctx.rvs, ctx.crvs), 0):
            continue

        h = partition_entropy(ctx.pmf, labels)

        if h <= optimal_h:
            optimal_h = h

        if np.isclose(h, optimal_b):
            break

        for new_part in _coarsen_neighbors(part):
            if new_part in checked:
                continue
            new_labels = labels_from_partition(new_part, ctx.outcome_to_flat, pmf_size)
            new_h = partition_entropy(ctx.pmf, new_labels)
            heapq.heappush(heap, (new_h, seq, new_part))
            seq += 1

    return optimal_h, {
        "visited": len(checked),
        "mss_warmstart": used_mss_warmstart,
        "meet_warmstart": False,
        "direction": "coarsen",
    }


def _search_refine(ctx, dist, rvs, crvs, optimal_b, pmf_size, *, probe=None, _max_visits=None):
    """Best-first refinement from the Gács–Körner meet partition."""
    if probe is None:
        probe = _probe_seeds(ctx, dist, rvs, crvs, pmf_size)

    if probe["meet_part"] is None:
        return None, {
            "visited": None,
            "mss_warmstart": False,
            "meet_warmstart": False,
            "direction": "refine",
        }

    meet_part = probe["meet_part"]
    h_meet = probe["h_meet"]
    b_meet = probe["b_meet"]

    if np.isclose(b_meet, 0) and np.isclose(h_meet, optimal_b):
        return h_meet, {
            "visited": 0,
            "mss_warmstart": False,
            "meet_warmstart": True,
            "direction": "refine",
        }

    optimal_h = float("inf")
    heap: list[tuple[float, int, frozenset]] = [(h_meet, 0, meet_part)]
    checked: set[frozenset] = set()
    seq = 1

    while heap:  # pragma: no branch
        _, _, part = heapq.heappop(heap)

        if part in checked:
            continue
        checked.add(part)

        if _max_visits is not None and len(checked) > _max_visits:
            if np.isfinite(optimal_h):
                break
            return None, {
                "visited": len(checked),
                "mss_warmstart": False,
                "meet_warmstart": True,
                "direction": "refine",
            }

        labels = labels_from_partition(part, ctx.outcome_to_flat, pmf_size)

        if np.isclose(conditional_dtc(ctx.pmf, labels, ctx.rvs, ctx.crvs), 0):
            h = partition_entropy(ctx.pmf, labels)
            if h <= optimal_h:
                optimal_h = h
            if np.isclose(h, optimal_b):
                break

        for new_part in refinements_by_binary_split(part):
            if new_part in checked:
                continue
            new_labels = labels_from_partition(new_part, ctx.outcome_to_flat, pmf_size)
            new_h = partition_entropy(ctx.pmf, new_labels)
            heapq.heappush(heap, (new_h, seq, new_part))
            seq += 1

    if not np.isfinite(optimal_h):
        return None, {
            "visited": len(checked),
            "mss_warmstart": False,
            "meet_warmstart": True,
            "direction": "refine",
        }

    return optimal_h, {
        "visited": len(checked),
        "mss_warmstart": False,
        "meet_warmstart": True,
        "direction": "refine",
    }


def _search_bidirectional(ctx, dist, rvs, crvs, optimal_b, pmf_size, *, probe=None):
    """
    Meet-in-the-middle search: refine from GK meet, coarsen from MSS/finest.

    Both frontiers share one ``checked`` set so partitions discovered from one
    side prune work on the other.
    """
    if probe is None:
        probe = _probe_seeds(ctx, dist, rvs, crvs, pmf_size)

    if probe["meet_part"] is None:
        return _search_coarsen(ctx, dist, rvs, optimal_b, pmf_size, probe=probe)

    meet_part = probe["meet_part"]
    upper_part = probe["upper_part"]
    h_meet = probe["h_meet"]
    b_meet = probe["b_meet"]
    h_upper = probe["h_upper"]
    used_mss = bool(probe["mss_valid"])
    used_meet = True

    if np.isclose(b_meet, 0) and np.isclose(h_meet, optimal_b):
        return h_meet, {
            "visited": 0,
            "mss_warmstart": used_mss,
            "meet_warmstart": used_meet,
            "direction": "bidirectional",
            "met_in_middle": 0,
        }

    if used_mss and np.isclose(h_upper, optimal_b):
        return h_upper, {
            "visited": 0,
            "mss_warmstart": used_mss,
            "meet_warmstart": used_meet,
            "direction": "bidirectional",
            "met_in_middle": 0,
        }

    optimal_h = float("inf")
    checked: set[frozenset] = set()
    met_in_middle = 0
    seq = 0
    heap_refine: list[tuple[float, int, frozenset]] = [(h_meet, seq, meet_part)]
    seq += 1
    heap_coarsen: list[tuple[float, int, frozenset]] = [(h_upper, seq, upper_part)]
    seq += 1

    while heap_refine or heap_coarsen:  # pragma: no branch
        h_ref = heap_refine[0][0] if heap_refine else float("inf")
        h_coa = heap_coarsen[0][0] if heap_coarsen else float("inf")

        if h_ref <= h_coa:
            h, _, part = heapq.heappop(heap_refine)
        else:
            h, _, part = heapq.heappop(heap_coarsen)

        if part in checked:
            met_in_middle += 1
            continue
        checked.add(part)

        labels = labels_from_partition(part, ctx.outcome_to_flat, pmf_size)
        b = conditional_dtc(ctx.pmf, labels, ctx.rvs, ctx.crvs)
        b_zero = np.isclose(b, 0)

        if b_zero:
            if h <= optimal_h:
                optimal_h = h
            if np.isclose(h, optimal_b):
                break

        for new_part in refinements_by_binary_split(part):
            if new_part in checked:
                continue
            new_h = partition_entropy(
                ctx.pmf,
                labels_from_partition(new_part, ctx.outcome_to_flat, pmf_size),
            )
            heapq.heappush(heap_refine, (new_h, seq, new_part))
            seq += 1

        if b_zero:
            for new_part in _coarsen_neighbors(part):
                if new_part in checked:
                    continue
                new_h = partition_entropy(
                    ctx.pmf,
                    labels_from_partition(new_part, ctx.outcome_to_flat, pmf_size),
                )
                heapq.heappush(heap_coarsen, (new_h, seq, new_part))
                seq += 1

    if not np.isfinite(optimal_h):
        return _search_coarsen(ctx, dist, rvs, optimal_b, pmf_size, probe=probe)

    return optimal_h, {
        "visited": len(checked),
        "mss_warmstart": used_mss,
        "meet_warmstart": used_meet,
        "direction": "bidirectional",
        "met_in_middle": met_in_middle,
    }


def functional_markov_chain(
    dist,
    rvs=None,
    crvs=None,
    *,
    _use_mss_warmstart=True,
    _strategy="auto",
    _stats=None,
):
    """
    Return H(W) for the smallest function W of `dist` which renders `rvs` independent.

    Parameters
    ----------
    dist : Distribution
        The distribution for which the smallest function will be constructed.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the total correlation. If None, then the
        total correlation is calculated over all random variables, which is
        equivalent to passing `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.

    Returns
    -------
    h : float
        The entropy of the smallest valid functional Markov variable W.

    Notes
    -----
    Three exact searches are available:

    * **Coarsen** (top-down): merge blocks starting from the finest outcome
      partition, or from the joint-MSS partition when it already satisfies
      B = 0.
    * **Refine** (bottom-up): split blocks starting from the Gács–Körner meet
      partition until B = 0.
    * **Bidirectional**: interleaved refine/coarsen with a shared visited set,
      growing from meet and MSS toward the middle.

    With ``_strategy='auto'`` (default), cheap probes on the meet and MSS
    seeds pick among the three.  See james2017multivariate.
    """
    optimal_b = dual_total_correlation(dist, rvs, crvs)
    ctx = prepare_functional_search(dist, rvs=rvs, crvs=crvs)
    pmf_size = int(np.prod(ctx.shape))
    n_outcomes = len(ctx.dist.outcomes)

    probe = _probe_seeds(ctx, dist, rvs, crvs, pmf_size, _use_mss_warmstart=_use_mss_warmstart)

    route = _route_auto_strategy(probe, optimal_b, n_outcomes) if _strategy == "auto" else _strategy

    if route == "coarsen":
        h, winner_stats = _search_coarsen(
            ctx,
            dist,
            rvs,
            optimal_b,
            pmf_size,
            _use_mss_warmstart=_use_mss_warmstart,
            probe=probe,
        )
    elif route == "refine":
        h, winner_stats = _search_refine(ctx, dist, rvs, crvs, optimal_b, pmf_size, probe=probe)
        if h is None:
            h, winner_stats = _search_coarsen(
                ctx,
                dist,
                rvs,
                optimal_b,
                pmf_size,
                probe=probe,
            )
    elif route == "bidirectional":
        h, winner_stats = _search_bidirectional(ctx, dist, rvs, crvs, optimal_b, pmf_size, probe=probe)
    else:
        msg = f"Unknown strategy: {route!r}"
        raise ValueError(msg)

    if _stats is not None:
        _stats.update(winner_stats)
        _stats["visited"] = winner_stats["visited"]
        _stats["strategy"] = winner_stats["direction"]
        if _strategy == "auto":
            _stats["route"] = route

    return h


@unitful
def functional_common_information(dist, rvs=None, crvs=None):
    """
    Compute the functional common information, F, of `dist`. It is the entropy
    of the smallest random variable W such that all the variables in `rvs` are
    rendered independent conditioned on W, and W is a function of `rvs`.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the functional common information is
        computed.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the total correlation. If None, then the
        total correlation is calculated over all random variables, which is
        equivalent to passing `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.

    Returns
    -------
    F : float
        The functional common information.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)

    dtc = dual_total_correlation(dist, rvs, crvs)
    ent = entropy(dist, rvs, crvs)
    if np.isclose(dtc, ent):
        return dtc

    return functional_markov_chain(dist, rvs, crvs)
