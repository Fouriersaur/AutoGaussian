"""
DISCRETE SEARCH -- two libraries, one certified flag (Sec. 6 of the algorithm flow).

AUTOSCATTER's breadth-first descent through the graph lattice rests on two
monotonicity rules:

    if a graph is VALID,   every extension of it is VALID
    if a graph is INVALID, every subgraph of it is INVALID

The first is a theorem (an extension can always set the extra couplings to
zero).  The second is a theorem *only for a certified* invalid.  For a graph
the oracle merely failed to solve, applying it deletes an entire subtree on the
strength of an optimiser's bad afternoon -- and the minimal solutions live at
the bottom of exactly those subtrees.  This module therefore keeps two
libraries and prunes with the second rule **only** when the entry carries
``certified=True``:

    for g in breadth_first_subgraphs(root):
        v, witness = oracle(g)
        if v is VALID:
            valid[g] = witness                       # extensions are implied
        elif not is_pruning_frontier(g):
            invalid[g] = InvalidEntry(certified=False, reason="provisional")
        else:
            v, w = escalate(g)                       # fresh seeds, then proofs
            VALID        -> valid[g] = w
            INVALID_CERT -> invalid[g] certified, subgraphs condemned
            otherwise    -> invalid[g] uncertified, subgraphs still tested

The price is paid in oracle calls, and the exchange rate is
:func:`is_pruning_frontier`: escalation (which is the expensive part) only runs
where condemning a subtree would actually save work.  What comes out is
``n_uncertified``, the number of graphs rejected without proof -- the honest
completeness caveat of Sec. 8, and the reason the invalid library is an output
of the algorithm rather than a scratch pad.
"""

import numpy as np
from tqdm import tqdm

from autogaussian.graph import any_subgraph_of, is_subgraph_of_any
from autogaussian.types import (
    InvalidEntry,
    Libraries,
    REASON_DEFAULT,
    REASON_FIT,
    REASON_HURWITZ,
    REASON_PROVISIONAL,
    Verdict,
    graph_key,
)

__all__ = ["discover", "escalate", "is_pruning_frontier", "one_step_reductions"]


def one_step_reductions(graph, space):
    """All graphs obtained by weakening exactly one slot by one admissible step."""
    graph = np.asarray(graph, dtype="int8")
    for slot_idx, value in enumerate(graph):
        if value <= 0:
            continue
        allowed = [v for v in space.possible_values[slot_idx] if v < value]
        if not allowed:
            continue
        reduced = graph.copy()
        reduced[slot_idx] = max(allowed)
        yield reduced


def is_pruning_frontier(graph, libraries, space):
    """Would certifying ``graph`` actually prune anything?

    ``True`` iff at least one immediate subgraph is still undecided *and* is not
    already condemned by some other certified invalid.  Escalation is the
    expensive rung of the ladder, so it is spent only where a certificate would
    buy a subtree.
    """
    certified = libraries.certified_invalid_graphs()
    for child in one_step_reductions(graph, space):
        key = graph_key(child)
        if key in libraries.valid or key in libraries.invalid:
            continue
        if certified.size and is_subgraph_of_any(child, certified):
            continue
        return True
    return False


def escalate(optimizer, graph, libraries, reason, num_tests=None,
             use_certificates=True, verbose=False, cache=None):
    """The Sec. 6 escalation ladder for one provisionally invalid graph.

    1. **Cheap certificates.**  The structural ones -- a passive graph cannot
       leave the vacuum, the pins are not a covariance matrix at all -- are
       decided in closed form or by one SDP.  They run *before* the restarts
       rather than after: a proof that no witness exists is exactly a proof
       that the restarts are wasted, and the graph-independent SDP is solved
       once per target and cached.
    2. **Fresh seeds.**  Re-run the oracle from new random starting points --
       different feasible components of the target manifold, which is where a
       stable branch usually hides (Sec. 5).  This rung can promote the graph
       to ``VALID``.
    3. **Hard certificates.**  PBH dark mode for a stability failure (and, when
       it lands, the SOS hierarchy currently stubbed out).
    4. **Give up, loudly.**  ``INVALID_DEFAULT`` -- uncertified, and therefore
       still non-condemning.

    Returns ``(Verdict, info)``.
    """
    libraries.num_escalations += 1
    cache = libraries.cache if cache is None else cache
    options = {} if num_tests is None else {"num_tests": num_tests}
    info = {"reason": reason}

    if use_certificates:
        from autogaussian import certificates as cert

        fired, detail = cert.certify_fit_infeasible(optimizer, graph, cache=cache)
        if fired:
            info["certificate"] = detail
            info["reason"] = detail["reason"]
            if verbose:
                print("      certificate fired: %s" % detail["reason"])
            return Verdict.INVALID_CERT, info

    success, infos = optimizer.test_graph(graph, **options)
    libraries.num_oracle_calls += len(infos)
    if success:
        witness = next(info for info in infos if info["success"])
        return Verdict.VALID, witness

    best = min(infos, key=lambda info: info["loss_reached"])
    failed_on_fit = not best["loss_below_tolerance"]
    info["best"] = best
    info["reason"] = REASON_FIT if failed_on_fit else REASON_HURWITZ

    if not use_certificates or failed_on_fit:
        return Verdict.INVALID_DEFAULT, info

    fired, detail = cert.certify_no_hurwitz(optimizer, graph)
    info["certificate"] = detail
    if fired:
        info["reason"] = detail["reason"]
        if verbose:
            print("      certificate fired: %s" % detail["reason"])
        return Verdict.INVALID_CERT, info
    return Verdict.INVALID_DEFAULT, info


def discover(
    optimizer,
    min_complexity=0,
    progress=True,
    verbose=True,
    escalate_num_tests=None,
    use_certificates=True,
    prune_uncertified=False,
    max_oracle_calls=None,
    perform_graph_reduction=True,
):
    """Breadth-first descent producing the two libraries (Sec. 6).

    Parameters
    ----------
    optimizer : :class:`autogaussian.optimizer.CovarianceArchitectureOptimizer`
        Supplies the graph space, the parametrisation and the oracle.
    min_complexity : int
        Stop the descent at this complexity level.
    escalate_num_tests : int, optional
        Restart budget of the escalation rung (default: the normal oracle
        budget again, spent on *fresh* seeds).
    use_certificates : bool
        Run :mod:`autogaussian.certificates` on the escalation rung.  With
        ``False`` every unresolved graph becomes ``INVALID_DEFAULT``, which is
        the M4 configuration: nothing is ever pruned by an invalid.
    prune_uncertified : bool
        **Unsound**, off by default: reinstate AUTOSCATTER's rule that *any*
        invalid condemns its subgraphs.  Fast, and re-opens exactly the
        false-negative subtree deletion this module exists to prevent.  Only
        useful for reproducing the old behaviour.
    max_oracle_calls : int, optional
        Hard budget; the descent stops early and the remaining graphs simply
        stay untested (they are neither valid nor invalid).

    Returns
    -------
    :class:`autogaussian.types.Libraries`
    """
    space = optimizer.space
    libraries = Libraries()
    if escalate_num_tests is None:
        # fresh seeds, not more of the same: doubling the budget buys much less
        # than restarting in a different feasible component (Sec. 5)
        escalate_num_tests = optimizer.kwargs_optimization.get("num_tests", 10)

    if verbose:
        print("start breadth-first search over %i slots (max complexity %i)"
              % (space.num_slots, space.max_complexity))

    for level in range(space.max_complexity, min_complexity - 1, -1):
        candidates = _candidates_at(space, level, libraries,
                                    prune_uncertified=prune_uncertified)
        if not candidates:
            continue
        if verbose:
            print("test %i graphs with %i degrees of freedom:" % (len(candidates), level))
        iterator = tqdm(candidates) if progress else candidates
        count_tested = 0
        count_invalid = 0
        # rebuilding the valid array per candidate dominates the level once the
        # library is large; it only changes when something valid is added
        valid_graphs = libraries.valid_graphs()
        num_valid = len(libraries.valid)

        for graph in iterator:
            if max_oracle_calls is not None and libraries.num_oracle_calls >= max_oracle_calls:
                if verbose:
                    print("oracle budget exhausted, stopping the descent")
                return libraries
            # a graph that has meanwhile become an extension of a valid graph
            # is valid by monotonicity -- no need to test it
            if len(libraries.valid) != num_valid:
                valid_graphs = libraries.valid_graphs()
                num_valid = len(libraries.valid)
            if valid_graphs.size and any_subgraph_of(graph, valid_graphs):
                continue
            if graph_key(graph) in libraries.valid or graph_key(graph) in libraries.invalid:
                continue

            success, infos = optimizer.test_graph(graph)
            libraries.num_oracle_calls += len(infos)
            count_tested += 1

            if success:
                witness = next(info for info in infos if info["success"])
                libraries.add_valid(graph, witness)
                if perform_graph_reduction:
                    # read off which elements the witness actually uses: the
                    # tested graph stays in the library (it *is* valid), the
                    # reduction is what can turn out to be minimal
                    reduced = optimizer.reduce_graph(witness["x"])
                    optimizer.solutions[graph_key(reduced)] = witness
                    libraries.add_valid(reduced, witness)
                continue

            best = min(infos, key=lambda info: info["loss_reached"])
            reason = REASON_FIT if not best["loss_below_tolerance"] else REASON_HURWITZ

            if not is_pruning_frontier(graph, libraries, space):
                # nothing to prune here: record the failure, condemn nothing
                count_invalid += 1
                libraries.add_invalid(InvalidEntry(
                    np.asarray(graph, dtype="int8"), certified=False,
                    reason=REASON_PROVISIONAL,
                    info={"loss_reached": best["loss_reached"],
                          "max_real_eigenvalue": best["max_real_eigenvalue"],
                          "failed_on": reason}))
                continue

            verdict, info = escalate(optimizer, graph, libraries, reason,
                                     num_tests=escalate_num_tests,
                                     use_certificates=use_certificates,
                                     verbose=verbose)
            if verdict is Verdict.VALID:
                libraries.add_valid(graph, info)
                if perform_graph_reduction:
                    reduced = optimizer.reduce_graph(info["x"])
                    optimizer.solutions[graph_key(reduced)] = info
                    libraries.add_valid(reduced, info)
            elif verdict is Verdict.INVALID_CERT:
                count_invalid += 1
                libraries.add_invalid(InvalidEntry(
                    np.asarray(graph, dtype="int8"), certified=True,
                    reason=info["reason"], info=info))
            else:
                count_invalid += 1
                libraries.add_invalid(InvalidEntry(
                    np.asarray(graph, dtype="int8"), certified=False,
                    reason=REASON_DEFAULT,
                    info={"failed_on": info["reason"],
                          "loss_reached": info["best"]["loss_reached"],
                          "max_real_eigenvalue": info["best"]["max_real_eigenvalue"]}))

        # per-level statistics, in the shape the reporting helpers expect
        optimizer.tested_complexities.append(level)
        optimizer.num_tested_graphs.append(count_tested)
        optimizer.num_tested_invalid_graphs.append(count_invalid)

    if verbose:
        print(libraries.summary())
    return libraries


def _candidates_at(space, level, libraries, prune_uncertified=False):
    """Graphs at one complexity level that still have to be tested.

    Two prunes are sound and always applied:

    * the graph contains a known valid graph  -> valid by monotonicity, and not
      minimal, so it cannot be part of the answer;
    * the graph is a subgraph of a **certified** invalid -> invalid by the
      certificate, which really does cover the whole subtree.

    ``prune_uncertified`` adds the unsound third one.
    """
    certified = libraries.certified_invalid_graphs()
    valid = libraries.valid_graphs()
    uncertified = None
    if prune_uncertified:
        rows = [np.frombuffer(k, dtype="int8") for k in libraries.invalid]
        uncertified = np.array(rows, dtype="int8") if rows else np.zeros((0, 0), dtype="int8")

    out = []
    for graph in space.iter_graphs_at_complexity(level):
        if certified.size and is_subgraph_of_any(graph, certified):
            continue
        if uncertified is not None and uncertified.size and is_subgraph_of_any(graph, uncertified):
            continue
        if valid.size and any_subgraph_of(graph, valid):
            continue
        out.append(graph)
    return out
