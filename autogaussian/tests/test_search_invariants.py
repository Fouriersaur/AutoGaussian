"""Two-library engine (Sec. 6, acceptance criteria Sec. 8.4).

These are the invariants that define "not broken" for the search.  The one
that matters most is the third: an invalid graph that was never *proved*
invalid must not remove its subgraphs from the search, because that is exactly
where a minimal solution hides when the optimiser has a bad afternoon.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogaussian import CovarianceArchitectureOptimizer, GraphSpace
from autogaussian.gallery import epr_source, single_mode_squeezer
from autogaussian.graph import any_subgraph_of
from autogaussian.search import discover, is_pruning_frontier, one_step_reductions
from autogaussian.types import InvalidEntry, Libraries, Verdict, graph_key

TOLERANCE = 1.0e-10


def build(problem, num_auxiliary_modes=0, num_tests=8, **kwargs):
    kwargs.update(problem.optimizer_kwargs())
    return CovarianceArchitectureOptimizer(
        problem.target, num_auxiliary_modes=num_auxiliary_modes, seed=0,
        make_initial_test=False, kwargs_optimization={"num_tests": num_tests}, **kwargs)


def test_oracle_never_emits_a_certified_verdict():
    """Invariant 1: certification is not the oracle's job (Sec. 9.2)."""
    optimizer = build(single_mode_squeezer(0.5))
    space = optimizer.space
    for graph in [space.fully_connected(), space.empty()]:
        verdict, _ = optimizer.oracle.verdict(
            optimizer.param.free_indices(graph), num_tests=3,
            rng=np.random.default_rng(0))
        assert verdict in (Verdict.VALID, Verdict.INVALID_PROV)
        assert not verdict.certified


def test_every_tested_graph_lands_in_exactly_one_library():
    """Invariant 3 of Sec. 5.7."""
    optimizer = build(single_mode_squeezer(0.5))
    libraries = discover(optimizer, progress=False, verbose=False)
    assert set(libraries.valid) & set(libraries.invalid) == set()
    assert len(libraries.valid) + len(libraries.invalid) == len(
        set(libraries.valid) | set(libraries.invalid))


def test_valid_entries_carry_a_witness_that_fits_and_is_stable():
    """Invariant 4: VALID is a claim backed by numbers."""
    optimizer = build(single_mode_squeezer(0.5))
    libraries = discover(optimizer, progress=False, verbose=False)
    assert libraries.valid
    for witness in libraries.valid.values():
        assert witness["loss_reached"] <= TOLERANCE
        assert witness["max_real_eigenvalue"] < 0.0
        assert optimizer.oracle.fit_loss(witness["x"]) <= TOLERANCE
        assert optimizer.oracle.abscissa(witness["x"]) < 0.0


def test_uncertified_invalids_never_condemn_their_subgraphs():
    """Invariant 2 -- the correctness-critical one (Sec. 5.7, Sec. 8.4).

    With the certificates switched off, *nothing* may be pruned by an invalid:
    every immediate subgraph of an uncertified invalid must itself have been
    decided, unless it was skipped for the one sound reason (it contains a
    valid graph, hence is valid and not minimal).
    """
    optimizer = build(single_mode_squeezer(0.5))
    libraries = discover(optimizer, progress=False, verbose=False,
                         use_certificates=False)

    assert libraries.n_certified() == 0
    assert libraries.n_uncertified() == len(libraries.invalid)

    valid_graphs = libraries.valid_graphs()
    for entry in libraries.invalid.values():
        assert not entry.certified
        for child in one_step_reductions(entry.graph, optimizer.space):
            covered_by_valid = valid_graphs.size and any_subgraph_of(child, valid_graphs)
            assert graph_key(child) in libraries.valid \
                or graph_key(child) in libraries.invalid \
                or covered_by_valid


def test_certified_invalids_do_prune():
    """The other side of the same coin: a certificate *is* allowed to condemn.

    The passive one-mode graph cannot squeeze, and that is a proof, so its
    subgraphs are removed from the search instead of being retested.
    """
    problem = single_mode_squeezer(0.5)
    space = GraphSpace(1, allow_two_mode_squeezing=False, allow_onsite_squeezing=False)
    optimizer = CovarianceArchitectureOptimizer(
        problem.target, num_auxiliary_modes=0, seed=0, make_initial_test=False,
        graph_space=space, optimize_gauge=False,
        kwargs_optimization={"num_tests": 3})
    libraries = discover(optimizer, progress=False, verbose=False)

    assert libraries.n_certified() >= 1
    certified = libraries.certified_invalid_graphs()
    assert certified.size
    # the condemned subgraphs were never tested
    for graph in certified:
        for child in one_step_reductions(graph, optimizer.space):
            assert graph_key(child) not in libraries.valid
            assert graph_key(child) not in libraries.invalid


def test_pruning_frontier_is_false_when_the_subtree_is_already_decided():
    """``is_pruning_frontier`` gates the expensive rung: escalating a graph
    whose subtree is already covered would buy nothing."""
    optimizer = build(single_mode_squeezer(0.5))
    space = optimizer.space
    graph = space.fully_connected()
    libraries = Libraries()

    assert is_pruning_frontier(graph, libraries, space)      # nothing decided yet
    for child in one_step_reductions(graph, space):
        libraries.add_invalid(InvalidEntry(np.asarray(child, dtype="int8")))
    assert not is_pruning_frontier(graph, libraries, space)  # all children decided


def test_completeness_statement_tracks_the_uncertified_count():
    optimizer = build(single_mode_squeezer(0.5))
    libraries = discover(optimizer, progress=False, verbose=False)
    statement = libraries.completeness_statement(optimizer.space)
    if libraries.n_uncertified() == 0:
        assert "complete and certified" in statement
    else:
        assert "complete up to at most %i" % libraries.n_uncertified() in statement
    assert optimizer.n_uncertified() == 0     # not run through the optimizer yet


def test_two_library_search_reproduces_the_epr_constructions():
    """Sec. 8.4 end to end: the engine still finds both textbook solutions, and
    reports how much of the answer is certified."""
    # detunings switched off: they are not part of either textbook solution and
    # dropping them shrinks the lattice from 324 graphs to 81
    optimizer = build(epr_source(), num_tests=12, allow_detunings=False)
    irreducibles = optimizer.perform_breadth_first_search(progress=False, verbose=False)
    descriptions = [set(optimizer.space.describe(graph)) for graph in irreducibles]
    assert {"two-mode squeezing 0-1 (complex)"} in descriptions
    assert any({"beam-splitter 0-1 (real)", "on-site squeezing 0 (real)",
                "on-site squeezing 1 (real)"} == d for d in descriptions)
    assert optimizer.libraries is not None
    assert optimizer.n_uncertified() == optimizer.libraries.n_uncertified()
    statement = optimizer.completeness_statement()
    assert "complete" in statement


if __name__ == "__main__":
    failures = 0
    for name, func in sorted(list(globals().items())):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print("PASS  %s" % name)
            except AssertionError as error:
                failures += 1
                print("FAIL  %s: %s" % (name, error))
    print("all search-invariant tests passed" if not failures else "%i failures" % failures)
    sys.exit(1 if failures else 0)
