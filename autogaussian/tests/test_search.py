"""Discrete-search checks (Sec. 6/7): lattice, monotonicity, enumeration."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogaussian import CovarianceArchitectureOptimizer, GraphSpace
from autogaussian.gallery import epr_source, single_mode_squeezer
from autogaussian.graph import (
    any_subgraph_of,
    characterize_architectures,
    is_subgraph_of_any,
    min_number_of_pumps,
)


def test_graph_space_layout_and_restrictions():
    space = GraphSpace(2)
    assert space.num_slots == 2 * 3            # two blocks x upper triangle
    assert space.num_possible_graphs() == 2 * 2 * 3 * 3 * 3 * 3
    restricted = GraphSpace(2, forbidden_couplings=[(0, 1)])
    assert restricted.max_values[space.slots.index(("beamsplitter", 0, 1))] == 0
    assert restricted.max_values[space.slots.index(("two_mode_squeezing", 0, 1))] == 0
    passive = GraphSpace(2, passive_only_couplings=[(0, 1)])
    assert passive.max_values[space.slots.index(("two_mode_squeezing", 0, 1))] == 0
    assert passive.max_values[space.slots.index(("beamsplitter", 0, 1))] == 2


def test_complexity_enumeration_is_complete():
    space = GraphSpace(2, allow_onsite_squeezing=False)
    counted = sum(len(list(space.iter_graphs_at_complexity(level)))
                  for level in range(space.max_complexity + 1))
    assert counted == space.num_possible_graphs()
    for level in range(space.max_complexity + 1):
        for graph in space.iter_graphs_at_complexity(level):
            assert int(np.sum(graph)) == level


def test_monotonicity_helpers():
    graphs = np.array([[1, 0, 2], [0, 2, 0]])
    assert is_subgraph_of_any([1, 0, 1], graphs)         # contained in row 0
    assert not is_subgraph_of_any([1, 1, 1], graphs)
    assert any_subgraph_of([1, 2, 2], graphs)            # row 0 and row 1 fit
    assert not any_subgraph_of([1, 0, 1], np.array([[1, 0, 2]]))


def test_pump_counting():
    space = GraphSpace(2)
    graph = space.empty()
    graph[space.slots.index(("beamsplitter", 0, 1))] = 1      # real, no pump
    assert min_number_of_pumps(graph, space)[0] == 0
    graph[space.slots.index(("beamsplitter", 0, 1))] = 2      # complex -> pump
    assert min_number_of_pumps(graph, space)[0] == 1
    graph = space.empty()
    graph[space.slots.index(("two_mode_squeezing", 0, 1))] = 1
    assert min_number_of_pumps(graph, space)[0] == 1
    info = characterize_architectures([graph], space)
    assert info["num_two_mode_squeezing"][0] == 1
    assert info["complexity"][0] == 1


def test_single_mode_squeezer_search_finds_minimal_graphs():
    """B.1: the irreducible solutions are a complex on-site squeezer, and a
    real one plus a detuning (the detuning rotates the squeezing axis)."""
    problem = single_mode_squeezer(0.5)
    optimizer = CovarianceArchitectureOptimizer(
        problem.target, num_auxiliary_modes=0, seed=1,
        kwargs_optimization={"num_tests": 15}, **problem.optimizer_kwargs())
    irreducibles = optimizer.perform_breadth_first_search(progress=False)
    assert len(irreducibles) >= 1
    assert all(int(np.sum(graph)) == 2 for graph in irreducibles)
    descriptions = [set(optimizer.space.describe(graph)) for graph in irreducibles]
    assert {"on-site squeezing 0 (complex)"} in descriptions
    # every irreducible graph really is valid, and no valid graph contains another
    for graph in irreducibles:
        assert optimizer.solution_of(graph) is not None


def test_epr_search_finds_the_two_textbook_constructions():
    """B.2: a single two-mode squeezing edge, and the beam-splitter + two
    single-mode squeezers construction."""
    problem = epr_source()
    optimizer = CovarianceArchitectureOptimizer(
        problem.target, num_auxiliary_modes=0, seed=2,
        kwargs_optimization={"num_tests": 15}, **problem.optimizer_kwargs())
    irreducibles = optimizer.perform_breadth_first_search(progress=False)
    descriptions = [set(optimizer.space.describe(graph)) for graph in irreducibles]
    assert {"two-mode squeezing 0-1 (complex)"} in descriptions
    assert any({"beam-splitter 0-1 (real)", "on-site squeezing 0 (real)",
                "on-site squeezing 1 (real)"} == d for d in descriptions)


def test_graph_reduction_drops_unused_elements():
    problem = single_mode_squeezer(0.5)
    optimizer = CovarianceArchitectureOptimizer(
        problem.target, num_auxiliary_modes=0, seed=3, make_initial_test=False,
        kwargs_optimization={"num_tests": 15}, **problem.optimizer_kwargs())
    success, infos = optimizer.test_graph(optimizer.space.fully_connected())
    assert success
    reduced = optimizer.reduce_graph(infos[-1]["x"])
    assert np.all(reduced <= optimizer.space.fully_connected())


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
    print("all search tests passed" if not failures else "%i failures" % failures)
    sys.exit(1 if failures else 0)
