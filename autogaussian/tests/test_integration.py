"""End-to-end pipeline (Sec. 9, acceptance criteria Sec. 8 / M6).

The deliverable of a run is three things, not one: the irreducible graphs, the
invalid library, and ``n_uncertified`` -- the exact strength of the claim.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogaussian import discover
from autogaussian.gallery import single_mode_squeezer
from autogaussian.types import InvalidEntry

TOLERANCE = 1.0e-10


def test_pipeline_reports_libraries_and_completeness():
    problem = single_mode_squeezer(0.5)
    result = discover(problem.target, num_auxiliary_modes=0, verbose=False,
                      kwargs_optimization={"num_tests": 8}, seed=0,
                      **problem.optimizer_kwargs())

    irreducibles = result["irreducibles"]
    assert len(irreducibles) >= 1
    optimizer = result["optimizer"]
    descriptions = [set(optimizer.space.describe(graph)) for graph in irreducibles]
    assert {"on-site squeezing 0 (complex)"} in descriptions

    # every reported device comes with a witness that meets both conditions
    for graph in irreducibles:
        solution = optimizer.solution_of(graph)
        assert solution is not None
        assert solution["loss_reached"] <= TOLERANCE
        assert solution["max_real_eigenvalue"] < 0.0

    libraries = result["libraries"]
    assert libraries is not None
    assert result["n_uncertified"] == libraries.n_uncertified()
    assert result["invalid"] == libraries.invalid
    assert all(isinstance(entry, InvalidEntry) for entry in libraries.invalid.values())

    statement = result["completeness"]
    if result["n_uncertified"] == 0:
        assert "complete and certified" in statement
    else:
        assert "complete up to at most" in statement
        # each uncertified rejection is named, so the caveat is auditable
        for entry in libraries.uncertified_entries():
            assert not entry.certified


def test_ranking_and_complexity_survive_the_two_library_engine():
    problem = single_mode_squeezer(0.5)
    result = discover(problem.target, num_auxiliary_modes=0, verbose=False,
                      kwargs_optimization={"num_tests": 8}, seed=1,
                      **problem.optimizer_kwargs())
    info = result["complexity"]
    assert len(info["complexity"]) == len(result["irreducibles"])
    assert np.all(np.diff(result["cost"]) >= 0)          # ranked, cheapest first


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
    print("all integration tests passed" if not failures else "%i failures" % failures)
    sys.exit(1 if failures else 0)
