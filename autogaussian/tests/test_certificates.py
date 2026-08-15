"""Infeasibility certificates (Sec. 6 ESCALATE, acceptance criteria Sec. 8.5).

Every test here is about the *asymmetry*: a certificate that fires is a proof
and may condemn a subtree, a certificate that does not fire says nothing at all.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogaussian import CovarianceArchitectureOptimizer, CovarianceTarget, GraphSpace
from autogaussian.certificates import (
    certify_fit_infeasible,
    certify_no_hurwitz,
    certify_passive_range,
    certify_target_unphysical,
    has_cvxpy,
    lyapunov_feasible,
    pbh_dark_mode,
    pbh_non_stabilizable,
    sos_no_hurwitz,
    symplectic_form,
)
from autogaussian.gallery import single_mode_squeezer

needs_cvxpy = pytest.mark.skipif(not has_cvxpy(), reason="cvxpy is not installed")


def build(target, space=None, **kwargs):
    kwargs.setdefault("kwargs_optimization", {"num_tests": 3})
    return CovarianceArchitectureOptimizer(
        target, num_auxiliary_modes=0, make_initial_test=False, seed=0,
        graph_space=space, **kwargs)


@needs_cvxpy
def test_target_below_the_uncertainty_floor_is_certified_infeasible():
    """Sec. 8.5: ``Var(x) Var(p) < 1`` is not a state, so no graph realises it."""
    target = CovarianceTarget(num_ports=1, name="below the floor")
    target.pin((0, 0), 0.4)
    target.pin((1, 1), 1.5)                     # 0.4 * 1.5 = 0.6 < 1
    target.pin((0, 1), 0.0)
    fired, detail = certify_target_unphysical(target, num_ports=1)
    assert fired
    assert detail["conclusive"]
    assert detail["reason"] == "fit_range"


@needs_cvxpy
def test_minimum_uncertainty_target_is_not_certified():
    """The mirror image: a legitimate pure squeezed state must *not* fire."""
    target = CovarianceTarget(num_ports=1, name="pure squeezed")
    target.pin((0, 0), 0.5)
    target.pin((1, 1), 2.0)
    target.pin((0, 1), 0.0)
    fired, detail = certify_target_unphysical(target, num_ports=1)
    assert not fired
    assert not detail["conclusive"]


def test_symplectic_form_normalisation():
    """Vacuum saturates ``V + i Omega >= 0`` in the vacuum-floor-1 convention."""
    Omega = symplectic_form(1)
    vacuum = np.eye(2)
    assert np.min(np.linalg.eigvalsh(vacuum + 1j * Omega)) > -1e-12
    squeezed = np.diag([0.5, 2.0])
    assert np.min(np.linalg.eigvalsh(squeezed + 1j * Omega)) > -1e-12
    unphysical = np.diag([0.5, 1.5])
    assert np.min(np.linalg.eigvalsh(unphysical + 1j * Omega)) < -1e-9


def test_passive_graph_cannot_squeeze_is_certified():
    """A beam-splitter-only graph maps vacuum to vacuum at every frequency, so
    a sub-vacuum pin is unreachable *for that graph* -- graph-dependent, and a
    proof rather than an optimiser's opinion."""
    problem = single_mode_squeezer(0.5)
    space = GraphSpace(1, allow_two_mode_squeezing=False, allow_onsite_squeezing=False)
    optimizer = build(problem.target, space=space, optimize_gauge=False)
    fired, detail = certify_passive_range(optimizer, optimizer.space.fully_connected())
    assert fired
    assert detail["reason"] == "passive_range"

    fired, detail = certify_fit_infeasible(optimizer, optimizer.space.fully_connected())
    assert fired


def test_active_graph_is_not_certified_by_the_passive_test():
    problem = single_mode_squeezer(0.5)
    optimizer = build(problem.target, optimize_gauge=False)
    fired, detail = certify_passive_range(optimizer, optimizer.space.fully_connected())
    assert not fired
    assert detail["reason"] == "not_passive"


def test_pbh_fires_on_a_structural_dark_mode():
    """Sec. 8.5: dissipators that leave a subspace uncoupled -> not
    stabilizable, for the whole parameter family."""
    A = np.diag([1.0, -1.0])              # mode 0 grows
    C = np.array([[0.0, 1.0]])            # ... and only mode 1 is damped
    fired, detail = pbh_non_stabilizable(A, C)
    assert fired
    assert detail["rank"] < detail["dimension"]

    damped = np.array([[1.0, 0.0], [0.0, 1.0]])
    fired, detail = pbh_non_stabilizable(A, damped)
    assert not fired


def test_pbh_is_inconclusive_when_every_mode_is_damped():
    """In this device family every mode is coupled to an input line, so the
    dark-mode test can never fire -- and says so instead of pretending."""
    problem = single_mode_squeezer(0.5)
    optimizer = build(problem.target, optimize_gauge=False)
    fired, detail = pbh_dark_mode(optimizer, optimizer.space.fully_connected(),
                                 rng=np.random.default_rng(0))
    assert not fired
    assert not detail["conclusive"]


def test_sos_certificate_is_a_declared_stub():
    problem = single_mode_squeezer(0.5)
    optimizer = build(problem.target, optimize_gauge=False)
    graph = optimizer.space.fully_connected()
    with pytest.raises(NotImplementedError):
        sos_no_hurwitz(optimizer, graph, method="sos")

    # the sampling stand-in gathers evidence but never certifies
    fired, detail = sos_no_hurwitz(optimizer, graph, num_samples=2,
                                   rng=np.random.default_rng(0))
    assert not fired
    assert not detail["conclusive"]

    fired, detail = certify_no_hurwitz(optimizer, graph, rng=np.random.default_rng(0))
    assert not fired


@needs_cvxpy
def test_lyapunov_check_agrees_with_the_abscissa():
    assert lyapunov_feasible(np.diag([-1.0, -2.0])) is True
    assert lyapunov_feasible(np.diag([0.5, -2.0])) is False


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
    print("all certificate tests passed" if not failures else "%i failures" % failures)
    sys.exit(1 if failures else 0)
