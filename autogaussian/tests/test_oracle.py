"""Oracle checks (Sec. 4/5): two-sided verdicts, gauge, constraints, sweeps."""

import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogaussian import (
    CovarianceArchitectureOptimizer,
    CovarianceTarget,
    GraphSpace,
    TransmissionConstraint,
    qidx,
)
from autogaussian.gallery import broadband_squeezer, single_mode_squeezer

KWARGS = {"num_tests": 15}


def build(target, num_auxiliary_modes=0, **kwargs):
    kwargs.setdefault("kwargs_optimization", KWARGS)
    return CovarianceArchitectureOptimizer(
        target, num_auxiliary_modes=num_auxiliary_modes, make_initial_test=False,
        seed=0, **kwargs)


def test_squeezing_target_is_valid_and_reproduces_the_target():
    problem = single_mode_squeezer(0.5)
    optimizer = build(problem.target, **problem.optimizer_kwargs())
    success, infos = optimizer.test_graph(optimizer.space.fully_connected())
    assert success
    V = np.real(optimizer.oracle.covariance(infos[-1]["x"], 0.0))
    assert np.allclose(V, [[0.5, 0.0], [0.0, 2.0]], atol=1e-4)


def test_passive_device_cannot_squeeze():
    """A two-sided verdict: with no anomalous block, sub-vacuum output is
    impossible, so the oracle must return INVALID (App. A.1)."""
    problem = single_mode_squeezer(0.5)
    space = GraphSpace(1, allow_two_mode_squeezing=False, allow_onsite_squeezing=False)
    optimizer = build(problem.target, graph_space=space, optimize_gauge=False)
    success, infos = optimizer.test_graph(optimizer.space.fully_connected())
    assert not success
    assert min(info["loss_reached"] for info in infos) > 1e-3


def test_stability_gate_rejects_above_threshold():
    """Sec. 5: loss alone is not enough -- M~ must sit in the left half-plane."""
    problem = single_mode_squeezer(0.5)
    optimizer = build(problem.target, **problem.optimizer_kwargs())
    x = np.zeros(optimizer.param.num_variables)
    x[optimizer.param.index("|nu_{0,0}|")] = 0.7      # above the threshold 1/2
    assert not optimizer.oracle.is_stable(x)
    x[optimizer.param.index("|nu_{0,0}|")] = 0.3
    assert optimizer.oracle.is_stable(x)


def test_free_target_symbol_is_optimised():
    """Free symbols in the target become optimisation variables."""
    t = sp.Symbol("t", real=True)
    target = CovarianceTarget(num_ports=1, name="free anti-squeezed quadrature")
    target.pin((0, 0), 0.25)
    target.pin((1, 1), t)
    target.pin((0, 1), 0.0)
    optimizer = build(target, optimize_gauge=False)
    success, infos = optimizer.test_graph(optimizer.space.fully_connected())
    assert success
    solution = infos[-1]["solution_dict"]["t"]
    assert solution > 1.0    # a pure 4x-squeezed state has to anti-squeeze


def test_form_pin_matches_nullifier_variance():
    """Quadratic-form pins (App. B.5) measure joint quadratures."""
    target = CovarianceTarget(num_ports=2, name="nullifier")
    vector = np.zeros(4)
    vector[qidx(0, 1)] = 1.0     # p_0
    vector[qidx(1, 0)] = -1.0    # x_1
    target.pin_form(vector, 0.2)
    optimizer = build(target, optimize_gauge=False)
    success, infos = optimizer.test_graph(optimizer.space.fully_connected())
    assert success
    V = np.real(optimizer.oracle.covariance(infos[-1]["x"], 0.0))
    assert np.isclose(vector @ V @ vector, 0.2, atol=1e-4)


def test_derivative_pins_flatten_the_spectrum():
    """App. B.4: pinned Omega-derivatives really do flatten sigma_xx(Omega)."""
    problem = broadband_squeezer(0.5, num_derivatives=2)
    optimizer = build(problem.target, **problem.optimizer_kwargs())
    success, infos = optimizer.test_graph(optimizer.space.fully_connected())
    assert success
    x = infos[-1]["x"]
    step = 1e-3
    values = [np.real(optimizer.oracle.covariance(x, omega)[0, 0])
              for omega in (-step, 0.0, step)]
    first = (values[2] - values[0]) / (2 * step)
    second = (values[2] - 2 * values[1] + values[0]) / step ** 2
    assert np.isclose(values[1], 0.5, atol=1e-4)
    assert abs(first) < 1e-3
    assert abs(second) < 1e-2


def test_transmission_constraint_prevents_decoupled_solution():
    """Sec. 11: without a transmission pin the directional target is met by
    simply decoupling the ports."""
    target = CovarianceTarget(num_ports=2, name="directional")
    target.pin_matrix([[1.0, 0.0, None, None],
                       [0.0, 1.0, None, None],
                       [None, None, 0.2, 0.0],
                       [None, None, 0.0, 5.0]])
    optimizer = build(target, num_auxiliary_modes=1, optimize_gauge=False,
                      constraints=(TransmissionConstraint(1, 0, 1.3),))
    success, infos = optimizer.test_graph(optimizer.space.fully_connected())
    assert success
    S, _ = optimizer.oracle.scattering(infos[-1]["x"], 0.0)
    assert np.isclose(abs(S[1, 0]) ** 2, 1.3, atol=1e-3)
    assert abs(S[0, 1]) ** 2 < abs(S[1, 0]) ** 2      # genuinely directional


def test_zero_frequency_target_freezes_decay_ratios():
    """App. A.4(c): kappa~ is inert at Omega = 0, so it must not be a free
    variable for a single-point target."""
    problem = single_mode_squeezer(0.5)
    optimizer = build(problem.target, **problem.optimizer_kwargs())
    frozen = optimizer.param.frozen_indices(optimizer.space.fully_connected())
    assert optimizer.param.log_kappa_idx[0] in frozen


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
    print("all oracle tests passed" if not failures else "%i failures" % failures)
    sys.exit(1 if failures else 0)
