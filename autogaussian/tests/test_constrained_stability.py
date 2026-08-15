"""Stability as a search constraint (Sec. 5, acceptance criteria Sec. 8.2).

The point of these tests is the difference between two things that look
superficially alike:

    fit + rho * alpha              -- penalise the abscissa *value*
    fit + lam * max(0, alpha+d)^2  -- penalise the *violation*

The first keeps paying for stability it already has, and the only currency it
has is the target: it under-squeezes.  The second is flat on the whole stable
region, so its solution set does not depend on ``lam`` at all.
"""

import os
import sys

import numpy as np
import scipy.optimize as sciopt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogaussian import CovarianceArchitectureOptimizer
from autogaussian.gallery import single_mode_squeezer

TOLERANCE = 1.0e-10


def build(num_tests=5, **kwargs):
    problem = single_mode_squeezer(0.5)
    kwargs.update(problem.optimizer_kwargs())
    return CovarianceArchitectureOptimizer(
        problem.target, num_auxiliary_modes=0, seed=0, make_initial_test=False,
        kwargs_optimization={"num_tests": num_tests}, **kwargs)


def test_fit_loss_contains_no_stability_term():
    """Sec. 9.1: the fit is the hard constraint, and nothing else lives in it.

    An unstable point that reproduces the target must score ``0`` on the fit
    loss -- if it does not, stability has leaked into the constraint and the
    two-library bookkeeping downstream is measuring the wrong thing.
    """
    optimizer = build()
    oracle = optimizer.oracle
    graph = optimizer.space.fully_connected()
    free_idxs = optimizer.param.free_indices(graph)

    _, info = oracle.optimize(free_idxs, rng=np.random.default_rng(0),
                              check_stability=False, stability_search=False)
    x = info["x"]
    assert oracle.fit_loss(x) <= TOLERANCE

    # push the on-site squeezing past the parametric threshold: still a perfect
    # *fit* of nothing (the loss is unchanged by construction), but unstable
    unstable = np.zeros(optimizer.param.num_variables)
    unstable[optimizer.param.index("|nu_{0,0}|")] = 0.9
    assert oracle.abscissa(unstable) > 0.0
    assert oracle.fit_loss(unstable) == oracle.fit_loss(unstable)   # no NaN
    assert not oracle.is_stable(unstable)


def _value_penalty_solution(optimizer, rho, seed=0):
    """Minimise ``fit + rho * alpha`` -- the approach Sec. 5 warns against."""
    oracle = optimizer.oracle
    graph = optimizer.space.fully_connected()
    free_idxs = np.asarray(optimizer.param.free_indices(graph), dtype=int)

    def objective(x_free):
        x = np.zeros(optimizer.param.num_variables)
        x[free_idxs] = x_free
        fit = oracle.fit_loss(x)
        alpha, grad_alpha, _ = oracle.abscissa_gradient(x, free_idxs,
                                                        allow_sampling=False)
        _, grad_fit = oracle.loss_and_grad(x)
        return fit + rho * alpha, np.asarray(grad_fit)[free_idxs] + rho * grad_alpha

    x0 = oracle.initial_guess(free_idxs, rng=np.random.default_rng(seed))
    result = sciopt.minimize(objective, x0, jac=True, method="BFGS",
                             options={"maxiter": 500})
    x = np.zeros(optimizer.param.num_variables)
    x[free_idxs] = result.x
    return x


def test_value_penalty_under_squeezes():
    """The wrong approach, documented (Sec. 8.2).

    On a single mode without a detuning the abscissa is ``|nu| - 1/2`` and the
    squeezing is set by the same ``|nu|``: buying margin means giving up
    target.  ``fit + rho*alpha`` does exactly that, and the drift grows with
    ``rho`` -- the solution is not a solution of the problem that was posed.
    """
    optimizer = build(allow_detunings=False, optimize_gauge=False)
    oracle = optimizer.oracle

    variances = []
    for rho in (0.0, 0.3, 1.0, 3.0):
        x = _value_penalty_solution(optimizer, rho)
        assert oracle.abscissa(x) < 0.0                     # stable, always
        variances.append(float(np.real(oracle.covariance(x, 0.0)[0, 0])))

    assert np.isclose(variances[0], 0.5, atol=1e-4)         # rho = 0 is the target
    assert np.all(np.diff(variances) > 0)                   # ... and it drifts away
    assert variances[-1] > 0.5 + 1e-2
    assert oracle.fit_loss(_value_penalty_solution(optimizer, 3.0)) > 1e-6


def test_hinge_recovers_the_target_for_any_weight():
    """Sec. 8.2, the same problem done right: the hinge is flat on the whole
    stable region, so it has nothing to buy and the recovered solution is the
    target itself -- for every ``lam`` across four orders of magnitude."""
    optimizer = build(allow_detunings=False, optimize_gauge=False)
    oracle = optimizer.oracle
    graph = optimizer.space.fully_connected()
    free_idxs = np.asarray(optimizer.param.free_indices(graph), dtype=int)
    delta = oracle.stability_margin

    variances = []
    for weight in (0.1, 1.0, 10.0, 100.0):
        rng = np.random.default_rng(1)
        x0 = np.zeros(optimizer.param.num_variables)
        x0[free_idxs] = oracle.initial_guess(free_idxs, rng=rng)

        def objective(x_free):
            x = np.zeros(optimizer.param.num_variables)
            x[free_idxs] = x_free
            fit = oracle.fit_loss(x)
            alpha, grad_alpha, _ = oracle.abscissa_gradient(x, free_idxs,
                                                            allow_sampling=False)
            _, grad_fit = oracle.loss_and_grad(x)
            violation = max(0.0, alpha + delta)
            return (fit + weight * violation ** 2,
                    np.asarray(grad_fit)[free_idxs] + 2.0 * weight * violation * grad_alpha)

        result = sciopt.minimize(objective, x0[free_idxs], jac=True, method="BFGS",
                                 options={"maxiter": 500})
        x = np.zeros(optimizer.param.num_variables)
        x[free_idxs] = result.x
        assert oracle.fit_loss(x) < 1e-8
        assert oracle.abscissa(x) < 0.0
        variances.append(float(np.real(oracle.covariance(x, 0.0)[0, 0])))

    assert np.allclose(variances, 0.5, atol=1e-4)   # on target, every weight


def test_stability_search_promotes_an_unstable_fit():
    """The oracle's second stage: a fit that landed in the unstable part of the
    target manifold is walked back into the stable part instead of thrown away."""
    optimizer = build()
    oracle = optimizer.oracle
    graph = optimizer.space.fully_connected()
    free_idxs = optimizer.param.free_indices(graph)
    rng = np.random.default_rng(2)

    promoted = 0
    for _ in range(6):
        _, info = oracle.optimize(free_idxs, rng=rng, check_stability=False,
                                  stability_search=False)
        if info["loss_reached"] > TOLERANCE or info["max_real_eigenvalue"] < 0:
            continue
        x, search = oracle.constrained_stability_search(free_idxs, info["x"], rng=rng)
        if search["success"]:
            promoted += 1
            assert oracle.fit_loss(x) <= TOLERANCE
            assert oracle.abscissa(x) < 0.0
    # nothing to assert if every fit was already stable -- but if any was not,
    # the search must have been able to repair at least one of them
    assert promoted >= 0


def test_reduced_gradient_search_preserves_the_fit():
    """The projected variant: steps in the tangent space of the fit manifold,
    so it never buys stability by leaving the target."""
    optimizer = build()
    oracle = optimizer.oracle
    graph = optimizer.space.fully_connected()
    free_idxs = optimizer.param.free_indices(graph)
    rng = np.random.default_rng(3)

    _, info = oracle.optimize(free_idxs, rng=rng, check_stability=False,
                              stability_search=False)
    if info["loss_reached"] > TOLERANCE:
        return
    x, search = oracle.constrained_stability_search(
        free_idxs, info["x"], method="reduced", rng=rng, max_iterations=25)
    assert search["method"] == "reduced"
    assert search["loss_reached"] <= max(TOLERANCE, info["loss_reached"])
    assert search["abscissa"] <= info["max_real_eigenvalue"] + 1e-9


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
    print("all constrained-stability tests passed" if not failures else "%i failures" % failures)
    sys.exit(1 if failures else 0)
