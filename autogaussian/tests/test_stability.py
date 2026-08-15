"""Stability primitives (Sec. 5, acceptance criteria Sec. 8.3).

The analytic abscissa gradient has to agree with finite differences where
``alpha`` is smooth, and the machinery has to *survive* the two places where it
is not: a rightmost tie and an exceptional point.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogaussian.stability import (
    abscissa_and_gradient,
    abscissa_gradient,
    eigenvalue_condition_number,
    gradient_sampling,
    is_hurwitz,
    min_norm_convex_hull,
    rightmost_eigenpairs,
    spectral_abscissa,
)


def _linear_family(seed=0, dimension=3, num_parameters=4):
    """``M(x) = A_0 + sum_k x_k A_k`` with random complex coefficients."""
    rng = np.random.default_rng(seed)
    matrices = [rng.normal(size=(dimension, dimension))
                + 1j * rng.normal(size=(dimension, dimension))
                for _ in range(num_parameters + 1)]

    def matrix(x):
        M = matrices[0].copy()
        for value, A in zip(x, matrices[1:]):
            M = M + value * A
        return M

    def jacobian(x):
        return np.array(matrices[1:])

    return matrix, jacobian, num_parameters


def test_analytic_gradient_matches_finite_difference():
    """Sec. 8.3: agreement to ~1e-6 away from any degeneracy."""
    matrix, jacobian, num_parameters = _linear_family()
    rng = np.random.default_rng(1)
    x = rng.normal(size=num_parameters)

    alpha, grad, diagnostics = abscissa_and_gradient(matrix(x), jacobian(x))
    assert diagnostics["trustworthy"]
    assert diagnostics["condition"] < 1e3

    step = 1e-6
    numerical = np.zeros(num_parameters)
    for k in range(num_parameters):
        plus, minus = x.copy(), x.copy()
        plus[k] += step
        minus[k] -= step
        numerical[k] = (spectral_abscissa(matrix(plus))
                        - spectral_abscissa(matrix(minus))) / (2 * step)
    assert np.allclose(grad, numerical, atol=1e-6)
    assert np.isclose(alpha, spectral_abscissa(matrix(x)))


def test_conjugate_tie_is_recognised_as_smooth():
    """A real matrix ties its rightmost pair by conjugation.

    That tie is spurious -- both members give the same gradient of ``Re lambda``
    -- and must not trigger the non-smooth fallback, otherwise every BdG matrix
    with an oscillating mode would be declared a corner.
    """
    M = np.array([[-0.1, 1.0], [-1.0, -0.1]])
    dM = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    pairs = rightmost_eigenpairs(M)
    assert len(pairs) == 2                                # conjugate pair
    assert np.isclose(np.real(pairs[0]["value"]), np.real(pairs[1]["value"]))

    alpha, grad, diagnostics = abscissa_and_gradient(M, dM)
    assert diagnostics["multiplicity"] == 2
    assert diagnostics["consistent"]
    assert diagnostics["trustworthy"]
    assert np.isclose(alpha, -0.1)
    assert np.allclose(grad, [1.0])                       # d/dt max Re eig(M + t I)


def test_genuine_tie_gives_a_clarke_direction():
    """Two *different* eigenvalues tied for rightmost: a Lipschitz corner.

    The gradients disagree, the point is flagged untrustworthy, and what comes
    back is the minimum-norm element of their convex hull.
    """
    M = np.diag([1j, -1j, -1.0])                          # distinct, tied in Re
    dM = np.array([np.diag([1.0, -1.0, 0.0]).astype(complex)])
    alpha, grad, diagnostics = abscissa_and_gradient(M, dM)
    assert diagnostics["multiplicity"] == 2
    assert not diagnostics["consistent"]
    assert not diagnostics["trustworthy"]
    # the two tied gradients are +1 and -1: the corner is a maximum, so the
    # Clarke steepest-descent element is 0
    assert np.isclose(alpha, 0.0)
    assert abs(grad[0]) < 1e-6


def test_condition_number_diverges_at_an_exceptional_point():
    """Sec. 8.3: ``1/|u^H v|`` grows like ``t^{-1/2}`` on approach to an EP."""
    def M(t):
        return np.array([[0.0, 1.0], [t, 0.0]], dtype=complex)

    distances = np.array([1e-2, 1e-4, 1e-6, 1e-8])
    conditions = np.array([eigenvalue_condition_number(M(t)) for t in distances])
    assert np.all(np.diff(conditions) > 0)                # monotonically worse
    # two decades in t buy one decade in the condition number: cond ~ t^{-1/2}
    ratios = conditions[:-1] / conditions[1:]
    assert np.allclose(ratios, np.sqrt(distances[1:] / distances[:-1]), rtol=0.05)


def test_gradient_sampling_stays_bounded_across_an_exceptional_point():
    """Sec. 8.3: the single analytic gradient diverges at an EP; sampling over a
    ball wide enough to straddle it returns a bounded step."""
    def matrix(x):
        return np.array([[0.0, 1.0], [x[0], 0.0]], dtype=complex)

    def jacobian(x):
        return np.array([[[0.0, 0.0], [1.0, 0.0]]], dtype=complex)

    near = np.array([1e-10])
    analytic = abscissa_gradient(matrix(near), jacobian(near))
    assert abs(analytic[0]) > 1e3                          # genuinely divergent

    sampled, info = gradient_sampling(matrix, jacobian, near, epsilon=1e-3,
                                      num_samples=40,
                                      rng=np.random.default_rng(0))
    assert info["num_used"] > 0
    assert abs(sampled[0]) < abs(analytic[0]) / 100.0
    assert np.isfinite(sampled[0])


def test_min_norm_convex_hull_picks_the_interior_zero():
    grad = min_norm_convex_hull(np.array([[1.0, 0.0], [-1.0, 0.0]]))
    assert np.allclose(grad, [0.0, 0.0], atol=1e-6)
    grad = min_norm_convex_hull(np.array([[2.0, 1.0], [2.0, -1.0]]))
    assert np.allclose(grad, [2.0, 0.0], atol=1e-6)


def test_non_finite_matrix_reports_an_unbounded_abscissa():
    """An optimiser line search does step a coupling into `exp` overflow.  The
    answer there is "unstable, no descent information" -- not an exception and
    not a silent NaN, both of which would propagate into the verdict."""
    M = np.array([[np.inf, 0.0], [0.0, -1.0]])
    dM = np.array([np.eye(2, dtype=complex)])

    assert spectral_abscissa(M) == np.inf
    assert rightmost_eigenpairs(M) == []
    alpha, grad, diagnostics = abscissa_and_gradient(M, dM)
    assert alpha == np.inf
    assert diagnostics["nonfinite"]
    assert not diagnostics["trustworthy"]
    assert np.allclose(grad, 0.0)
    assert not is_hurwitz(M)

    nan = np.array([[np.nan, 1.0], [0.0, -1.0]])
    assert spectral_abscissa(nan) == np.inf
    assert abscissa_and_gradient(nan, dM)[0] == np.inf


def test_is_hurwitz_respects_the_margin():
    M = np.diag([-0.5, -2.0])
    assert is_hurwitz(M)
    assert is_hurwitz(M, delta=0.4)
    assert not is_hurwitz(M, delta=0.6)
    assert not is_hurwitz(np.diag([0.1, -2.0]))


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
    print("all stability tests passed" if not failures else "%i failures" % failures)
    sys.exit(1 if failures else 0)
