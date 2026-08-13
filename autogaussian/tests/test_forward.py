"""Forward-map checks (Sec. 3 / App. A) against closed-form results."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp

from autogaussian.forward import (
    is_stable,
    output_covariance_quadrature,
    response_matrices,
)
from autogaussian.nambu import (
    build_H_bdg,
    channel_covariance,
    duan_sum,
    symplectic_eigenvalues,
    vacuum_covariance,
)


def device(g, nu, gamma=None, kappa=None):
    g = jnp.asarray(np.atleast_2d(np.asarray(g, dtype=complex)))
    nu = jnp.asarray(np.atleast_2d(np.asarray(nu, dtype=complex)))
    n = g.shape[0]
    gamma = jnp.zeros(n) if gamma is None else jnp.asarray(np.asarray(gamma, dtype=float))
    kappa = jnp.ones(n) if kappa is None else jnp.asarray(np.asarray(kappa, dtype=float))
    return build_H_bdg(g, nu), gamma, kappa


def covariance(H, gamma, kappa, Omega=0.0, sigma_in=None, thetas=None, num_ports=None):
    n = H.shape[0] // 2
    if sigma_in is None:
        sigma_in = vacuum_covariance(n)
    if thetas is None:
        thetas = jnp.zeros(n)
    return np.asarray(output_covariance_quadrature(
        H, gamma, kappa, Omega, sigma_in, thetas, num_ports))


def test_passive_lossless_is_unitary_and_keeps_vacuum():
    """No squeezing, no loss -> S unitary and vacuum stays vacuum."""
    g = [[0.3, 0.7 + 0.2j], [0.7 - 0.2j, -0.15]]
    nu = np.zeros((2, 2))
    H, gamma, kappa = device(g, nu)
    for Omega in (0.0, 0.37, -1.2):
        S, N = response_matrices(H, gamma, kappa, Omega)
        assert np.allclose(np.asarray(S) @ np.conj(np.asarray(S)).T, np.eye(4), atol=1e-10)
        V = covariance(H, gamma, kappa, Omega)
        assert np.allclose(V, np.eye(4), atol=1e-10)


def test_single_mode_squeezer_matches_analytics():
    """App. E:  sigma_min(0) = ((kappa/2 - lambda)/(kappa/2 + lambda))^2."""
    for lam in (0.1, 0.25, 0.3, 0.45):
        H, gamma, kappa = device([[0.0]], [[1j * lam]])
        V = covariance(H, gamma, kappa, 0.0)
        expected_min = ((0.5 - lam) / (0.5 + lam)) ** 2
        assert np.isclose(V[1, 1].real, expected_min, rtol=1e-9)
        assert np.isclose(V[0, 0].real, 1.0 / expected_min, rtol=1e-9)
        assert np.isclose(V[0, 1].real, 0.0, atol=1e-9)
        # lossless -> pure state
        assert np.allclose(symplectic_eigenvalues(np.real(V)), 1.0, atol=1e-8)


def test_squeezing_phase_rotates_with_arg_nu():
    """A real nu squeezes the +-45 degree quadratures, not x/p."""
    lam = 0.3
    H, gamma, kappa = device([[0.0]], [[lam]])
    V = np.real(covariance(H, gamma, kappa, 0.0))
    assert np.isclose(V[0, 0], V[1, 1], rtol=1e-9)
    assert not np.isclose(V[0, 1], 0.0, atol=1e-6)
    eigenvalues = np.linalg.eigvalsh(V)
    expected_min = ((0.5 - lam) / (0.5 + lam)) ** 2
    assert np.isclose(eigenvalues[0], expected_min, rtol=1e-8)


def test_decay_ratios_inert_at_zero_frequency():
    """App. A.4(c): kappa~ drops out at Omega = 0 and shapes the spectrum off it."""
    g = [[0.2, 0.5], [0.5, -0.1]]
    nu = [[0.0, 0.25j], [0.25j, 0.0]]
    H, gamma, _ = device(g, nu)
    V1 = covariance(H, gamma, jnp.array([1.0, 1.0]), 0.0)
    V2 = covariance(H, gamma, jnp.array([0.4, 2.5]), 0.0)
    assert np.allclose(V1, V2, atol=1e-10)
    W1 = covariance(H, gamma, jnp.array([1.0, 1.0]), 0.5)
    W2 = covariance(H, gamma, jnp.array([0.4, 2.5]), 0.5)
    assert not np.allclose(W1, W2, atol=1e-3)


def test_loss_degrades_squeezing():
    """App. A.4(b) / Sec. 11: intrinsic loss mixes vacuum in through N."""
    lam = 0.3
    floor = []
    for gamma in (0.0, 0.05, 0.2, 1.0):
        H, gam, kappa = device([[0.0]], [[1j * lam]], gamma=[gamma])
        V = np.real(covariance(H, gam, kappa, 0.0))
        floor.append(V[1, 1])
    assert np.all(np.diff(floor) > 0)
    assert floor[0] < 1.0


def test_thermal_input_passes_through_uncoupled_mode():
    """An uncoupled lossless mode has S = -1: the declared input comes back."""
    H, gamma, kappa = device([[0.0]], [[0.0]])
    n = 0.7
    sigma_in = channel_covariance(n=np.array([n]), m=0.0)
    V = np.real(covariance(H, gamma, kappa, 0.0, sigma_in=sigma_in))
    assert np.allclose(V, (2 * n + 1) * np.eye(2), atol=1e-10)


def test_two_mode_squeezing_is_entangled():
    """A single squeezing edge makes an EPR pair: local thermal, joint squeezed.

    ``arg(nu) = pi/2`` puts the correlations in the ``x-x`` / ``p-p`` entries,
    which is the App. B.2 target form ``[[a,0,c,0],[0,a,0,-c],...]``.
    """
    lam = 0.25
    H, gamma, kappa = device(np.zeros((2, 2)), [[0.0, 1j * lam], [1j * lam, 0.0]])
    V = np.real(covariance(H, gamma, kappa, 0.0))
    assert V[0, 0] > 1.0 and V[2, 2] > 1.0          # local ports are thermal
    assert V[0, 2] > 0.0 and V[1, 3] < 0.0          # EPR correlation pattern
    assert duan_sum(V, 0, 1) < 4.0                  # certified entangled
    assert np.allclose(symplectic_eigenvalues(V), 1.0, atol=1e-8)


def test_stability_gate():
    """Sec. 5: a single-mode squeezer goes unstable at lambda = kappa/2."""
    stable = device([[0.0]], [[0.45j]])
    unstable = device([[0.0]], [[0.55j]])
    assert is_stable(*stable)
    assert not is_stable(*unstable)


def test_gauge_is_a_symplectic_rotation():
    """App. A.4(e): theta rotates each 2x2 block; eigenvalues are invariant."""
    lam = 0.3
    H, gamma, kappa = device([[0.0]], [[lam]])
    V0 = np.real(covariance(H, gamma, kappa, 0.0, thetas=jnp.array([0.0])))
    V1 = np.real(covariance(H, gamma, kappa, 0.0, thetas=jnp.array([0.4])))
    assert not np.allclose(V0, V1, atol=1e-6)
    assert np.allclose(np.sort(np.linalg.eigvalsh(V0)),
                       np.sort(np.linalg.eigvalsh(V1)), atol=1e-9)


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
    print("%s" % ("all forward-map tests passed" if not failures
                  else "%i failures" % failures))
    sys.exit(1 if failures else 0)
