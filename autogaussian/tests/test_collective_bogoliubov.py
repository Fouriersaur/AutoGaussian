"""D2 -- Bogoliubov channels and the collected/uncollected split
(Addendum Sec. 8, D2).  **This file is the gate for the Sec. 3 ``^ddag`` FLAG.**

The addendum leaves the Nambu adjoint provisional: "the precise placement of the
two ``sigma_z`` metrics and the channel-space doubling of ``K`` must be pinned
against the closed-form single-Bogoliubov-channel DPO **before** it is trusted."
That is done here, and the verdict is:

    X^ddag = sigma_z^sys . X^H . sigma_z^ch          <-- VERIFIED

The discriminating test is *not* the Sec. 3 sum rule.  Both candidate placements
(``^H`` and ``sigma_z ^H sigma_z``) satisfy ``S sigma_z S^H + N sigma_z N^H =
sigma_z`` to 1e-16, because the two differ only by the sign of the effective
bath squeezing angle and both are perfectly physical.  What separates them is
consistency with the **Lindblad master equation itself**: the response-side
diffusion ``D_eff = sum_c K_c^ddag sigma_in,c (K_c^ddag)^H`` must reproduce the
exact steady state through the Lyapunov equation ``M sigma_c + sigma_c M^H +
D_eff = 0``.  The tests below build that exact steady state by diagonalising the
Lindbladian in a truncated Fock space -- a route that shares no code with the
Nambu machinery -- and check the residual.  The metric placement converges to
zero with the truncation (8.6e-5 -> 1.0e-9 as the cutoff grows); the plain
adjoint sits at a constant 1.57 and never converges.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp

from autogaussian.channels import (
    Access,
    Channel,
    ChannelSet,
    dissipative_drift,
    nambu_adjoint,
    nambu_coupling,
)
from autogaussian.forward import (
    collective_dynamical_matrix,
    collective_output_covariance_quadrature,
    collective_response_matrices,
    scattering_sum_rule_residual,
)
from autogaussian.nambu import (
    build_H_bdg,
    pauli_z,
    symplectic_eigenvalues,
    vacuum_covariance,
)

KAPPA = 1.0
LAMBDA = 0.3
GAMMA = 0.6
R = 0.5
OMEGAS = (0.0, 0.4, -1.1, 3.0)


# ---------------------------------------------------------------------------
# exact reference: the Lindblad steady state in a truncated Fock space
# ---------------------------------------------------------------------------

def _fock_operators(num_modes, cutoff):
    ladder = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    ops = []
    for i in range(num_modes):
        factors = [np.eye(cutoff)] * num_modes
        factors[i] = ladder
        out = factors[0]
        for f in factors[1:]:
            out = np.kron(out, f)
        ops.append(out)
    return ops


def _lindblad_steady_state(H, jumps):
    """Steady state of ``rho_dot = -i[H, rho] + sum D[L] rho`` by direct
    diagonalisation of the Liouvillian.  Independent of everything in
    :mod:`autogaussian`."""
    dim = H.shape[0]
    eye = np.eye(dim)
    L = -1j * (np.kron(H, eye) - np.kron(eye, H.T))
    for J in jumps:
        JdJ = J.conj().T @ J
        L = L + np.kron(J, J.conj()) - 0.5 * (np.kron(JdJ, eye) + np.kron(eye, JdJ.T))
    values, vectors = np.linalg.eig(L)
    rho = vectors[:, np.argmin(np.abs(values))].reshape(dim, dim)
    rho = rho / np.trace(rho)
    return (rho + rho.conj().T) / 2.0


def _intracavity_covariance(rho, ops):
    """``sigma_c[m,n] = <{A_m, A_n^dag}>`` with ``A = (a_1..a_N, a_1^dag..)``."""
    n = len(ops)
    out = np.zeros((2 * n, 2 * n), dtype=complex)
    ex = lambda O: np.trace(rho @ O)
    for i in range(n):
        for j in range(n):
            ai, aj = ops[i], ops[j]
            out[i, j] = ex(ai @ aj.conj().T) + ex(aj.conj().T @ ai)
            out[i, j + n] = ex(ai @ aj) + ex(aj @ ai)
            out[i + n, j] = ex(ai.conj().T @ aj.conj().T) + ex(aj.conj().T @ ai.conj().T)
            out[i + n, j + n] = ex(ai.conj().T @ aj) + ex(aj @ ai.conj().T)
    return out


def _diffusion(channels, dagger):
    """``D_eff = sum_c K_c^ddag sigma_in,c (K_c^ddag)^H`` with vacuum inputs."""
    total = None
    for access in (Access.CONTROLLED, Access.UNCONTROLLED):
        rows = channels.of_access(access)
        if not rows:
            continue
        K = np.asarray(channels.coupling(access))
        Kd = dagger(K, len(rows), channels.num_modes)
        term = Kd @ np.eye(2 * len(rows)) @ Kd.conj().T
        total = term if total is None else total + term
    return total


def _plain_adjoint(X, num_channels, num_modes):
    return np.asarray(X).conj().T


def _metric_adjoint(X, num_channels, num_modes):
    return np.asarray(nambu_adjoint(X, num_channels, num_modes))


def _lyapunov_residual(H_bdg, channels, sigma_c, dagger):
    M = np.asarray(collective_dynamical_matrix(H_bdg, channels))
    D = _diffusion(channels, dagger)
    return np.linalg.norm(M @ sigma_c + sigma_c @ M.conj().T + D)


# ---------------------------------------------------------------------------
# THE GATE: single-mode DPO with one Bogoliubov uncontrolled channel
# ---------------------------------------------------------------------------

def _single_mode_dpo():
    """Coherent squeezing ``nu = i lambda``, a CONTROLLED particle-conserving
    drain, and an UNCONTROLLED Bogoliubov channel on vacuum."""
    g = np.array([[0.0]], dtype=complex)
    nu = np.array([[1j * LAMBDA]], dtype=complex)
    H_bdg = build_H_bdg(jnp.asarray(g), jnp.asarray(nu))
    channels = ChannelSet([
        Channel(u=[np.sqrt(KAPPA)], access=Access.CONTROLLED),
        Channel(u=[np.sqrt(GAMMA) * np.cosh(R)],
                v=[np.sqrt(GAMMA) * np.sinh(R)], access=Access.UNCONTROLLED),
    ], num_modes=1)
    return H_bdg, channels


def _single_mode_fock(cutoff=40):
    (a,) = _fock_operators(1, cutoff)
    ad = a.conj().T
    H = 0.5 * (1j * LAMBDA * ad @ ad + np.conj(1j * LAMBDA) * a @ a)
    jumps = [np.sqrt(KAPPA) * a,
             np.sqrt(GAMMA) * (np.cosh(R) * a + np.sinh(R) * ad)]
    rho = _lindblad_steady_state(H, jumps)
    return _intracavity_covariance(rho, [a])


def test_dagger_metric_placement_matches_the_lindblad_steady_state():
    """**The D2 gate.**  ``^ddag = sigma_z^sys ^H sigma_z^ch`` reproduces the
    exact Lindblad steady state through the Lyapunov equation."""
    H_bdg, channels = _single_mode_dpo()
    sigma_c = _single_mode_fock()
    residual = _lyapunov_residual(H_bdg, channels, sigma_c, _metric_adjoint)
    assert residual < 1e-9


def test_plain_adjoint_is_ruled_out():
    """Guard: dropping the ``sigma_z`` metrics is not a harmless simplification.

    It leaves the Lyapunov residual at O(1) -- the response would inject a bath
    squeezed along the wrong quadrature.
    """
    H_bdg, channels = _single_mode_dpo()
    sigma_c = _single_mode_fock()
    residual = _lyapunov_residual(H_bdg, channels, sigma_c, _plain_adjoint)
    assert residual > 1.0


def test_sum_rule_does_not_discriminate_between_the_two_placements():
    """Records *why* the gate is the Lyapunov check and not the sum rule.

    Both placements are pseudo-unitary; they differ only by the sign of the
    effective bath squeezing angle, and both are physical.  Anyone re-deriving
    ``^ddag`` from the sum rule alone will pick the wrong one half the time.
    """
    H_bdg, channels = _single_mode_dpo()
    assert scattering_sum_rule_residual(H_bdg, channels, 0.0) < 1e-9

    sz = np.asarray(pauli_z(1))
    M = np.asarray(collective_dynamical_matrix(H_bdg, channels))
    chi = np.linalg.inv(M)
    Kk = np.asarray(channels.K_kappa())
    Kg = np.asarray(channels.K_Gamma())
    S = np.eye(2) + Kk @ chi @ Kk.conj().T
    N = Kk @ chi @ Kg.conj().T
    naive = np.linalg.norm(S @ sz @ S.conj().T + N @ sz @ N.conj().T - sz)
    assert naive < 1e-9          # the wrong placement passes the sum rule too


@pytest.mark.parametrize("cutoff,bound", [(20, 1e-3), (30, 1e-6), (40, 1e-9)])
def test_gate_residual_converges_with_the_fock_cutoff(cutoff, bound):
    """The residual is truncation-limited, not model-limited: it goes to zero as
    the Fock cutoff grows, which is what makes the gate a real verification."""
    H_bdg, channels = _single_mode_dpo()
    sigma_c = _single_mode_fock(cutoff=cutoff)
    assert _lyapunov_residual(H_bdg, channels, sigma_c, _metric_adjoint) < bound


def test_channel_doubling_validated_on_a_collective_bogoliubov_channel():
    """Two modes, a coherent hop, private drains **and** a collective Bogoliubov
    uncontrolled channel with independent phases on ``u`` and ``v``.

    The single-mode gate cannot see the channel-space doubling (there
    ``sigma_z^ch == sigma_z^sys``).  This one can: the uncontrolled block has one
    channel against two modes, so the two metrics have different sizes and the
    assembly is only consistent if the doubling is right.
    """
    J, k1, k2, phi = 0.35, 0.5, 0.7, 0.8
    g = np.array([[0.0, J], [J, 0.0]], dtype=complex)
    H_bdg = build_H_bdg(jnp.asarray(g), jnp.asarray(np.zeros((2, 2), dtype=complex)))

    u = np.sqrt(GAMMA) * np.cosh(0.35) * np.array([1.0, np.exp(1j * phi)])
    v = np.sqrt(GAMMA) * np.sinh(0.35) * np.array([np.exp(-0.4j), 1.0])
    channels = ChannelSet([
        Channel(u=[np.sqrt(k1), 0.0], access=Access.CONTROLLED),
        Channel(u=[0.0, np.sqrt(k2)], access=Access.CONTROLLED),
        Channel(u=u, v=v, access=Access.UNCONTROLLED),
    ], num_modes=2)
    assert channels[-1].is_collective
    assert channels.num_uncontrolled == 1 and channels.num_modes == 2

    cutoff = 7
    a1, a2 = _fock_operators(2, cutoff)
    H = J * (a1.conj().T @ a2 + a2.conj().T @ a1)
    jumps = [np.sqrt(k1) * a1, np.sqrt(k2) * a2,
             u[0] * a1 + u[1] * a2 + v[0] * a1.conj().T + v[1] * a2.conj().T]
    sigma_c = _intracavity_covariance(_lindblad_steady_state(H, jumps), [a1, a2])

    metric = _lyapunov_residual(H_bdg, channels, sigma_c, _metric_adjoint)
    plain = _lyapunov_residual(H_bdg, channels, sigma_c, _plain_adjoint)
    assert metric < 1e-3         # truncation-limited at this small cutoff
    assert plain > 1.0


# ---------------------------------------------------------------------------
# D2 -- the collected / uncollected split
# ---------------------------------------------------------------------------

def _collected_only():
    """Every channel CONTROLLED and particle-conserving (Sec. 5 canonical form):
    the squeezing is generated coherently, in ``H_BdG``."""
    g = np.array([[0.0]], dtype=complex)
    nu = np.array([[1j * LAMBDA]], dtype=complex)
    H_bdg = build_H_bdg(jnp.asarray(g), jnp.asarray(nu))
    channels = ChannelSet(
        [Channel(u=[np.sqrt(KAPPA)], access=Access.CONTROLLED)], num_modes=1)
    return H_bdg, channels


def _uncollected_bogoliubov():
    """The squeezing-generating channel routed **out** of the monitored set."""
    g = np.array([[0.0]], dtype=complex)
    H_bdg = build_H_bdg(jnp.asarray(g), jnp.asarray(np.zeros((1, 1), dtype=complex)))
    channels = ChannelSet([
        Channel(u=[np.sqrt(KAPPA)], access=Access.CONTROLLED),
        Channel(u=[np.sqrt(GAMMA) * np.cosh(R)],
                v=[np.sqrt(GAMMA) * np.sinh(R)], access=Access.UNCONTROLLED),
    ], num_modes=1)
    return H_bdg, channels


def _quadrature(H_bdg, channels, Omega):
    n_ctrl = channels.num_controlled
    return np.real(np.asarray(collective_output_covariance_quadrature(
        H_bdg, channels, Omega, vacuum_covariance(n_ctrl), jnp.zeros(n_ctrl))))


@pytest.mark.parametrize("Omega", OMEGAS)
def test_collected_route_emits_a_pure_field(Omega):
    """``det sigma_out(Omega) = 1 +- 1e-9`` at every grid point: nothing routes
    through ``K_Gamma``, so the emitted field stays on the pure-state variety."""
    H_bdg, channels = _collected_only()
    assert channels.num_uncontrolled == 0
    V = _quadrature(H_bdg, channels, Omega)
    assert np.isclose(np.linalg.det(V), 1.0, atol=1e-9)
    assert np.allclose(symplectic_eigenvalues(V), 1.0, atol=1e-9)


def test_collected_route_matches_the_b1_closed_form():
    """``sigma_min(0) = ((kappa/2 - lambda)/(kappa/2 + lambda))^2``."""
    H_bdg, channels = _collected_only()
    V = _quadrature(H_bdg, channels, 0.0)
    expected = ((KAPPA / 2 - LAMBDA) / (KAPPA / 2 + LAMBDA)) ** 2
    assert np.isclose(V[1, 1], expected, rtol=1e-9)
    assert np.isclose(V[0, 0], 1.0 / expected, rtol=1e-9)


@pytest.mark.parametrize("Omega", OMEGAS)
def test_uncollected_bogoliubov_route_leaves_the_pure_variety(Omega):
    """``det sigma_out(Omega) > 1``: the squeezing is generated by a channel the
    detector does not own, so its bundled bath noise lands in the output."""
    H_bdg, channels = _uncollected_bogoliubov()
    assert channels.num_uncontrolled == 1
    assert channels.uncontrolled[0].channel_class.value == "bogoliubov"
    V = _quadrature(H_bdg, channels, Omega)
    assert np.linalg.det(V) > 1.0 + 1e-6
    assert np.all(symplectic_eigenvalues(V) > 1.0 + 1e-9)


def test_uncollected_bogoliubov_still_squeezes_the_output():
    """It is a genuine squeezing resource, just an impure one -- the B.8-imp
    behaviour the addendum points at."""
    H_bdg, channels = _uncollected_bogoliubov()
    V = _quadrature(H_bdg, channels, 0.0)
    assert np.min(np.linalg.eigvalsh(V)) < 1.0        # squeezed below vacuum
    assert np.linalg.det(V) > 1.0                     # but not pure


@pytest.mark.parametrize("Omega", OMEGAS)
def test_output_is_a_bona_fide_covariance(Omega):
    """Physicality of the Bogoliubov path: symplectic eigenvalues >= 1."""
    for build in (_collected_only, _uncollected_bogoliubov, _single_mode_dpo):
        H_bdg, channels = build()
        V = _quadrature(H_bdg, channels, Omega)
        assert np.min(symplectic_eigenvalues(V)) > 1.0 - 1e-9


@pytest.mark.parametrize("Omega", OMEGAS)
def test_sum_rule_holds_on_the_bogoliubov_path(Omega):
    for build in (_collected_only, _uncollected_bogoliubov, _single_mode_dpo):
        H_bdg, channels = build()
        assert scattering_sum_rule_residual(H_bdg, channels, Omega) < 1e-9


def test_single_mode_bogoliubov_squeezes_through_diffusion_not_drift():
    """Sec. 2, table row 2: a *single-mode* Bogoliubov jump has ``B_diss = 0``.

    Its anomalous content cancels in the drift (``V^H U`` and ``U^H V`` are the
    same scalar), so the drift is indistinguishable from a plain drain of rate
    ``|u|^2 - |v|^2``.  The squeezing it produces lives entirely in the
    **diffusion** -- which is precisely why the ``^ddag`` gate above had to be a
    Lyapunov/steady-state test and could not be read off ``M``.
    """
    _, collected = _collected_only()
    _, bogoliubov = _uncollected_bogoliubov()
    assert np.allclose(np.asarray(collected.dissipative_drift())[:1, 1:], 0.0, atol=1e-14)
    assert np.allclose(np.asarray(bogoliubov.dissipative_drift())[:1, 1:], 0.0, atol=1e-14)
    # ...yet the emitted field is squeezed: the effect is all in D_eff
    V = _quadrature(*_uncollected_bogoliubov(), 0.0)
    assert np.min(np.linalg.eigvalsh(V)) < 1.0


def test_two_mode_bogoliubov_does_feed_the_anomalous_block():
    """Sec. 2, table row 3: ``B_12 != 0`` once the jump spans two modes, and the
    partner mode is anti-damped (``A_22 > 0``)."""
    r = 0.45
    channels = ChannelSet([
        Channel(u=[np.cosh(r), 0.0], v=[0.0, np.sinh(r)],
                access=Access.UNCONTROLLED),
    ], num_modes=2)
    M_diss = np.asarray(channels.dissipative_drift())
    assert abs(M_diss[0, 3]) > 1e-6                       # B_diss entry
    assert M_diss[1, 1].real > 0.0                        # anti-damped partner


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
