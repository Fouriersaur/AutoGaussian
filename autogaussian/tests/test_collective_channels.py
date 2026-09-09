"""D0 -- collective dissipative channels: reduction guard (Addendum Sec. 8).

The prime invariant of the addendum is that private channels are the *diagonal
corner* of the collective search space, not a separate code path.  This file is
the regression guard for that claim:

  * the Sec. 2 drift assembly reproduces the addendum's three verified sub-cases;
  * the Sec. 5 canonical form is enforced at construction;
  * the collective forward map reproduces the scalar (kappa-rescaled) one
    *identically* on the private corner -- for ``M``, ``S``, ``N`` and
    ``sigma_out(Omega)``, across devices with squeezing, complex phases,
    intrinsic loss and non-unit decay ratios;
  * the App. F golden numbers come back through the collective path;
  * ``kappa~`` stays live: perturbing a support-1 controlled amplitude moves
    ``sigma_out(Omega != 0)`` and leaves ``sigma_out(0)`` alone.

If any of this goes red the generalisation is wrong and nothing downstream
(D1-D4) may be trusted.
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
    ChannelClass,
    ChannelSet,
    dissipative_drift,
    nambu_adjoint,
    private_channel_set,
)
from autogaussian.forward import (
    collective_dynamical_matrix,
    collective_output_covariance_quadrature,
    collective_response_matrices,
    dynamical_matrix,
    output_covariance_quadrature,
    response_matrices,
    scattering_sum_rule_residual,
    unrescale_hamiltonian,
)
from autogaussian.nambu import build_H_bdg, duan_sum, vacuum_covariance

TOL = 1e-12


# ---------------------------------------------------------------------------
# devices exercised by the reduction test
# ---------------------------------------------------------------------------

def _device(g, nu, gamma=None, kappa_tilde=None):
    g = jnp.asarray(np.atleast_2d(np.asarray(g, dtype=complex)))
    nu = jnp.asarray(np.atleast_2d(np.asarray(nu, dtype=complex)))
    n = g.shape[0]
    gamma = jnp.zeros(n) if gamma is None else jnp.asarray(np.asarray(gamma, float))
    kappa_tilde = (jnp.ones(n) if kappa_tilde is None
                   else jnp.asarray(np.asarray(kappa_tilde, float)))
    return build_H_bdg(g, nu), gamma, kappa_tilde


LAMBDA = 0.3          # the App. F operating point: ((1/2-l)/(1/2+l))^2 = 0.0625

DEVICES = {
    # single-mode squeezer (App. F row 1)
    "single_mode_squeezer": _device([[0.0]], [[1j * LAMBDA]]),
    # EPR pair (App. F row 2)
    "epr_pair": _device(np.zeros((2, 2)),
                        [[0.0, 1j * LAMBDA], [1j * LAMBDA, 0.0]]),
    # reciprocal squeezing: BS edge + squeezing edge (App. F row 3)
    "reciprocal": _device([[0.0, 0.4485284280936455], [0.4485284280936455, 0.0]],
                          [[0.0, 0.28822742474916385],
                           [0.28822742474916385, 0.0]]),
    # lossy, detuned, complex-phase two-mode device with unequal decay ratios
    "lossy_detuned": _device([[0.2, 0.5 + 0.15j], [0.5 - 0.15j, -0.1]],
                             [[0.12j, 0.25 * np.exp(0.7j)],
                              [0.25 * np.exp(0.7j), 0.0]],
                             gamma=[0.13, 0.41], kappa_tilde=[0.4, 2.5]),
    # three modes with phases and a damped auxiliary -- the structure the App. F
    # directional entry uses (2 signal + 1 damped aux, phases on every edge)
    "directional_like": _device(
        [[0.05, 0.6 * np.exp(0.9j), 0.35],
         [0.6 * np.exp(-0.9j), -0.08, 0.42 * np.exp(-0.3j)],
         [0.35, 0.42 * np.exp(0.3j), 0.0]],
        [[0.0, 0.18j, 0.0], [0.18j, 0.0, 0.0], [0.0, 0.0, 0.0]],
        gamma=[0.0, 0.0, 1.7], kappa_tilde=[1.0, 1.0, 3.0]),
}

OMEGAS = (0.0, 0.37, -1.2, 4.5)


def _collective(device):
    """The private channel set + unrescaled BdG matrix for a scalar device."""
    H, gamma, kappa_tilde = device
    channels = private_channel_set(gamma, kappa_tilde, include_lossless=True)
    return unrescale_hamiltonian(H, kappa_tilde), channels


# ---------------------------------------------------------------------------
# Sec. 5 -- canonical form, enforced at construction
# ---------------------------------------------------------------------------

def test_collected_bogoliubov_is_rejected_at_construction():
    """CONTROLLED => PARTICLE_CONSERVING (Addendum Sec. 5).

    A collected Bogoliubov jump is degenerate with a plain drain on a squeezed
    input; the two are distinct graph+bath specs with identical
    ``sigma_out(Omega)``, so allowing both would corrupt every minimality claim.
    """
    with pytest.raises(ValueError, match="canonical form"):
        Channel(u=[1.0, 0.0], v=[0.3, 0.0], access=Access.CONTROLLED)


def test_uncontrolled_bogoliubov_is_allowed():
    """Anomalous jumps are permitted exactly where they are honestly loss."""
    channel = Channel(u=[1.0, 0.0], v=[0.3, 0.0], access=Access.UNCONTROLLED)
    assert channel.channel_class is ChannelClass.BOGOLIUBOV
    assert channel.access is Access.UNCONTROLLED


def test_support_classification():
    private = Channel(u=[0.0, 1.0], access=Access.CONTROLLED)
    collective = Channel(u=[1.0, 1.0], access=Access.CONTROLLED)
    assert private.support == (1,) and private.is_private
    assert collective.support == (0, 1) and collective.is_collective
    assert private.channel_class is ChannelClass.PARTICLE_CONSERVING


# ---------------------------------------------------------------------------
# Sec. 2 -- the three verified drift sub-cases
# ---------------------------------------------------------------------------

def test_drift_subcase_private_particle_conserving():
    """``sqrt(kappa) a_i`` -> ``A_diss = -kappa_i/2`` on ``(i,i)``, ``B_diss = 0``."""
    kappa = 0.7
    U = np.array([[0.0, np.sqrt(kappa)]])
    V = np.zeros_like(U)
    M_diss = np.asarray(dissipative_drift(U, V))
    expected = np.zeros((4, 4), dtype=complex)
    expected[1, 1] = expected[3, 3] = -0.5 * kappa
    assert np.allclose(M_diss, expected, atol=TOL)


def test_drift_subcase_single_mode_bogoliubov():
    """``cosh r a + sinh r a^dag`` -> ``A_diss = -1/2``: the squeezing lives in
    the diffusion, not in the drift, and the damping rate is 1 (not cosh 2r)."""
    r = 0.8
    U = np.array([[np.cosh(r)]])
    V = np.array([[np.sinh(r)]])
    M_diss = np.asarray(dissipative_drift(U, V))
    assert np.isclose(M_diss[0, 0], -0.5, atol=TOL)
    assert np.isclose(M_diss[1, 1], -0.5, atol=TOL)
    assert np.allclose(M_diss[0, 1], 0.0, atol=TOL)


def test_drift_subcase_two_mode_bogoliubov_antidamps_partner():
    """``cosh r a_1 + sinh r a_2^dag`` -> ``A_11 = -cosh^2 r``/2,
    ``A_22 = +sinh^2 r``/2 (mode 2 is **anti-damped**), ``B_12 != 0``.

    **Refinement of Addendum Sec. 2/Sec. 6 (verified here).**  The addendum says
    a lone two-mode-squeezing jump "destabilizes on its own".  The diagonal
    anti-damping ``A_22 > 0`` is real, but the full ``M_diss`` spectrum is
    ``{-1/2, -1/2, 0, 0}``: the jump is **marginally** stable, not exponentially
    unstable.  The operational conclusion is unchanged -- it is not Hurwitz, so
    it cannot furnish a stable witness and still has to be balanced by collected
    loss on the partner mode -- but the margin it costs is a zero eigenvalue,
    not a positive one.  Assert the honest version.
    """
    r = 0.6
    U = np.array([[np.cosh(r), 0.0]])
    V = np.array([[0.0, np.sinh(r)]])
    M_diss = np.asarray(dissipative_drift(U, V))
    A = M_diss[:2, :2]
    B = M_diss[:2, 2:]
    assert np.isclose(A[0, 0], -0.5 * np.cosh(r) ** 2, atol=TOL)
    assert np.isclose(A[1, 1], +0.5 * np.sinh(r) ** 2, atol=TOL)
    assert abs(B[0, 1]) > 1e-3
    abscissa = np.max(np.real(np.linalg.eigvals(M_diss)))
    assert abscissa >= -1e-12          # not Hurwitz: no stability margin at all
    assert abscissa <= 1e-12           # but only marginally so
    spectrum = np.sort(np.real(np.linalg.eigvals(M_diss)))
    assert np.allclose(spectrum, [-0.5, -0.5, 0.0, 0.0], atol=1e-9)


def test_particle_conserving_drift_is_negative_semidefinite():
    """``V = 0`` -> ``A_diss = -U^H U /2 <= 0``: collective particle-conserving
    channels can only move eigenvalues left (Addendum Sec. 6)."""
    rng = np.random.default_rng(0)
    U = rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))
    M_diss = np.asarray(dissipative_drift(U, np.zeros_like(U)))
    A = M_diss[:4, :4]
    assert np.allclose(A, np.conj(A).T, atol=1e-12)          # Hermitian
    assert np.max(np.linalg.eigvalsh(A)) <= 1e-12            # negative semidefinite
    assert np.allclose(M_diss[:4, 4:], 0.0, atol=TOL)        # B_diss = 0


def test_collective_particle_conserving_couples_shared_modes():
    """``(U^H U)_{jk} != 0`` iff one channel touches both j and k -- what makes
    a collective channel a *coupling* and not merely a rate."""
    shared = np.asarray(dissipative_drift(np.array([[1.0, 1.0]]), np.zeros((1, 2))))
    separate = np.asarray(dissipative_drift(np.eye(2), np.zeros((2, 2))))
    assert abs(shared[0, 1]) > 1e-9
    assert abs(separate[0, 1]) < TOL


# ---------------------------------------------------------------------------
# Sec. 3 -- the ^ddag adjoint collapses to ^H on the particle-conserving path
# ---------------------------------------------------------------------------

def test_nambu_adjoint_collapses_for_particle_conserving():
    """The Sec. 3 FLAG's safe half: for ``V = 0`` the two sigma_z metrics cancel
    and ``K^ddag == K^H``.  (The Bogoliubov placement stays provisional until D2.)"""
    rng = np.random.default_rng(1)
    U = rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))
    channels = ChannelSet(
        [Channel(u=U[mu], access=Access.CONTROLLED) for mu in range(3)], num_modes=4)
    K = channels.K_kappa()
    assert np.allclose(np.asarray(nambu_adjoint(K, 3, 4)),
                       np.asarray(jnp.conj(K).T), atol=TOL)


@pytest.mark.parametrize("name", list(DEVICES))
@pytest.mark.parametrize("Omega", OMEGAS)
def test_scattering_sum_rule(name, Omega):
    """``S sigma_z S^H + N sigma_z N^H = sigma_z`` -- the routing identity behind
    "purity is a routing condition, not a dark state" (Addendum Sec. 3).

    Note the **ordinary** adjoint.  The addendum writes this rule with ``^ddag``
    while also defining ``X^ddag = Sigma_z X^H Sigma_z``; the two together
    collapse to ``S S^H + N N^H = I``, which the existing private forward map
    violates by O(1).  See :func:`scattering_sum_rule_residual`.
    """
    H_bdg, channels = _collective(DEVICES[name])
    assert scattering_sum_rule_residual(H_bdg, channels, Omega) < 1e-9


def test_addendum_sum_rule_as_literally_written_is_false():
    """Guard against "fixing" the sum rule back to the addendum's wording.

    ``X^ddag = Sigma_z X^H Sigma_z`` substituted into
    ``S sigma_z S^ddag + N sigma_z N^ddag = sigma_z`` gives ``S S^H + N N^H = I``.
    That is *not* an identity of this forward map -- it fails by O(1) on the
    lossy two-mode device.  Recorded so the correction is not silently reverted.
    """
    H_bdg, channels = _collective(DEVICES["lossy_detuned"])
    S, N = collective_response_matrices(H_bdg, channels, 0.0)
    S, N = np.asarray(S), np.asarray(N)
    naive = np.linalg.norm(S @ S.conj().T + N @ N.conj().T - np.eye(S.shape[0]))
    assert naive > 1.0
    assert scattering_sum_rule_residual(H_bdg, channels, 0.0) < 1e-9


# ---------------------------------------------------------------------------
# D0 -- the reduction guard proper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(DEVICES))
def test_private_reduction_drift(name):
    """``M`` from the private channel set == the rescaled ``dynamical_matrix``."""
    device = DEVICES[name]
    H_bdg, channels = _collective(device)
    assert channels.all_private
    assert np.allclose(np.asarray(collective_dynamical_matrix(H_bdg, channels)),
                       np.asarray(dynamical_matrix(*device)), atol=TOL)


@pytest.mark.parametrize("name", list(DEVICES))
@pytest.mark.parametrize("Omega", OMEGAS)
def test_private_reduction_response(name, Omega):
    """``S`` and ``N`` reproduce the scalar path.

    Column order in ``N`` matches because :func:`private_channel_set` emits the
    uncontrolled channels in mode order (``include_lossless=True`` keeps the
    zero-rate ones so the shapes line up with the scalar ``sqrt(gamma)``).
    """
    device = DEVICES[name]
    H_bdg, channels = _collective(device)
    S_ref, N_ref = response_matrices(*device, Omega)
    S, N = collective_response_matrices(H_bdg, channels, Omega)
    assert np.allclose(np.asarray(S), np.asarray(S_ref), atol=1e-10)
    assert np.allclose(np.asarray(N), np.asarray(N_ref), atol=1e-10)


@pytest.mark.parametrize("name", list(DEVICES))
@pytest.mark.parametrize("Omega", OMEGAS)
def test_private_reduction_output_covariance(name, Omega):
    """``sigma_out(Omega)`` agrees, including the gauge rotation and the port
    restriction -- the end of the Sec. 3 chain."""
    device = DEVICES[name]
    H, _, kappa_tilde = device
    num_modes = H.shape[0] // 2
    H_bdg, channels = _collective(device)
    thetas = jnp.asarray(np.linspace(0.0, 0.9, num_modes))
    sigma_sig = vacuum_covariance(num_modes)

    reference = output_covariance_quadrature(
        *device, Omega, sigma_sig, thetas, num_ports=num_modes)
    collective = collective_output_covariance_quadrature(
        H_bdg, channels, Omega, sigma_sig, thetas, num_ports=num_modes)
    assert np.allclose(np.asarray(collective), np.asarray(reference), atol=1e-10)


def _collective_quadrature(name, Omega=0.0, sigma_sig=None):
    device = DEVICES[name]
    num_modes = device[0].shape[0] // 2
    H_bdg, channels = _collective(device)
    if sigma_sig is None:
        sigma_sig = vacuum_covariance(num_modes)
    return np.real(np.asarray(collective_output_covariance_quadrature(
        H_bdg, channels, Omega, sigma_sig, jnp.zeros(num_modes),
        num_ports=num_modes)))


# ---------------------------------------------------------------------------
# D0 -- the App. F golden numbers, reached through the collective path
# ---------------------------------------------------------------------------

def test_app_f_single_mode_squeezing():
    """12 dB: ``sigma_min(0) = ((kappa/2 - lambda)/(kappa/2 + lambda))^2 = 0.0625``."""
    V = _collective_quadrature("single_mode_squeezer")
    expected = ((0.5 - LAMBDA) / (0.5 + LAMBDA)) ** 2
    assert np.isclose(expected, 0.0625, atol=1e-9)
    assert np.isclose(V[1, 1], expected, rtol=1e-9)
    assert np.isclose(V[0, 0], 1.0 / expected, rtol=1e-9)


def test_app_f_epr_joint_and_local():
    """Joint quadratures ~0.062, local variances ~8.0, Duan sum well under the
    separability bound."""
    V = _collective_quadrature("epr_pair")
    joint_x = 0.5 * (V[0, 0] + V[2, 2] - 2 * V[0, 2])
    joint_p = 0.5 * (V[1, 1] + V[3, 3] + 2 * V[1, 3])
    assert np.isclose(joint_x, 0.0625, rtol=1e-6)
    assert np.isclose(joint_p, 0.0625, rtol=1e-6)
    assert np.isclose(V[0, 0], 8.0, rtol=5e-3)
    assert np.isclose(V[2, 2], 8.0, rtol=5e-3)
    assert duan_sum(V, 0, 1) < 4.0


def test_app_f_reciprocal_squeezing_at_both_ports():
    """5 dB (~0.315) at **both** ports of the BS + squeezing two-mode device."""
    V = _collective_quadrature("reciprocal")
    port0 = np.linalg.eigvalsh(V[:2, :2])
    port1 = np.linalg.eigvalsh(V[2:, 2:])
    assert np.isclose(port0[0], port1[0], rtol=1e-9)
    assert np.isclose(port0[0], 0.315, atol=5e-3)


def test_app_f_loss_degrades_and_aux_stays_near_vacuum():
    """The directional-style three-mode device: the heavily damped auxiliary
    reads close to vacuum while the signal ports do not."""
    V = _collective_quadrature("directional_like")
    aux = np.linalg.eigvalsh(V[4:, 4:])
    signal = np.linalg.eigvalsh(V[:2, :2])
    assert np.allclose(aux, 1.0, atol=0.2)
    assert not np.allclose(signal, 1.0, atol=0.2)


# ---------------------------------------------------------------------------
# D0 -- kappa~ liveness survives the promotion to channel amplitudes
# ---------------------------------------------------------------------------

def test_kappa_liveness_preserved_under_collective_map():
    """Addendum Sec. 4/Sec. 8: a support-1 CONTROLLED amplitude *is*
    ``sqrt(kappa~_i)``.  Perturbing it must move ``sigma_out(Omega != 0)`` and
    leave ``sigma_out(0)`` invariant to ~1e-9 -- the base spec's defining
    "``kappa~`` is live" property, restated in the channel language."""
    H, gamma, kappa_tilde = DEVICES["lossy_detuned"]
    num_modes = 2
    thetas = jnp.zeros(num_modes)
    sigma_sig = vacuum_covariance(num_modes)

    def out(kappa_vec, Omega):
        kappa_vec = jnp.asarray(kappa_vec)
        channels = private_channel_set(gamma, kappa_vec, include_lossless=True)
        H_bdg = unrescale_hamiltonian(H, kappa_vec)
        return np.real(np.asarray(collective_output_covariance_quadrature(
            H_bdg, channels, Omega, sigma_sig, thetas, num_ports=num_modes)))

    base = np.asarray(kappa_tilde)
    perturbed = base * np.array([1.0, 1.6])

    assert np.allclose(out(base, 0.0), out(perturbed, 0.0), atol=1e-9)
    assert not np.allclose(out(base, 0.6), out(perturbed, 0.6), atol=1e-3)


def test_private_controlled_amplitude_is_sqrt_kappa():
    """The identification is literal, not merely equivalent."""
    channels = private_channel_set(gamma=[0.25, 0.0], kappa_tilde=[0.4, 2.5],
                                   include_lossless=True)
    controlled = channels.controlled
    assert np.isclose(controlled[0].u[0], np.sqrt(0.4))
    assert np.isclose(controlled[1].u[1], np.sqrt(2.5))
    uncontrolled = channels.uncontrolled
    assert np.isclose(uncontrolled[0].u[0], np.sqrt(0.25 * 0.4))
    assert np.isclose(uncontrolled[1].u[1], 0.0)


def test_zero_rate_uncontrolled_channels_are_inert():
    """Dropping a zero-amplitude loss row changes nothing -- the continuity that
    the Sec. 7 monotonicity argument leans on."""
    H, gamma, kappa_tilde = DEVICES["single_mode_squeezer"]
    H_bdg = unrescale_hamiltonian(H, kappa_tilde)
    with_row = private_channel_set(gamma, kappa_tilde, include_lossless=True)
    without_row = private_channel_set(gamma, kappa_tilde, include_lossless=False)
    assert with_row.num_uncontrolled == 1 and without_row.num_uncontrolled == 0
    sigma_sig = vacuum_covariance(1)
    a = collective_output_covariance_quadrature(
        H_bdg, with_row, 0.31, sigma_sig, jnp.zeros(1), num_ports=1)
    b = collective_output_covariance_quadrature(
        H_bdg, without_row, 0.31, sigma_sig, jnp.zeros(1), num_ports=1)
    assert np.allclose(np.asarray(a), np.asarray(b), atol=TOL)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
