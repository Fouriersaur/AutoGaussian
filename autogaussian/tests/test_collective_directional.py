"""D1 -- particle-conserving collective channels + native directionality
(Addendum Sec. 8, D1).

The cell: two modes, one real coherent hop ``J``, one shared
``PARTICLE_CONSERVING CONTROLLED`` channel ``L = sqrt(gamma)(a_0 + e^{i phi} a_1)``.
At ``phi = -pi/2`` and ``J = gamma/2`` the drift becomes one-way, and the
*output* field is directional -- with **no synthetic flux** (every coherent
coupling is real) and **no auxiliary mode** (the App. F directional entry needs
2 signal modes plus a damped auxiliary; this needs 2 modes total).

Two notes on what the addendum leaves implicit.

**Phase sign.**  The addendum quotes ``phi = pi/2  ->  M_01 ~ 0, M_10 ~ -i gamma``.
Under this package's conventions (``M = -i sigma_z H_BdG + M_diss``,
``A_diss = -U^H U / 2``, row index = output mode) the phase that produces exactly
those entries is ``phi = -pi/2``; ``phi = +pi/2`` gives the mirror cell, one-way
in the other direction.  Both are asserted below, so the test is
convention-free: whichever sign convention a reader brings, one of the two
phases kills each direction.

**Readout.**  A single collective channel is a *single* input/output port, so
"``sigma_out`` transports forward but not backward" is not expressible on it
alone -- transport is a statement about two ports.  The output-level assertions
therefore add one private collected channel per mode as the readout.  That is
free: private channels contribute only ``-kappa/2`` on the diagonal of
``A_diss``, so they cannot disturb the off-diagonal one-way condition -- which
is itself asserted below rather than assumed.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp

from autogaussian.channels import Access, Channel, ChannelSet
from autogaussian.forward import (
    collective_dynamical_matrix,
    collective_is_stable,
    collective_output_covariance_quadrature,
    collective_response_matrices,
    scattering_sum_rule_residual,
)
from autogaussian.nambu import build_H_bdg, channel_covariance, vacuum_covariance

GAMMA = 1.0
J_TUNED = GAMMA / 2.0
PHI_TUNED = -np.pi / 2.0          # see the module docstring on the sign
OMEGAS = (0.0, 0.4, -1.3, 5.0)


def cell(J=J_TUNED, gamma=GAMMA, phi=PHI_TUNED, kappa_port=0.0):
    """The D1 cell.  ``kappa_port > 0`` adds the private readout ports.

    Channel order (= output-port order): private port 0, private port 1, then
    the shared collective channel.  The coherent block carries a **real** hop
    only -- no synthetic flux anywhere.
    """
    g = np.array([[0.0, J], [J, 0.0]], dtype=complex)
    nu = np.zeros((2, 2), dtype=complex)
    H_bdg = build_H_bdg(jnp.asarray(g), jnp.asarray(nu))

    channels = []
    if kappa_port > 0.0:
        root = np.sqrt(kappa_port)
        channels.append(Channel(u=[root, 0.0], access=Access.CONTROLLED))
        channels.append(Channel(u=[0.0, root], access=Access.CONTROLLED))
    shared = np.sqrt(gamma) * np.array([1.0, np.exp(1j * phi)])
    channels.append(Channel(u=shared, access=Access.CONTROLLED))
    return H_bdg, ChannelSet(channels, num_modes=2)


def normal_block(M):
    return np.asarray(M)[:2, :2]


# ---------------------------------------------------------------------------
# the shared channel is genuinely collective and genuinely particle-conserving
# ---------------------------------------------------------------------------

def test_shared_channel_is_collective_and_particle_conserving():
    _, channels = cell()
    shared = channels[-1]
    assert shared.is_collective and shared.support == (0, 1)
    assert shared.channel_class.value == "particle_conserving"
    assert shared.access is Access.CONTROLLED
    # a collective channel is a coupling, not a rate: it feeds the off-diagonal
    M_diss = np.asarray(channels.dissipative_drift())
    assert abs(M_diss[0, 1]) > 1e-9
    assert np.allclose(M_diss[:2, 2:], 0.0, atol=1e-14)      # B_diss = 0


# ---------------------------------------------------------------------------
# D1(a) -- the one-way drift
# ---------------------------------------------------------------------------

def test_one_way_drift_at_the_tuning_point():
    """``M_01 = 0``, ``M_10 = -i gamma`` at ``phi = -pi/2``, ``J = gamma/2``."""
    H_bdg, channels = cell()
    A = normal_block(collective_dynamical_matrix(H_bdg, channels))
    assert np.isclose(A[0, 1], 0.0, atol=1e-14)
    assert np.isclose(A[1, 0], -1j * GAMMA, atol=1e-14)
    assert np.isclose(A[0, 0], -0.5 * GAMMA, atol=1e-14)
    assert np.isclose(A[1, 1], -0.5 * GAMMA, atol=1e-14)


def test_opposite_phase_reverses_the_one_way_direction():
    """Sign-convention-free form of the claim: ``+pi/2`` is the mirror cell."""
    H_bdg, channels = cell(phi=+np.pi / 2)
    A = normal_block(collective_dynamical_matrix(H_bdg, channels))
    assert np.isclose(A[1, 0], 0.0, atol=1e-14)
    assert np.isclose(A[0, 1], -1j * GAMMA, atol=1e-14)


@pytest.mark.parametrize("J,phi", [
    (J_TUNED, 0.0),                 # no phase -> reciprocal
    (J_TUNED, -np.pi / 4),           # wrong phase
    (0.8 * J_TUNED, PHI_TUNED),      # wrong hop
    (1.4 * J_TUNED, PHI_TUNED),
])
def test_detuning_away_from_the_tuning_point_restores_reciprocity(J, phi):
    """Both conditions are needed: ``J = gamma/2`` *and* ``phi = -pi/2``."""
    H_bdg, channels = cell(J=J, phi=phi)
    A = normal_block(collective_dynamical_matrix(H_bdg, channels))
    assert abs(A[0, 1]) > 1e-6


def test_no_synthetic_flux_and_no_auxiliary_mode():
    """The coherent graph is a single **real** hop on **two** modes.

    The App. F directional entry buys the same asymmetry with 2 signal modes
    plus a damped auxiliary and complex (flux-carrying) couplings; here the
    asymmetry comes from the dissipative colour instead.
    """
    H_bdg, channels = cell(kappa_port=0.2)
    assert channels.num_modes == 2
    g = np.asarray(H_bdg)[:2, :2]
    assert np.allclose(g.imag, 0.0, atol=1e-14)              # no synthetic flux
    assert np.allclose(np.asarray(H_bdg)[:2, 2:], 0.0, atol=1e-14)   # no squeezing


def test_particle_conserving_collective_only_moves_eigenvalues_left():
    """``A_diss = -U^H U /2 <= 0`` is unconditionally stabilising (Sec. 6)."""
    H_bdg, channels = cell(kappa_port=0.2)
    assert collective_is_stable(H_bdg, channels)
    bare = np.max(np.real(np.linalg.eigvals(
        np.asarray(collective_dynamical_matrix(H_bdg, ChannelSet(
            [c for c in channels if c.is_private], num_modes=2))))))
    full = np.max(np.real(np.linalg.eigvals(
        np.asarray(collective_dynamical_matrix(H_bdg, channels)))))
    assert full <= bare + 1e-12


# ---------------------------------------------------------------------------
# D1(b) -- directionality of the *output* field
# ---------------------------------------------------------------------------

KAPPA_PORT = 0.2


@pytest.mark.parametrize("Omega", OMEGAS + (20.0,))
def test_scattering_is_one_way_at_every_frequency(Omega):
    """``S[0,1] = 0`` to machine precision at **every** ``Omega``, ``S[1,0] != 0``.

    The isolation is broadband, not a single-frequency coincidence: with
    ``B_diss = 0`` and no coherent squeezing the drift is block diagonal in
    Nambu space with a *triangular* normal block, so ``chi = inv(M + i Omega I)``
    inherits the triangularity at every frequency.

    The *transmission* is not flat -- ``|S[1,0]|`` rolls off roughly as
    ``1/Omega^2`` far off resonance (0.556 at ``Omega = 0`` down to 5.0e-4 at
    ``Omega = 20``).  What is frequency-independent is the **isolation ratio**,
    ``|S[1,0]| / |S[0,1]| ~ 3e16`` at every point on the grid, i.e. the backward
    amplitude is zero to numerical noise rather than merely small.  Assert the
    ratio, which is the actual claim, not an absolute forward floor.
    """
    H_bdg, channels = cell(kappa_port=KAPPA_PORT)
    S, _ = collective_response_matrices(H_bdg, channels, Omega)
    S = np.asarray(S)
    forward, backward = abs(S[1, 0]), abs(S[0, 1])
    assert backward < 1e-13                          # exactly blocked
    assert forward > 0.0                             # forward path exists
    assert forward / max(backward, 1e-300) > 1e10    # and dominates completely
    assert abs(S[0, 2]) > 1e-3 or abs(S[2, 0]) > 1e-3   # the shared port is live


def test_forward_transmission_rolls_off_while_isolation_stays_exact():
    """Separates the two frequency behaviours the previous test asserts jointly."""
    H_bdg, channels = cell(kappa_port=KAPPA_PORT)
    grid = [0.0, 0.4, 1.3, 5.0, 20.0]
    forward, backward = [], []
    for Omega in grid:
        S = np.asarray(collective_response_matrices(H_bdg, channels, Omega)[0])
        forward.append(abs(S[1, 0]))
        backward.append(abs(S[0, 1]))
    assert np.all(np.diff(forward) < 0.0)            # transmission rolls off
    assert forward[0] > 0.5 and forward[-1] < 1e-3
    assert max(backward) < 1e-13                     # isolation does not


@pytest.mark.parametrize("Omega", OMEGAS)
def test_output_covariance_transports_forward_but_not_backward(Omega):
    """The D1 claim at the ``sigma_out`` level.

    Drive one port with a thermal state and leave everything else in vacuum.
    Injecting at port 0 heats port 1; injecting at port 1 leaves port 0 exactly
    at the vacuum floor.
    """
    H_bdg, channels = cell(kappa_port=KAPPA_PORT)
    n_ctrl = channels.num_controlled
    thetas = jnp.zeros(n_ctrl)
    n_thermal = 2.0

    def out(driven_port):
        occupation = np.zeros(n_ctrl)
        occupation[driven_port] = n_thermal
        sigma_sig = channel_covariance(n=occupation, m=0.0, num_modes=n_ctrl)
        return np.real(np.asarray(collective_output_covariance_quadrature(
            H_bdg, channels, Omega, sigma_sig, thetas, num_ports=2)))

    forward = out(0)          # drive port 0, read port 1
    backward = out(1)         # drive port 1, read port 0

    # forward: port 1 is heated above the vacuum floor.  The excess tracks the
    # |S[1,0]|^2 roll-off, so the floor is scaled rather than fixed.
    S = np.asarray(collective_response_matrices(H_bdg, channels, Omega)[0])
    excess = 2.0 * n_thermal * abs(S[1, 0]) ** 2
    assert forward[2, 2] > 1.0 + 0.5 * excess
    assert forward[3, 3] > 1.0 + 0.5 * excess
    assert excess > 0.0

    # backward: port 0 does not move off the vacuum floor at all, at any Omega
    # and for any drive strength -- the backward amplitude is identically zero.
    assert np.allclose(backward[:2, :2], np.eye(2), atol=1e-12)


def test_reciprocal_cell_transports_both_ways():
    """Control: switch the phase off and the same graph is reciprocal."""
    H_bdg, channels = cell(phi=0.0, kappa_port=KAPPA_PORT)
    S = np.asarray(collective_response_matrices(H_bdg, channels, 0.0)[0])
    assert abs(S[0, 1]) > 1e-2
    assert abs(S[1, 0]) > 1e-2
    assert np.isclose(abs(S[0, 1]), abs(S[1, 0]), rtol=1e-9)


@pytest.mark.parametrize("Omega", OMEGAS)
def test_sum_rule_holds_for_the_collective_cell(Omega):
    """The Sec. 3 routing identity survives a genuinely collective channel."""
    H_bdg, channels = cell(kappa_port=KAPPA_PORT)
    assert scattering_sum_rule_residual(H_bdg, channels, Omega) < 1e-9


def test_all_controlled_cell_emits_a_pure_field():
    """Every channel is CONTROLLED, so nothing routes through ``K_Gamma``: the
    emitted field stays pure at every frequency (Addendum Sec. 3)."""
    from autogaussian.nambu import symplectic_eigenvalues

    H_bdg, channels = cell(kappa_port=KAPPA_PORT)
    assert channels.num_uncontrolled == 0
    n_ctrl = channels.num_controlled
    for Omega in OMEGAS:
        V = np.real(np.asarray(collective_output_covariance_quadrature(
            H_bdg, channels, Omega, vacuum_covariance(n_ctrl), jnp.zeros(n_ctrl))))
        assert np.allclose(symplectic_eigenvalues(V), 1.0, atol=1e-9)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
