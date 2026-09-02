"""
FORWARD MAP: graph -> sigma_out(Omega)   (Sec. 3 and App. A of the algorithm flow).

    Step 1   D(Omega) = -i sigma_z H - (1 + gamma)/2 + i Omega kappa~^{-1}
    Step 2   D^{-1}                                   (susceptibility, rescaled)
    Step 3   S = 1 + D^{-1}          (scattering, AUTOSCATTER's S)
             N = D^{-1} sqrt(gamma)  (noise response of the intrinsic-loss channels)
             S_cal = [ S | N ]
    Step 4   sigma_out = S_cal sigma_in S_cal^dag  -> rotate to quadratures, keep ports

All quantities are dimensionless (App. A.4):  the raw ``g, nu, Delta, kappa,
Gamma`` collapse into ``{C_ij, Phi, Delta_i/kappa_i, gamma_i, kappa~_i}``.

Note on the frequency convention: the second Nambu block is the Fourier
transform of ``a^dag(t)``, i.e. ``[a(-Omega)]^dag``.  With that (standard BdG)
convention the doubled input-output relation holds at a single ``Omega`` and
``sigma_out(Omega)`` is the symmetrised output spectrum.
"""

import numpy as np
import jax
import jax.numpy as jnp

from autogaussian.nambu import (
    pauli_z,
    nambu_to_quadrature,
    quadrature_matrix,
    stack_input_covariance,
)

__all__ = [
    "inverse_susceptibility",
    "response_matrices",
    "full_response",
    "output_covariance_nambu",
    "output_covariance_quadrature",
    "noise_response_block",
    "noise_response_amplitude",
    "dynamical_matrix",
    "max_real_eigenvalue",
    "is_stable",
]


def _double(vec):
    """Replicate a per-mode vector over the two Nambu blocks."""
    return jnp.concatenate([vec, vec])


def inverse_susceptibility(H, gamma, kappa_tilde, Omega):
    """``D(Omega) = -i sigma_z H - (1 + gamma)/2 + i Omega kappa~^{-1}``.

    ``chi = D^{-1}`` is the rescaled susceptibility ``kappa^{1/2} chi_raw
    kappa^{1/2}``; its poles are the device resonances (App. A.2).
    """
    num_modes = H.shape[0] // 2
    sz = pauli_z(num_modes)
    gamma_full = _double(jnp.asarray(gamma))
    kappa_full = _double(jnp.asarray(kappa_tilde))
    return (
        -1j * (sz @ H)
        - jnp.diag(1.0 + gamma_full).astype(jnp.complex128) / 2.0
        + 1j * Omega * jnp.diag(1.0 / kappa_full).astype(jnp.complex128)
    )


def response_matrices(H, gamma, kappa_tilde, Omega):
    """Return ``(S, N)``: the scattering matrix and the intrinsic-loss noise
    response, both ``2N x 2N`` in Nambu space.

    ``S = 1 + D^{-1}``, ``N = D^{-1} sqrt(gamma)`` (App. A.4(b)).
    """
    num_modes = H.shape[0] // 2
    D = inverse_susceptibility(H, gamma, kappa_tilde, Omega)
    chi = jnp.linalg.inv(D)
    S = jnp.eye(2 * num_modes, dtype=jnp.complex128) + chi
    gamma_full = _double(jnp.asarray(gamma))
    N = chi @ jnp.diag(jnp.sqrt(jnp.abs(gamma_full))).astype(jnp.complex128)
    return S, N


def full_response(H, gamma, kappa_tilde, Omega):
    """``S_cal(Omega) = [ S(Omega) | N(Omega) ]`` -- response to *all* input
    channels (signal ports + auxiliaries + intrinsic-loss channels)."""
    S, N = response_matrices(H, gamma, kappa_tilde, Omega)
    return jnp.concatenate([S, N], axis=1)


def output_covariance_nambu(S_cal, sigma_in_total):
    """The covariance sandwich ``sigma_out = S_cal sigma_in S_cal^dag``
    (App. A.3).  Linear in ``sigma_in``, quadratic in the response."""
    return S_cal @ sigma_in_total @ jnp.conj(S_cal).T


def output_covariance_quadrature(
    H, gamma, kappa_tilde, Omega, sigma_in_signal, thetas, num_ports=None,
    sigma_in_noise=None,
):
    """Full Sec.-3 chain: graph parameters -> ``V(Omega)`` in the interleaved
    quadrature basis ``(x_1, p_1, x_2, p_2, ...)``, restricted to the first
    ``num_ports`` modes (the monitored signal ports).

    ``sigma_in_noise`` declares the covariance of the intrinsic-loss channels
    (vacuum if omitted); a thermal block there is what makes a designated loss
    channel hot.
    """
    num_modes = H.shape[0] // 2
    S_cal = full_response(H, gamma, kappa_tilde, Omega)
    sigma_in_total = stack_input_covariance(sigma_in_signal, num_modes,
                                            sigma_noise=sigma_in_noise)
    sigma_out = output_covariance_nambu(S_cal, sigma_in_total)
    W = quadrature_matrix(thetas, num_modes)
    V = nambu_to_quadrature(sigma_out, W)
    if num_ports is None:
        return V
    return V[: 2 * num_ports, : 2 * num_ports]


# ---------------------------------------------------------------------------
# read-only diagnostic: the noise susceptibility hot channel -> monitored port
# ---------------------------------------------------------------------------

def noise_response_block(H, gamma, kappa_tilde, Omega, port, channel, thetas=None):
    """The 2x2 quadrature sub-block of ``N(Omega)`` from intrinsic-loss channel
    ``channel`` into monitored ``port`` -- i.e. ``N_{j,k*}(Omega)`` of App.
    B.3(h).

    This is the object that governs how much of a hot bath reaches the
    monitored port: turning channel ``k*`` from vacuum to occupation ``n``
    inflates the port-``j`` quadrature block by

        d sigma_out,jj = 2 n * [ N e_{k*} N^dag ]_{jj-block} .

    **Diagnostic only.**  It is deliberately *not* a fit term: whether purity is
    recovered by driving this block to zero (noise evasion) or by some other
    mechanism is a question to measure, not to assume.

    Returns the real 2x2 matrix ``[[xx, xp], [px, pp]]`` of the contribution
    ``[ N e_{k*} N^dag ]`` in the quadrature basis, i.e. the *per-unit-``2n``*
    inflation of the port block.
    """
    num_modes = H.shape[0] // 2
    _, N = response_matrices(H, gamma, kappa_tilde, Omega)
    if thetas is None:
        thetas = jnp.zeros(num_modes)
    W = quadrature_matrix(thetas, num_modes)

    channel = int(channel)
    selector = jnp.zeros((2 * num_modes, 2 * num_modes), dtype=jnp.complex128)
    selector = selector.at[channel, channel].set(1.0)
    selector = selector.at[channel + num_modes, channel + num_modes].set(1.0)

    contribution = N @ selector @ jnp.conj(N).T
    block = nambu_to_quadrature(contribution, W)
    rows = slice(2 * int(port), 2 * int(port) + 2)
    return jnp.real(block[rows, rows])


def noise_response_amplitude(H, gamma, kappa_tilde, Omega, port, channel):
    """The raw Nambu amplitudes ``N[port, channel]`` and ``N[port, channel+N]``
    -- the normal and anomalous parts of the hot-channel susceptibility.

    ``noise_response_block`` is the physically meaningful (gauge-covariant)
    object; this one is handy when the question is *which* Nambu path carries
    the bath.
    """
    num_modes = H.shape[0] // 2
    _, N = response_matrices(H, gamma, kappa_tilde, Omega)
    port, channel = int(port), int(channel)
    return N[port, channel], N[port, channel + num_modes]


# ---------------------------------------------------------------------------
# Sec. 5 -- stability gate
# ---------------------------------------------------------------------------

def dynamical_matrix(H, gamma, kappa_tilde):
    """Dimensionless dynamical matrix ``M~`` (in units of ``kappa_ref``).

    ``M = kappa^{1/2} A kappa^{1/2}`` with ``A = -i sigma_z H - (1+gamma)/2``;
    ``M~ = kappa~^{1/2} A kappa~^{1/2}`` has the same eigenvalues as ``M`` up to
    the overall scale ``kappa_ref``, so it decides stability.
    """
    num_modes = H.shape[0] // 2
    sz = pauli_z(num_modes)
    gamma_full = _double(jnp.asarray(gamma))
    kappa_full = _double(jnp.asarray(kappa_tilde))
    A = -1j * (sz @ H) - jnp.diag(1.0 + gamma_full).astype(jnp.complex128) / 2.0
    root = jnp.diag(jnp.sqrt(jnp.abs(kappa_full))).astype(jnp.complex128)
    return root @ A @ root


def max_real_eigenvalue(H, gamma, kappa_tilde):
    """``max Re eig(M~)`` -- negative means a stable steady state exists."""
    M = dynamical_matrix(H, gamma, kappa_tilde)
    return jnp.max(jnp.real(jnp.linalg.eigvals(M)))


def is_stable(H, gamma, kappa_tilde, margin=0.0):
    """Sec. 5 stability gate: ``all( Re eig(M~) < -margin )``."""
    return bool(np.real(max_real_eigenvalue(H, gamma, kappa_tilde)) < -margin)
