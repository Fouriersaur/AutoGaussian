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
    H, gamma, kappa_tilde, Omega, sigma_in_signal, thetas, num_ports=None
):
    """Full Sec.-3 chain: graph parameters -> ``V(Omega)`` in the interleaved
    quadrature basis ``(x_1, p_1, x_2, p_2, ...)``, restricted to the first
    ``num_ports`` modes (the monitored signal ports).
    """
    num_modes = H.shape[0] // 2
    S_cal = full_response(H, gamma, kappa_tilde, Omega)
    sigma_in_total = stack_input_covariance(sigma_in_signal, num_modes)
    sigma_out = output_covariance_nambu(S_cal, sigma_in_total)
    W = quadrature_matrix(thetas, num_modes)
    V = nambu_to_quadrature(sigma_out, W)
    if num_ports is None:
        return V
    return V[: 2 * num_ports, : 2 * num_ports]


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
