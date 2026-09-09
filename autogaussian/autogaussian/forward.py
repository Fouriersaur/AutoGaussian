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
    "unrescale_hamiltonian",
    "collective_dynamical_matrix",
    "collective_susceptibility",
    "collective_response_matrices",
    "collective_output_covariance_nambu",
    "collective_output_covariance_quadrature",
    "collective_max_real_eigenvalue",
    "collective_is_stable",
    "scattering_sum_rule_residual",
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


# ---------------------------------------------------------------------------
# COLLECTIVE DISSIPATIVE CHANNELS (Addendum Sec. 2, Sec. 3)
# ---------------------------------------------------------------------------
#
# The scalar path above is written in the kappa-*rescaled* variables of App.
# A.4: kappa has been absorbed into ``chi = kappa^{1/2} chi_raw kappa^{1/2}``,
# which is why the collected damping is hard-coded to ``1`` and ``kappa~``
# survives only as the frequency rescale ``+ i Omega / kappa~``.  Both moves
# assume kappa is *diagonal*.  A collective channel set makes it a matrix, so
# the rescaling is no longer a similarity transformation and the collective
# path has to be written unrescaled, exactly as the base spec Sec. 3 states it:
#
#     M      = -i sigma_z H_BdG + M_diss
#     chi    = inv(M + i Omega I)
#     S      = I + K_kappa chi K_kappa^ddag
#     N      =     K_kappa chi K_Gamma^ddag
#
# The two paths agree identically on the private corner.  With
# ``K_kappa = diag(sqrt(kappa~))``, ``K_Gamma = diag(sqrt(gamma kappa~))`` and
# ``H_BdG = kappa~^{1/2} H kappa~^{1/2}`` one has
#
#     D = kappa~^{-1/2} (M + i Omega I) kappa~^{-1/2}
#       = -i sigma_z H - (1 + gamma)/2 + i Omega kappa~^{-1}
#
# (sigma_z commutes with the doubled diagonal kappa~), hence
# ``chi = kappa~^{-1/2} D^{-1} kappa~^{-1/2}`` and
#
#     K_kappa chi K_kappa^ddag = D^{-1}          -> S matches
#     K_kappa chi K_Gamma^ddag = D^{-1} sqrt(gamma)  -> N matches
#
# so D0 is an algebraic identity, not merely a numerical coincidence.  It is
# still asserted numerically in the tests (Addendum Sec. 8, D0) because that is
# what catches a typo in the assembly.
#
# One structural consequence: the output field lives in **channel space**.  With
# ``Mc`` controlled channels ``S`` is ``2 Mc x 2 Mc``, and the monitored ports
# are the first ``num_ports`` *controlled channels*, not modes.  The private
# corner has one controlled channel per mode in mode order, which is why the
# base spec could conflate the two.


def unrescale_hamiltonian(H, kappa_tilde):
    """``H_BdG = kappa~^{1/2} H kappa~^{1/2}`` -- the bridge from the rescaled
    Hamiltonian the parametrisation emits to the unrescaled one the collective
    drift consumes (both in units of ``kappa_ref``)."""
    root = jnp.diag(jnp.sqrt(jnp.abs(_double(jnp.asarray(kappa_tilde))))).astype(jnp.complex128)
    return root @ jnp.asarray(H, dtype=jnp.complex128) @ root


def collective_dynamical_matrix(H_bdg, channels):
    """``M = -i sigma_z H_BdG + M_diss`` (Addendum Sec. 2).

    ``H_bdg`` is the **unrescaled** BdG matrix; ``channels`` a
    :class:`~autogaussian.channels.ChannelSet`.  Both accesses damp, so
    ``M_diss`` is built from every channel; only the response separates them.
    """
    num_modes = H_bdg.shape[0] // 2
    sz = pauli_z(num_modes)
    return -1j * (sz @ jnp.asarray(H_bdg, dtype=jnp.complex128)) + channels.dissipative_drift()


def collective_susceptibility(M, Omega):
    """``chi(Omega) = inv(M + i Omega I)``."""
    size = M.shape[0]
    return jnp.linalg.inv(M + 1j * Omega * jnp.eye(size, dtype=jnp.complex128))


def collective_response_matrices(H_bdg, channels, Omega):
    """``(S, N)`` for a collective channel set (Addendum Sec. 3).

    ``S`` is ``2 Mk x 2 Mk`` and ``N`` is ``2 Mk x 2 Mg`` with ``Mk``/``Mg`` the
    number of controlled/uncontrolled channels.  ``K_kappa`` is the measurement
    aperture: it is the left factor of *both* terms (the only route to the
    detector).  ``K_Gamma`` appears only on the right of ``N`` -- the
    environment pushes noise in, you never look out of it.
    """
    from autogaussian.channels import nambu_adjoint

    num_modes = channels.num_modes
    n_ctrl, n_unctrl = channels.num_controlled, channels.num_uncontrolled
    M = collective_dynamical_matrix(H_bdg, channels)
    chi = collective_susceptibility(M, Omega)

    K_kappa = channels.K_kappa()
    K_Gamma = channels.K_Gamma()
    K_kappa_dd = nambu_adjoint(K_kappa, n_ctrl, num_modes)
    K_Gamma_dd = nambu_adjoint(K_Gamma, n_unctrl, num_modes)

    S = jnp.eye(2 * n_ctrl, dtype=jnp.complex128) + K_kappa @ chi @ K_kappa_dd
    N = K_kappa @ chi @ K_Gamma_dd
    return S, N


def collective_output_covariance_nambu(S, N, sigma_sig, sigma_noise=None):
    """``sigma_out = S sigma_sig S^ddag + N sigma_noise N^ddag`` (Addendum Sec. 3).

    Both sandwiches use the ordinary conjugate transpose: ``sigma_out`` is a
    covariance, and the ``sigma_z`` metric of the Nambu adjoint belongs to the
    *scattering* relation, not to the covariance sandwich.  ``sigma_noise``
    defaults to vacuum in every uncontrolled channel.
    """
    from autogaussian.nambu import vacuum_covariance

    n_unctrl = N.shape[1] // 2
    out = S @ jnp.asarray(sigma_sig, dtype=jnp.complex128) @ jnp.conj(S).T
    if n_unctrl:
        if sigma_noise is None:
            sigma_noise = vacuum_covariance(n_unctrl)
        out = out + N @ jnp.asarray(sigma_noise, dtype=jnp.complex128) @ jnp.conj(N).T
    return out


def collective_output_covariance_quadrature(
    H_bdg, channels, Omega, sigma_sig, thetas, num_ports=None, sigma_noise=None,
):
    """Full Sec.-3 chain for a collective channel set, in the interleaved
    quadrature basis ``(x_1, p_1, x_2, p_2, ...)`` **indexed by controlled
    channel**, restricted to the first ``num_ports`` of them.

    ``thetas`` are the per-*channel* quadrature reference phases (the gauge of
    Sec. 1.5), one per controlled channel.
    """
    n_ctrl = channels.num_controlled
    S, N = collective_response_matrices(H_bdg, channels, Omega)
    sigma_out = collective_output_covariance_nambu(S, N, sigma_sig, sigma_noise)
    W = quadrature_matrix(thetas, n_ctrl)
    V = nambu_to_quadrature(sigma_out, W)
    if num_ports is None:
        return V
    return V[: 2 * num_ports, : 2 * num_ports]


def collective_max_real_eigenvalue(H_bdg, channels):
    """``max Re eig(M)`` for the collective drift -- the Sec. 5 stability gate.

    Unlike the rescaled ``dynamical_matrix``, no similarity rescaling is
    applied (there is none available once the rate matrices are non-diagonal),
    so this returns the abscissa of the physical drift in units of
    ``kappa_ref``.
    """
    M = collective_dynamical_matrix(H_bdg, channels)
    return jnp.max(jnp.real(jnp.linalg.eigvals(M)))


def collective_is_stable(H_bdg, channels, margin=0.0):
    return bool(np.real(collective_max_real_eigenvalue(H_bdg, channels)) < -margin)


def scattering_sum_rule_residual(H_bdg, channels, Omega):
    """``|| S sigma_z S^H + N sigma_z N^H - sigma_z ||`` (Addendum Sec. 3).

    The full-scattering sum rule -- the sharp statement of "purity is a routing
    condition, not a dark state": the monitored map ``S`` is symplectic exactly
    where ``N`` carries no weight, so ``det sigma_out = 1`` wherever nothing
    routes through ``K_Gamma`` into the monitored quadrature.

    **Correction to Addendum Sec. 3 (verified numerically, see
    ``tests/test_collective_channels.py``).**  The addendum writes this rule
    with the Nambu adjoint, ``S sigma_z S^ddag + N sigma_z N^ddag = sigma_z``,
    while also defining ``X^ddag = Sigma_z X^H Sigma_z``.  Those two statements
    are inconsistent: substituting the second into the first collapses the
    metrics and yields ``S S^H + N N^H = I``, which the existing (private,
    particle-conserving) forward map violates by O(1) -- residual 2.1 on the
    lossy two-mode device at ``Omega = 0``.  The identity that does hold, to
    ~1e-15 on every device in the golden set, is the pseudo-unitarity written
    with the **ordinary** conjugate transpose, as implemented here.  The
    ``^ddag`` of the *response* formulas ``S = I + K_kappa chi K_kappa^ddag``
    is a separate object and is unaffected; on the particle-conserving path it
    provably collapses to ``^H`` anyway (see
    :func:`autogaussian.channels.nambu_adjoint`).
    """
    n_ctrl, n_unctrl = channels.num_controlled, channels.num_uncontrolled
    S, N = collective_response_matrices(H_bdg, channels, Omega)
    sz_ctrl = pauli_z(n_ctrl).astype(jnp.complex128)
    lhs = S @ sz_ctrl @ jnp.conj(S).T
    if n_unctrl:
        sz_unctrl = pauli_z(n_unctrl).astype(jnp.complex128)
        lhs = lhs + N @ sz_unctrl @ jnp.conj(N).T
    return float(jnp.linalg.norm(lhs - sz_ctrl))
