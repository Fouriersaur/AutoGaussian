"""
Nambu / BdG algebra (Sec. 0, Sec. 2, App. A.1 of the algorithm flow).

Conventions used everywhere in this package
-------------------------------------------
Nambu vector (size 2N, N = number of modes):

    A = (a_1, ..., a_N, a_1^dag, ..., a_N^dag)

Hamiltonian (rotating frame, RWA, quadratic):

    H = sum_jk g_jk a_j^dag a_k + 1/2 sum_jk ( nu_jk a_j^dag a_k^dag + h.c. )

with ``g`` Hermitian (beam-splitter couplings, detunings on the diagonal) and
``nu`` symmetric (two-mode squeezing off-diagonal, degenerate/on-site squeezing
on the diagonal).  The BdG matrix is

    H_BdG = [[ g   ,  nu    ],
             [ nu* ,  g*    ]]

Everything is stored *dimensionless*: ``H = kappa^{-1/2} H_BdG kappa^{-1/2}``,
``gamma_i = Gamma_i / kappa_i``, ``kappa~_i = kappa_i / kappa_ref``,
``Omega = omega / kappa_ref``  (App. A.4).

Covariance normalisation: the (symmetrised) Nambu covariance is

    sigma^N_mn = < { A_m , A_n^dag } >

so that vacuum gives ``sigma^N = 1`` and the quadratures
``x_j = a_j e^{-i theta_j} + a_j^dag e^{+i theta_j}``,
``p_j = -i( a_j e^{-i theta_j} - a_j^dag e^{+i theta_j} )``
have vacuum variance 1 ("vacuum floor normalised to variance 1", Sec. 0).

The quadrature covariance is

    V = 1/2 W sigma^N W^dag ,

returned in the *interleaved* basis ``(x_1, p_1, x_2, p_2, ...)`` used by the
target gallery (App. B).
"""

import numpy as np
import jax.numpy as jnp

__all__ = [
    "pauli_z",
    "build_H_bdg",
    "quadrature_matrix",
    "nambu_to_quadrature",
    "channel_covariance",
    "vacuum_covariance",
    "thermal_bath",
    "squeezed_bath",
    "stack_input_covariance",
    "duan_sum",
    "symplectic_eigenvalues",
    "variance_to_dB",
]


def pauli_z(num_modes):
    """``sigma_z = diag(1_N, -1_N)`` -- the relative sign between the ``a`` and
    ``a^dag`` blocks of the equations of motion (App. A.1)."""
    return jnp.diag(jnp.concatenate([jnp.ones(num_modes), -jnp.ones(num_modes)]))


def build_H_bdg(g, nu):
    """Assemble ``H_BdG = [[g, nu], [nu*, g*]]`` from the normal (``g``) and
    anomalous (``nu``) blocks.

    ``g`` must be Hermitian, ``nu`` symmetric; this is *not* checked here (the
    parametrisation in :mod:`autogaussian.parametrization` enforces it by
    construction).
    """
    return jnp.block([[g, nu], [jnp.conj(nu), jnp.conj(g)]])


def quadrature_matrix(thetas, num_modes=None):
    """Nambu -> quadrature transformation ``W`` (2N x 2N).

    Row ``2j``   is ``x_j = a_j e^{-i theta_j} + a_j^dag e^{i theta_j}``
    Row ``2j+1`` is ``p_j = -i(a_j e^{-i theta_j} - a_j^dag e^{i theta_j})``

    so that the output is ordered ``(x_1, p_1, x_2, p_2, ...)``.  The angles
    ``theta_j`` are the per-port quadrature reference phases -- the gauge of
    Sec. 1.5 / App. A.4(e).
    """
    thetas = jnp.asarray(thetas)
    if num_modes is None:
        num_modes = thetas.shape[0]
    minus = jnp.exp(-1j * thetas)
    plus = jnp.exp(+1j * thetas)

    rows_x = jnp.arange(num_modes) * 2
    rows_p = rows_x + 1
    cols_a = jnp.arange(num_modes)
    cols_adag = cols_a + num_modes

    W = jnp.zeros((2 * num_modes, 2 * num_modes), dtype=jnp.complex128)
    W = W.at[rows_x, cols_a].set(minus)
    W = W.at[rows_x, cols_adag].set(plus)
    W = W.at[rows_p, cols_a].set(-1j * minus)
    W = W.at[rows_p, cols_adag].set(+1j * plus)
    return W


def nambu_to_quadrature(sigma_nambu, W):
    """``V = 1/2 W sigma^N W^dag`` (App. A.3).

    ``V`` is Hermitian; its real part is the symmetric physical quadrature
    covariance, its (antisymmetric) imaginary part encodes the
    ``x``/``p`` cross-spectrum asymmetry at ``Omega != 0``.
    """
    return 0.5 * (W @ sigma_nambu @ jnp.conj(W).T)


def channel_covariance(n=0.0, m=0.0, num_modes=None):
    """Nambu covariance (2N x 2N) of a set of *independent* input channels.

    Parameters
    ----------
    n : float or array of length N
        Thermal occupation of each channel.
    m : complex or array of length N
        Anomalous amplitude ``m_j = <a_j a_j>`` of each channel (nonzero for a
        squeezed bath).
    num_modes : int, optional
        Required if both ``n`` and ``m`` are scalars.

    Notes
    -----
    Per channel the block is ``[[2n+1, 2m], [2m*, 2n+1]]``; vacuum (``n=m=0``)
    gives the identity.  A pure squeezed vacuum satisfies ``|m|^2 = n(n+1)``.
    """
    n = jnp.asarray(n, dtype=jnp.float64)
    m = jnp.asarray(m, dtype=jnp.complex128)
    if num_modes is None:
        num_modes = max(int(np.size(n)), int(np.size(m)))
    n = jnp.broadcast_to(n, (num_modes,))
    m = jnp.broadcast_to(m, (num_modes,))

    diag = jnp.diag(2.0 * n + 1.0).astype(jnp.complex128)
    anomalous = jnp.diag(2.0 * m)
    return jnp.block([[diag, anomalous], [jnp.conj(anomalous), diag]])


def vacuum_covariance(num_modes):
    """``sigma^N = 1_{2N}`` -- vacuum in every channel."""
    return jnp.eye(2 * num_modes, dtype=jnp.complex128)


def thermal_bath(n):
    """``(n, m)`` of a thermal bath with occupation ``n``."""
    return float(n), 0.0 + 0.0j


def squeezed_bath(r, phi=0.0):
    """``(n, m)`` of a pure squeezed vacuum with squeeze parameter ``r`` and
    angle ``phi``.

    ``Var(x_phi/2) = e^{-2r}`` for ``phi = 0`` (in the vacuum-floor-1
    normalisation).
    """
    n = float(np.sinh(r) ** 2)
    m = float(np.cosh(r) * np.sinh(r)) * np.exp(1j * phi)
    return n, complex(m)


def stack_input_covariance(sigma_signal, num_modes):
    """``sigma_in = blockdiag(signal inputs, vacuum)`` for the combined input
    vector ``(a^in, a^noise)`` of the response ``S_cal = [S | N]`` (Sec. 3)."""
    z = jnp.zeros((2 * num_modes, 2 * num_modes), dtype=jnp.complex128)
    vac = vacuum_covariance(num_modes)
    return jnp.block([[sigma_signal, z], [z, vac]])


# ---------------------------------------------------------------------------
# small diagnostics used by the gallery / post-processing
# ---------------------------------------------------------------------------

def variance_to_dB(variance):
    """Squeezing in dB relative to the vacuum floor 1 (positive = squeezed)."""
    return -10.0 * np.log10(np.asarray(variance, dtype=float))


def symplectic_eigenvalues(V):
    """Symplectic eigenvalues of a real 2n x 2n covariance matrix ``V`` given in
    the interleaved ``(x_1,p_1,...)`` basis with vacuum floor 1.

    Purity check: a pure state has all symplectic eigenvalues equal to 1.
    """
    V = np.real(np.asarray(V))
    n = V.shape[0] // 2
    Omega = np.zeros_like(V)
    for j in range(n):
        Omega[2 * j, 2 * j + 1] = 1.0
        Omega[2 * j + 1, 2 * j] = -1.0
    ev = np.linalg.eigvals(1j * Omega @ V)
    ev = np.sort(np.abs(ev))
    return ev[n:]


def duan_sum(V, port1=0, port2=1):
    """Duan/EPR quantity ``Var(x_1 - x_2) + Var(p_1 + p_2)`` (vacuum floor 1 =>
    separability bound 4 in this normalisation, i.e. ``< 4`` certifies
    entanglement for symmetric states; the gallery quotes the halved
    convention)."""
    V = np.real(np.asarray(V))
    i, j = 2 * port1, 2 * port2
    var_xm = V[i, i] + V[j, j] - 2 * V[i, j]
    var_pp = V[i + 1, i + 1] + V[j + 1, j + 1] + 2 * V[i + 1, j + 1]
    return float(var_xm + var_pp)
