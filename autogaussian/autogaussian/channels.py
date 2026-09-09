"""
COLLECTIVE DISSIPATIVE CHANNELS (Addendum Sec. 1, Sec. 2, Sec. 5).

The base spec gives every mode two private jumps, ``sqrt(kappa_i) a_i``
(collected) and ``sqrt(Gamma_i) a_i`` (lost), so the damping enters the drift
as the scalar-diagonal ``-(kappa + Gamma)/2``.  Here that scaffold is
generalised to a searchable set of **collective** jumps, one row per channel:

    L_mu = sum_i ( u_{mu,i} a_i + v_{mu,i} a_i^dag )      U, V in C^{Mc x N}

with three attributes the search chooses:

    support   {i : u_{mu,i} != 0 or v_{mu,i} != 0}   (1 -> private, >=2 -> collective)
    class     PARTICLE_CONSERVING (v row = 0)  |  BOGOLIUBOV (v row != 0)
    access    CONTROLLED (driven and detected, rate-type kappa)
              UNCONTROLLED (the environment owns both ends, rate-type Gamma)

**Private channels are the diagonal corner of this space, not a separate code
path** (Addendum prime invariant): a support-1 ``CONTROLLED`` amplitude *is*
``sqrt(kappa_i)`` and a support-1 ``UNCONTROLLED`` amplitude *is*
``sqrt(Gamma_i)``.  :func:`private_channel_set` builds exactly that corner from
the base spec's ``(gamma, kappa_tilde)`` so the reduction is testable.

Drift (Addendum Sec. 2, DERIVED)
--------------------------------
::

    M = -i sigma_z H_BdG + M_diss

    M_diss = [[ A_diss      ,  B_diss      ],
              [ conj(B_diss),  conj(A_diss)]]

    A_diss = 1/2 ( conj(V^H V) - U^H U )       normal    (dissipative hopping / (anti)damping)
    B_diss = 1/2 ( conj(V^H U) - U^H V )       anomalous (dissipative squeezing)

``U``/``V`` stack **all** channels regardless of access -- collected and lost
photons damp the modes alike.  Only the response (Sec. 3) distinguishes them.

Nambu coupling matrices (Addendum Sec. 3)
-----------------------------------------
A channel set of ``Mc`` rows has the ``2 Mc x 2 N`` Nambu coupling matrix

    K = [[ U      ,  V       ],
         [ conj(V),  conj(U) ]]

``K_kappa`` is built from the CONTROLLED rows, ``K_Gamma`` from the
UNCONTROLLED ones.  Note the output field lives in **channel space**: with
``Mc`` controlled channels ``S`` is ``2 Mc x 2 Mc``, not ``2 N x 2 N``.  For a
private controlled set the two coincide, which is why the base spec could
conflate them.

Canonical form (Addendum Sec. 5, CONVENTION -- enforced here)
-------------------------------------------------------------
``L = sqrt(kappa) a`` on a squeezed input and
``L = sqrt(kappa)(cosh r a + sinh r a^dag)`` on vacuum give the *same*
``sigma_out(omega)`` from two different graph+bath specs.  Left alone that
degeneracy corrupts every minimality claim, so:

    CONTROLLED  =>  PARTICLE_CONSERVING     (collected rows carry V = 0)
    BOGOLIUBOV  =>  UNCONTROLLED            (anomalous jumps only where they are honestly loss)

Anomalous content that is generated *and* collected belongs in ``H_BdG``
(coherent ``nu``) or in ``sigma_sig`` (a squeezed input), never in a collected
jump.  :class:`Channel` rejects violations at construction.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
import jax.numpy as jnp

__all__ = [
    "ChannelClass",
    "Access",
    "Channel",
    "ChannelSet",
    "private_channel_set",
    "nambu_coupling",
    "nambu_adjoint",
    "dissipative_drift",
]

_ZERO_TOL = 1.0e-14


class ChannelClass(Enum):
    """Whether the jump row has anomalous content (Addendum Sec. 1)."""

    PARTICLE_CONSERVING = "particle_conserving"
    BOGOLIUBOV = "bogoliubov"


class Access(Enum):
    """Who owns the two ends of the channel (Addendum Sec. 1).

    ``CONTROLLED`` = you drive it *and* detect it (rate-type ``kappa``);
    ``UNCONTROLLED`` = the environment owns both ends (rate-type ``Gamma``).
    This flag is the *only* thing separating ``K_kappa`` from ``K_Gamma``;
    microscopically the two are the same object.
    """

    CONTROLLED = "controlled"
    UNCONTROLLED = "uncontrolled"


@dataclass(frozen=True)
class Channel:
    """One jump row ``L = sum_i (u_i a_i + v_i a_i^dag)``.

    Parameters
    ----------
    u, v : complex arrays of length ``num_modes``
        The normal and anomalous amplitude rows.  ``v = 0`` (or omitted) makes
        the channel ``PARTICLE_CONSERVING``.
    access : Access

    Raises
    ------
    ValueError
        If a ``CONTROLLED`` channel carries anomalous amplitudes -- the Sec. 5
        canonical form, enforced at construction rather than checked later.
    """

    u: np.ndarray
    v: np.ndarray
    access: Access

    def __init__(self, u, v=None, access=Access.CONTROLLED):
        u = np.atleast_1d(np.asarray(u, dtype=complex))
        if v is None:
            v = np.zeros_like(u)
        else:
            v = np.atleast_1d(np.asarray(v, dtype=complex))
        if u.shape != v.shape:
            raise ValueError("u and v must have the same length (got %r and %r)"
                             % (u.shape, v.shape))
        access = Access(access)
        if access is Access.CONTROLLED and np.any(np.abs(v) > _ZERO_TOL):
            raise ValueError(
                "a CONTROLLED channel must be PARTICLE_CONSERVING (v = 0): the "
                "Sec. 5 canonical form.  A collected Bogoliubov jump is "
                "degenerate with a plain drain on a squeezed input, and the "
                "degeneracy would corrupt every minimality claim.  Put the "
                "anomalous content in H_BdG (coherent nu) or in sigma_sig, or "
                "declare the channel UNCONTROLLED.")
        object.__setattr__(self, "u", u.copy())
        object.__setattr__(self, "v", v.copy())
        object.__setattr__(self, "access", access)

    # -- attributes the search reads --------------------------------------

    @property
    def num_modes(self):
        return int(self.u.shape[0])

    @property
    def channel_class(self):
        if np.any(np.abs(self.v) > _ZERO_TOL):
            return ChannelClass.BOGOLIUBOV
        return ChannelClass.PARTICLE_CONSERVING

    @property
    def support(self):
        """Modes the channel touches -- ``len(support) == 1`` means private."""
        touched = (np.abs(self.u) > _ZERO_TOL) | (np.abs(self.v) > _ZERO_TOL)
        return tuple(int(i) for i in np.flatnonzero(touched))

    @property
    def is_private(self):
        return len(self.support) <= 1

    @property
    def is_collective(self):
        return len(self.support) >= 2

    def __str__(self):
        return "%s %s support=%s" % (self.access.value, self.channel_class.value,
                                     list(self.support))


class ChannelSet:
    """An ordered list of :class:`Channel` rows over a common mode register.

    Ordering matters: the CONTROLLED channels, in the order given, are the
    **output ports** -- ``sigma_out`` is indexed by them, and the monitored
    ports are the first ``num_ports`` of them.  For a private controlled set
    (one channel per mode, in mode order) channel index == mode index, which is
    the base spec's convention.
    """

    def __init__(self, channels: Sequence[Channel], num_modes=None):
        channels = list(channels)
        if num_modes is None:
            if not channels:
                raise ValueError("num_modes is required for an empty channel set")
            num_modes = channels[0].num_modes
        self.num_modes = int(num_modes)
        for channel in channels:
            if channel.num_modes != self.num_modes:
                raise ValueError("channel %s spans %i modes, the set has %i"
                                 % (channel, channel.num_modes, self.num_modes))
        self.channels = channels

    # -- views -------------------------------------------------------------

    def __len__(self):
        return len(self.channels)

    def __iter__(self):
        return iter(self.channels)

    def __getitem__(self, idx):
        return self.channels[idx]

    def of_access(self, access):
        access = Access(access)
        return [c for c in self.channels if c.access is access]

    @property
    def controlled(self):
        return self.of_access(Access.CONTROLLED)

    @property
    def uncontrolled(self):
        return self.of_access(Access.UNCONTROLLED)

    @property
    def num_controlled(self):
        return len(self.controlled)

    @property
    def num_uncontrolled(self):
        return len(self.uncontrolled)

    @property
    def all_private(self):
        """True when the set is the diagonal corner -- the base-spec scaffold."""
        return all(c.is_private for c in self.channels)

    def rows(self, channels=None):
        """``(U, V)`` stacked over ``channels`` (all of them by default)."""
        if channels is None:
            channels = self.channels
        if not channels:
            zero = np.zeros((0, self.num_modes), dtype=complex)
            return zero, zero.copy()
        U = np.stack([c.u for c in channels])
        V = np.stack([c.v for c in channels])
        return U, V

    # -- the objects the forward map consumes ------------------------------

    def coupling(self, access):
        """Nambu coupling matrix ``K`` (``2 Mc x 2 N``) of one access class."""
        U, V = self.rows(self.of_access(access))
        return nambu_coupling(U, V)

    def K_kappa(self):
        """``K_kappa`` -- the measurement aperture (Addendum Sec. 3)."""
        return self.coupling(Access.CONTROLLED)

    def K_Gamma(self):
        """``K_Gamma`` -- the environment's coupling; appears only on the right."""
        return self.coupling(Access.UNCONTROLLED)

    def dissipative_drift(self):
        """``M_diss`` (``2 N x 2 N``) from **all** channels (Addendum Sec. 2)."""
        U, V = self.rows()
        return dissipative_drift(U, V)

    def describe(self):
        return [str(c) for c in self.channels]


# ---------------------------------------------------------------------------
# assembly primitives (kept free-standing so tests can hit them directly)
# ---------------------------------------------------------------------------

def nambu_coupling(U, V):
    """``K = [[U, V], [conj(V), conj(U)]]`` -- ``2 Mc x 2 N`` (Addendum Sec. 3).

    A private particle-conserving set (``U = diag(sqrt(kappa))``, ``V = 0``)
    gives ``K = diag(sqrt(kappa), sqrt(kappa))``, i.e. the base spec's
    per-mode rate vector replicated over the two Nambu blocks.
    """
    U = jnp.asarray(U, dtype=jnp.complex128)
    V = jnp.asarray(V, dtype=jnp.complex128)
    return jnp.block([[U, V], [jnp.conj(V), jnp.conj(U)]])


def nambu_adjoint(X, num_channels, num_modes):
    """The Nambu adjoint ``X^ddag = sigma_z^sys . X^H . sigma_z^ch``.

    ``X`` maps system Nambu space (``2 N``) to channel Nambu space (``2 Mc``),
    so its adjoint runs the other way and the two ``sigma_z`` metrics sit on
    the sides that make the shapes work: system metric on the left of the
    ``2 N x 2 Mc`` result, channel metric on the right.

    **VERIFIED** (Addendum Sec. 3 lists this as NEEDS VERIFICATION; D2 is now
    green -- see ``tests/test_collective_bogoliubov.py``).

    For ``PARTICLE_CONSERVING`` channels this provably collapses to the ordinary
    ``X^H``: the two sign flips cancel on the lower block.  For ``BOGOLIUBOV``
    channels the placement above is pinned against the exact Lindblad steady
    state -- the response-side diffusion
    ``D_eff = sum_c K_c^ddag sigma_in,c (K_c^ddag)^H`` must satisfy the Lyapunov
    equation ``M sigma_c + sigma_c M^H + D_eff = 0`` for the ``sigma_c`` obtained
    by diagonalising the Liouvillian in a truncated Fock space.  It does, with a
    residual that vanishes with the cutoff (8.6e-5 -> 1.0e-9); dropping the two
    ``sigma_z`` metrics leaves the residual at a constant 1.57.  The
    channel-space doubling is validated separately on a collective Bogoliubov
    channel spanning two modes, where ``sigma_z^ch`` and ``sigma_z^sys`` have
    different sizes.

    Note the Sec. 3 sum rule does **not** discriminate: both placements are
    pseudo-unitary, differing only by the sign of the effective bath squeezing
    angle.  Anyone re-deriving this from the sum rule alone will get it right
    half the time.
    """
    from autogaussian.nambu import pauli_z

    X = jnp.asarray(X, dtype=jnp.complex128)
    sz_sys = pauli_z(int(num_modes)).astype(jnp.complex128)
    sz_ch = pauli_z(int(num_channels)).astype(jnp.complex128)
    return sz_sys @ jnp.conj(X).T @ sz_ch


def dissipative_drift(U, V):
    """``M_diss`` from the stacked jump rows (Addendum Sec. 2, DERIVED).

        A_diss = 1/2 ( conj(V^H V) - U^H U )
        B_diss = 1/2 ( conj(V^H U) - U^H V )
        M_diss = [[A, B], [conj(B), conj(A)]]

    ``V = 0`` gives ``A_diss = -1/2 U^H U <= 0`` and ``B_diss = 0``: a Hermitian
    negative-semidefinite addition to the normal block -- pure dissipative
    hopping, unconditionally stabilising.  Its off-diagonal
    ``(U^H U)_{jk} = sum_mu conj(u_{mu,j}) u_{mu,k}`` is nonzero **iff one
    channel couples both j and k**, which is what makes a collective channel a
    coupling and not just a rate.
    """
    U = jnp.asarray(U, dtype=jnp.complex128)
    V = jnp.asarray(V, dtype=jnp.complex128)
    UhU = jnp.conj(U).T @ U
    VhV = jnp.conj(V).T @ V
    VhU = jnp.conj(V).T @ U
    UhV = jnp.conj(U).T @ V
    A = 0.5 * (jnp.conj(VhV) - UhU)
    B = 0.5 * (jnp.conj(VhU) - UhV)
    return jnp.block([[A, B], [jnp.conj(B), jnp.conj(A)]])


# ---------------------------------------------------------------------------
# the diagonal corner: the base spec's private scaffold, as a channel set
# ---------------------------------------------------------------------------

def private_channel_set(gamma, kappa_tilde, include_lossless=False):
    """The base-spec scaffold expressed in the collective language.

    Parameters
    ----------
    gamma : array of length N
        Intrinsic-loss ratios ``gamma_i = Gamma_i / kappa_i`` (base spec App. A.4).
    kappa_tilde : array of length N
        Decay ratios ``kappa~_i = kappa_i / kappa_ref``.
    include_lossless : bool
        Emit an UNCONTROLLED channel even for modes with ``gamma_i = 0``.  Off
        by default so ``num_uncontrolled`` counts the *live* loss channels; the
        zero-amplitude row is physically inert either way.

    Returns
    -------
    ChannelSet
        ``N`` private CONTROLLED channels with ``u_ii = sqrt(kappa~_i)``, in
        mode order (so channel index == mode index == port index), followed by
        the private UNCONTROLLED channels with ``u_ii = sqrt(gamma_i kappa~_i)``.

    Notes
    -----
    Everything is in units of ``kappa_ref``: ``kappa_i -> kappa~_i`` and
    ``Gamma_i = gamma_i kappa_i -> gamma_i kappa~_i``.  With this set the
    collective forward map is *algebraically identical* to the base spec's
    rescaled one -- see :func:`autogaussian.forward.collective_response_matrices`.
    """
    gamma = np.atleast_1d(np.asarray(gamma, dtype=float))
    kappa_tilde = np.atleast_1d(np.asarray(kappa_tilde, dtype=float))
    num_modes = int(kappa_tilde.shape[0])
    if gamma.shape[0] != num_modes:
        raise ValueError("gamma and kappa_tilde must have the same length")

    channels = []
    for i in range(num_modes):
        u = np.zeros(num_modes, dtype=complex)
        u[i] = np.sqrt(np.abs(kappa_tilde[i]))
        channels.append(Channel(u, access=Access.CONTROLLED))
    for i in range(num_modes):
        rate = np.abs(gamma[i]) * np.abs(kappa_tilde[i])
        if rate <= 0.0 and not include_lossless:
            continue
        u = np.zeros(num_modes, dtype=complex)
        u[i] = np.sqrt(rate)
        channels.append(Channel(u, access=Access.UNCONTROLLED))
    return ChannelSet(channels, num_modes=num_modes)
