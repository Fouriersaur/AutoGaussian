"""
Optional equality constraints ``f_j = 0`` added to the loss (Sec. 1.4, Sec. 4).

They are evaluated on a :class:`ForwardContext` -- the already-computed forward
map at the target frequencies -- and return an array of residuals that the
oracle squares and adds to the loss.

The most important one in practice is a *connectivity / transmission* pin:
without it, an under-specified directional target such as "squeeze port 1,
vacuum port 2" is satisfied trivially by decoupling the ports (Sec. 11).
"""

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

__all__ = [
    "ForwardContext",
    "BaseConstraint",
    "TransmissionConstraint",
    "MinimumTransmission",
    "IsolationConstraint",
    "EqualCooperativities",
    "CooperativityBudget",
    "CollectiveCooperativityBudget",
    "DissipativeRateBudget",
    "LinewidthAnchor",
    "PurityFloor",
    "QuadratureSpectrum",
    "MinimumIntrinsicLoss",
    "CustomConstraint",
]


@dataclass
class ForwardContext:
    """Everything the forward map produced at the target frequencies."""

    x: jnp.ndarray
    H: jnp.ndarray
    gamma: jnp.ndarray
    kappa_tilde: jnp.ndarray
    thetas: jnp.ndarray
    omegas: jnp.ndarray            # (K,)
    S: jnp.ndarray                 # (K, 2N, 2N) scattering matrix
    N: jnp.ndarray                 # (K, 2N, 2N) intrinsic-loss noise response
    V: jnp.ndarray                 # (K, 2N, 2N) quadrature covariance, all modes
    num_modes: int
    num_ports: int

    def omega_index(self, omega):
        return int(np.argmin(np.abs(np.asarray(self.omegas) - float(omega))))


class BaseConstraint:
    def __call__(self, ctx):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


@dataclass
class TransmissionConstraint(BaseConstraint):
    """Pin the phase-preserving power transmission ``|S[out, in]|^2 = value``.

    Use it to force a *single connected device*: pinning a nonzero forward
    transmission rules out the trivial decoupled solution (Sec. 11, App. F).
    """

    port_out: int
    port_in: int
    value: float
    omega: float = 0.0
    weight: float = 1.0

    def __call__(self, ctx):
        k = ctx.omega_index(self.omega)
        amplitude = ctx.S[k, self.port_out, self.port_in]
        return jnp.array([jnp.sqrt(self.weight) * (jnp.abs(amplitude) ** 2 - self.value)])


@dataclass
class MinimumTransmission(BaseConstraint):
    """One-sided version: penalise only ``|S[out,in]|^2 < minimum``."""

    port_out: int
    port_in: int
    minimum: float
    omega: float = 0.0
    weight: float = 1.0

    def __call__(self, ctx):
        k = ctx.omega_index(self.omega)
        amplitude = ctx.S[k, self.port_out, self.port_in]
        deficit = self.minimum - jnp.abs(amplitude) ** 2
        return jnp.array([jnp.sqrt(self.weight) * jnp.maximum(deficit, 0.0)])


@dataclass
class IsolationConstraint(BaseConstraint):
    """``|S[out, in]|^2 = 0`` -- backward direction silenced."""

    port_out: int
    port_in: int
    omega: float = 0.0
    weight: float = 1.0

    def __call__(self, ctx):
        k = ctx.omega_index(self.omega)
        return jnp.array([jnp.sqrt(self.weight) * jnp.abs(ctx.S[k, self.port_out, self.port_in])])


@dataclass
class EqualCooperativities(BaseConstraint):
    """Force several couplings to share one magnitude (symmetry constraint).

    ``pairs`` are ``(i, j)`` mode index pairs; ``block`` is ``'normal'`` for
    beam-splitter couplings, ``'anomalous'`` for squeezing couplings.
    """

    pairs: Sequence[Tuple[int, int]]
    block: str = "normal"
    weight: float = 1.0

    def __call__(self, ctx):
        n = ctx.num_modes
        matrix = ctx.H[:n, :n] if self.block == "normal" else ctx.H[:n, n:]
        magnitudes = jnp.array([jnp.abs(matrix[i, j]) for i, j in self.pairs])
        return jnp.sqrt(self.weight) * (magnitudes - magnitudes[0])


@dataclass
class CooperativityBudget(BaseConstraint):
    """One-sided cap ``4|H_ij|^2 <= maximum`` on every coupling (Sec. 7.3).

    A hardware statement -- pumps carry finite power -- and the cheapest way to
    tell a *device* from a *limit point*.  An architecture that meets the target
    only as ``C -> infinity`` fails at every finite budget and its loss degrades
    monotonically as the budget tightens; an architecture with a genuine
    interior solution is indifferent to a loose one.  Tightening the fit
    tolerance does **not** do this: it just walks further along the diverging
    ray.

    ``include_detunings`` also caps ``4 Delta_i^2``, which matters when the ray
    runs a detuning and a squeezing amplitude to infinity together at fixed
    ratio (the B.10 order-3 case).
    """

    maximum: float
    include_detunings: bool = True
    weight: float = 1.0

    def __call__(self, ctx):
        n = ctx.num_modes
        normal = ctx.H[:n, :n]
        if not self.include_detunings:
            normal = normal - jnp.diag(jnp.diag(normal))
        entries = jnp.concatenate([jnp.abs(normal).ravel(),
                                   jnp.abs(ctx.H[:n, n:]).ravel()])
        excess = 4.0 * entries ** 2 - self.maximum
        return jnp.sqrt(self.weight) * jnp.maximum(excess, 0.0)


@dataclass
class CollectiveCooperativityBudget(BaseConstraint):
    """``CooperativityBudget`` for the collective engine (Addendum Sec. 4).

    The base-spec budget caps ``4|H_ij|^2``, which is the cooperativity *only*
    in the normalisation where every ``kappa_i`` is pinned to 1.  On the
    collective scaffold the collected rate is a free jump amplitude, so
    ``4|H_ij|^2`` is not dimensionless and a cap on it is not a pump budget: the
    scale gauge ``(H, K) -> (sH, sK)`` walks straight through it.  The
    dimensionless statement is

        C_ij = 4 |H_ij|^2 / (kappa_i kappa_j) <= maximum ,

    with ``kappa_i`` the total rate the mode's controlled channels add up to --
    the same quantity :func:`~autogaussian.collective_oracle.collective_cooperativities`
    reports, and the reason that helper exists rather than ``physical_report``.

    ``include_detunings`` also caps ``4 Delta_i^2 / kappa_i^2``, matching the
    base-spec flag.

    Using this on the base spec is harmless but pointless: there ``kappa~_i`` is
    a live variable too, so the two budgets differ, and
    :class:`CooperativityBudget` is the one App. A.4 is written against.
    """

    maximum: float
    include_detunings: bool = True
    weight: float = 1.0

    def __call__(self, ctx):
        n = ctx.num_modes
        kappa = jnp.asarray(ctx.kappa_tilde)[:n]
        outer = kappa[:, None] * kappa[None, :]
        # an undamped mode is rejected by the stability gate, not here
        safe = jnp.where(outer > 0.0, outer, 1.0)

        normal = ctx.H[:n, :n]
        if not self.include_detunings:
            normal = normal - jnp.diag(jnp.diag(normal))
        cooperativities = jnp.concatenate([
            (4.0 * jnp.abs(normal) ** 2 / safe).ravel(),
            (4.0 * jnp.abs(ctx.H[:n, n:]) ** 2 / safe).ravel(),
        ])
        excess = cooperativities - self.maximum
        return jnp.sqrt(self.weight) * jnp.maximum(excess, 0.0)

    def __str__(self):
        return "CollectiveCooperativityBudget(C <= %g)" % self.maximum


@dataclass
class DissipativeRateBudget(BaseConstraint):
    """Cap every per-mode damping rate at ``maximum`` reference linewidths.

    :class:`CooperativityBudget` and :class:`CollectiveCooperativityBudget` both
    read ``ctx.H``, so they are budgets on *coherent* pumps only.  On the base
    spec that is the whole story, because the rates are pinned there.  On the
    collective scaffold it leaves a hole: a device can meet the target with no
    Hamiltonian at all -- a purely dissipative solution -- and then no coherent
    budget touches it, while the jump amplitudes that do the work run away.
    Those amplitudes are pumps too: an anomalous (Bogoliubov) dissipator is
    parametrically driven, so an unbounded one is the same ``C -> infinity``
    limit point in a different colour.

    The residual caps the total rate each mode sees from its uncontrolled
    channels -- and, with ``include_controlled``, from its controlled ones -- in
    units of ``reference_mode``'s collected rate, which is the linewidth
    :class:`LinewidthAnchor` fixed.  Use the two together: the anchor says which
    frequency is 1, this says nothing may be more than ``maximum`` of it.
    """

    maximum: float
    reference_mode: int = 0
    include_controlled: bool = True
    weight: float = 1.0

    def __call__(self, ctx):
        reference = jnp.asarray(ctx.kappa_tilde)[int(self.reference_mode)]
        safe = jnp.where(reference > 0.0, reference, 1.0)
        rates = jnp.asarray(ctx.gamma)
        if self.include_controlled:
            rates = jnp.concatenate([rates, jnp.asarray(ctx.kappa_tilde)])
        excess = rates / safe - self.maximum
        return jnp.sqrt(self.weight) * jnp.maximum(excess, 0.0)

    def __str__(self):
        return "DissipativeRateBudget(rates <= %g kappa~_%i)" % (
            self.maximum, self.reference_mode)


@dataclass
class LinewidthAnchor(BaseConstraint):
    """``kappa~_mode = value`` -- declare which linewidth ``Omega`` is read in.

    The forward map sees ``Omega`` only through ``Omega/kappa`` (App. A.4), so
    the collected rate is a stretch factor on the frequency axis.  In the base
    spec it is bounded by ``log_decay_ratio_bound``; on the collective scaffold
    it is a jump amplitude, that bound does not exist, and the scale gauge
    ``(H, K) -> (sH, sK)`` makes the stretch exactly free.

    Any target whose pins constrain the *shape* of a spectrum but not its
    *width* -- a flat band above all (B.4) -- is degenerate under that gauge and
    needs this anchor, which is the collective-engine spelling of
    ``kappa~ == 1``: it fixes the time unit to be mode ``mode``'s own linewidth,
    so a pinned band at ``|Omega| <= W`` means ``W`` of *that* linewidth.  One
    anchor is enough; it removes the one-parameter redundancy, and the remaining
    rates stay free to move relative to it.
    """

    mode: int = 0
    value: float = 1.0
    weight: float = 1.0

    def __call__(self, ctx):
        kappa = jnp.asarray(ctx.kappa_tilde)[int(self.mode)]
        return jnp.array([jnp.sqrt(self.weight) * (kappa - self.value)])

    def __str__(self):
        return "LinewidthAnchor(kappa~_%i == %g)" % (self.mode, self.value)


# ---------------------------------------------------------------------------
# gauge-free conditions on one port block (App. B.3(h))
# ---------------------------------------------------------------------------

def _port_block(ctx, port, omega):
    """The real 2x2 quadrature block of one monitored port at one frequency."""
    k = ctx.omega_index(omega)
    rows = slice(2 * int(port), 2 * int(port) + 2)
    return jnp.real(ctx.V[k][rows, rows])


@dataclass
class PurityFloor(BaseConstraint):
    """``det sigma_out(omega)[port block] = 1`` -- the minimum-added-noise floor.

    In the vacuum-floor-1 normalisation the single-mode purity is
    ``1/sqrt(det)``, so ``det = 1`` is a pure (minimum-uncertainty) emitted
    mode.  It is a *gauge-free* statement about the block, which is why it is a
    constraint rather than a pin on individual entries.

    Its point is adversarial: with a hot intrinsic-loss channel the noise term
    ``N sigma_noise N^dag`` actively pushes ``det > 1``, so holding this
    equality is what the topology has to buy.
    """

    port: int = 0
    omega: float = 0.0
    value: float = 1.0
    weight: float = 1.0

    def __call__(self, ctx):
        block = _port_block(ctx, self.port, self.omega)
        det = block[0, 0] * block[1, 1] - block[0, 1] * block[1, 0]
        return jnp.array([jnp.sqrt(self.weight) * (det - self.value)])

    def __str__(self):
        return "PurityFloor(port=%i, Omega=%g, det=%g)" % (self.port, self.omega,
                                                           self.value)


@dataclass
class QuadratureSpectrum(BaseConstraint):
    """Pin the *eigenvalues* of one port's quadrature block, gauge-free.

    A squeezed output of depth ``r`` has block eigenvalues
    ``(e^{-2r}, e^{+2r})``.  Pinning ``sigma_xx`` and ``sigma_pp`` separately
    would additionally fix the squeeze angle -- only legitimate when the gauge
    is fixed and the target names a quadrature.  Pinning the eigenvalues instead
    is equivalent to pinning the two invariants

        trace = sum(eigenvalues) ,    det = prod(eigenvalues) ,

    which is what this constraint does.  For a pure squeezed state the ``det``
    residual coincides with :class:`PurityFloor`; both are still worth declaring
    because the purity floor is the constraint that must survive the hot bath.
    """

    port: int = 0
    eigenvalues: Tuple[float, float] = (1.0, 1.0)
    omega: float = 0.0
    weight: float = 1.0

    @classmethod
    def squeezed(cls, port, r, omega=0.0, weight=1.0):
        """Depth-``r`` squeezing: eigenvalues ``(e^{-2r}, e^{+2r})``."""
        r = float(r)
        return cls(port=port, eigenvalues=(float(np.exp(-2.0 * r)),
                                           float(np.exp(+2.0 * r))),
                   omega=omega, weight=weight)

    @property
    def r(self):
        """The squeeze depth the eigenvalue pair encodes (``-log(min)/2``)."""
        return -0.5 * float(np.log(min(self.eigenvalues)))

    def __call__(self, ctx):
        block = _port_block(ctx, self.port, self.omega)
        trace = block[0, 0] + block[1, 1]
        det = block[0, 0] * block[1, 1] - block[0, 1] * block[1, 0]
        lo, hi = float(self.eigenvalues[0]), float(self.eigenvalues[1])
        return jnp.sqrt(self.weight) * jnp.array([trace - (lo + hi), det - lo * hi])

    def __str__(self):
        return "QuadratureSpectrum(port=%i, Omega=%g, eig=(%g, %g))" % (
            self.port, self.omega, self.eigenvalues[0], self.eigenvalues[1])



@dataclass
class MinimumIntrinsicLoss(BaseConstraint):
    """One-sided floor ``gamma_k >= minimum`` on one mode's intrinsic loss.

    Needed whenever a *hot* bath is declared on channel ``k``.  The intrinsic
    loss has to stay a live optimisation variable -- a channel with no rate
    cannot carry a bath -- but a free ``gamma_k`` gives the optimiser a trivial
    escape: send ``gamma_k -> 0`` and the bath is disconnected rather than
    defeated, which is not the problem that was posed.  The floor is the
    hardware statement that the loss is a real defect of a given size; the
    optimiser may still trade ``gamma_k`` upward.

    Setting ``minimum = 0`` restores the degenerate target on purpose, which is
    how the "switch the channel off" escape is demonstrated rather than assumed.
    """

    mode: int
    minimum: float
    weight: float = 1.0

    def __call__(self, ctx):
        deficit = self.minimum - ctx.gamma[int(self.mode)]
        return jnp.array([jnp.sqrt(self.weight) * jnp.maximum(deficit, 0.0)])

    def __str__(self):
        return "MinimumIntrinsicLoss(mode=%i, gamma >= %g)" % (self.mode, self.minimum)


@dataclass
class CustomConstraint(BaseConstraint):
    """Wrap any ``ctx -> residual array`` callable."""

    func: Callable
    name: str = "custom"

    def __call__(self, ctx):
        return jnp.atleast_1d(self.func(ctx))

    def __str__(self):
        return "CustomConstraint(%s)" % self.name
