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
class CustomConstraint(BaseConstraint):
    """Wrap any ``ctx -> residual array`` callable."""

    func: Callable
    name: str = "custom"

    def __call__(self, ctx):
        return jnp.atleast_1d(self.func(ctx))

    def __str__(self):
        return "CustomConstraint(%s)" % self.name
