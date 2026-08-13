"""
INPUT -- problem specification (Sec. 1 of the algorithm flow).

A target is a set of *pinned entries* of the output covariance
``sigma_out(Omega)`` on a frequency set; everything not pinned is free (``*``).
Three flavours, all expressible here:

* single point       -- ``pin(entry, value, omega=0)``
* point + derivatives-- ``pin_derivative(entry, order=m, value=..)`` (flat band)
* full spectrum      -- ``pin_spectrum(entry, omegas, values)``

Entries index the interleaved quadrature basis ``(x_1, p_1, x_2, p_2, ...)``
of the *ports*; use :func:`qidx` to build them from ``(port, quadrature)``.

Pinned values may be numbers or sympy expressions; free symbols in them become
optimisation variables (the analogue of AUTOSCATTER's free symbols in
``S_target``), which is how "any 3 dB squeezing, don't care about the
anti-squeezed quadrature" style targets are written.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import sympy as sp

__all__ = ["Pin", "CovarianceTarget", "qidx", "X", "P"]

X = 0
P = 1

PART_FULL = "full"
PART_REAL = "real"
PART_IMAG = "imag"


def qidx(port, quadrature=X):
    """Row/column index of quadrature ``x`` (0) or ``p`` (1) of ``port``."""
    return 2 * int(port) + int(quadrature)


@dataclass
class Pin:
    """One pinned entry of ``sigma_out``."""

    row: int
    col: int
    value: Any = 0.0            # number or sympy expression
    omega: float = 0.0
    weight: float = 1.0
    order: int = 0              # d^order / dOmega^order
    part: str = PART_FULL       # 'full' | 'real' | 'imag'
    form: Optional[Any] = None  # real vector v -> pins the variance v^T V v

    def __str__(self):
        if self.form is not None:
            return "Var(v.R)(Omega=%g) = %s   v=%s  (w=%g)" % (
                self.omega, self.value, np.array2string(np.asarray(self.form),
                                                        precision=3), self.weight)
        name = "sigma_out"
        if self.order:
            name = "d^%i/dOmega^%i " % (self.order, self.order) + name
        return "%s[%i,%i](Omega=%g) = %s   (w=%g, %s)" % (
            name, self.row, self.col, self.omega, self.value, self.weight, self.part)


class CovarianceTarget:
    """Container of pins over the port quadrature covariance.

    Parameters
    ----------
    num_ports : int
        Number of monitored signal ports ``P``; the target lives on the
        ``2P x 2P`` port block.
    name : str
    """

    def __init__(self, num_ports, name=""):
        self.num_ports = int(num_ports)
        self.name = name
        self.pins = []

    # -- construction ------------------------------------------------------

    def _check(self, row, col):
        limit = 2 * self.num_ports
        if not (0 <= row < limit and 0 <= col < limit):
            raise ValueError(
                "entry (%i,%i) outside the %ix%i port block" % (row, col, limit, limit))

    def pin(self, entry, value, omega=0.0, weight=1.0, part=PART_FULL):
        """Pin ``sigma_out(omega)[entry] = value``."""
        row, col = entry
        self._check(row, col)
        self.pins.append(Pin(row=row, col=col, value=value, omega=float(omega),
                             weight=float(weight), order=0, part=part))
        return self

    def pin_derivative(self, entry, order, value=0.0, omega=0.0, weight=1.0,
                       part=PART_FULL):
        """Pin ``d^order sigma_out / dOmega^order`` at ``omega`` (broadband /
        flat-band condition, Sec. 1.1 and App. B.4)."""
        row, col = entry
        self._check(row, col)
        self.pins.append(Pin(row=row, col=col, value=value, omega=float(omega),
                             weight=float(weight), order=int(order), part=part))
        return self

    def pin_spectrum(self, entry, omegas, values, weight=1.0, part=PART_FULL):
        """Pin one entry against a target function on a dense grid."""
        values = np.broadcast_to(np.asarray(values, dtype=complex), np.shape(omegas))
        for omega, value in zip(np.asarray(omegas), values):
            self.pin(entry, complex(value), omega=omega, weight=weight, part=part)
        return self

    def pin_form(self, vector, value, omega=0.0, weight=1.0):
        """Pin a quadratic form ``v^T sigma_out(omega) v = value``.

        This is how joint-quadrature conditions are written: nullifier
        variances ``Var(p_i - x_j)`` (App. B.5), Duan sums, EPR variances --
        none of which is a single matrix entry.
        """
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (2 * self.num_ports,):
            raise ValueError("form vector must have length %i" % (2 * self.num_ports))
        self.pins.append(Pin(row=-1, col=-1, value=value, omega=float(omega),
                             weight=float(weight), order=0, part=PART_REAL,
                             form=vector))
        return self

    def pin_matrix(self, matrix, omega=0.0, weights=None, part=PART_FULL):
        """Pin a whole ``2P x 2P`` matrix; ``None`` / ``nan`` / ``'*'`` entries
        are left free (App. B convention)."""
        matrix = np.asarray(matrix, dtype=object)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                if value is None or (isinstance(value, str) and value.strip() == "*"):
                    continue
                if isinstance(value, float) and np.isnan(value):
                    continue
                weight = 1.0 if weights is None else float(np.asarray(weights)[row, col])
                if weight == 0.0:
                    continue
                self.pin((row, col), value, omega=omega, weight=weight, part=part)
        return self

    # -- introspection -----------------------------------------------------

    @property
    def free_symbols(self):
        symbols = set()
        for pin in self.pins:
            if isinstance(pin.value, sp.Expr):
                symbols |= pin.value.free_symbols
        return sorted(symbols, key=lambda s: s.name)

    @property
    def omegas(self):
        """Sorted unique frequencies appearing in the pins."""
        return np.unique(np.array([pin.omega for pin in self.pins], dtype=float))

    def __len__(self):
        return len(self.pins)

    def __str__(self):
        head = "CovarianceTarget(%s, %i ports, %i pins)" % (
            self.name or "unnamed", self.num_ports, len(self.pins))
        return "\n".join([head] + ["  " + str(pin) for pin in self.pins])

    def describe(self):
        print(str(self))

    # -- convenience -------------------------------------------------------

    def matrix_at(self, omega=0.0, fill=np.nan):
        """The pinned entries at one frequency, as a matrix with ``fill`` for
        free entries (order-0 pins only)."""
        out = np.full((2 * self.num_ports, 2 * self.num_ports), fill, dtype=object)
        for pin in self.pins:
            if pin.form is None and pin.order == 0 and np.isclose(pin.omega, omega):
                out[pin.row, pin.col] = pin.value
        return out
