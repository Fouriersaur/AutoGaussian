"""
Worked target gallery (App. B of the algorithm flow).

Every entry returns a :class:`GalleryProblem` bundling
``(target, sigma_in builder, extra constraints, notes)`` so it can be fed
straight into :class:`autogaussian.optimizer.CovarianceArchitectureOptimizer`.

Convention: a number is a *pinned* entry, ``None`` is free (``*``), quadrature
basis ``(x_1, p_1, x_2, p_2, ...)``, vacuum floor = 1.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from autogaussian.constraints import (
    IsolationConstraint,
    MinimumTransmission,
    TransmissionConstraint,
)
from autogaussian.nambu import channel_covariance
from autogaussian.target import CovarianceTarget, P, X, qidx

__all__ = [
    "GalleryProblem",
    "single_mode_squeezer",
    "epr_source",
    "directional_squeezed_source",
    "broadband_squeezer",
    "cv_graph_state",
    "backaction_evading_readout",
    "noise_diode",
    "GALLERY",
]


@dataclass
class GalleryProblem:
    target: CovarianceTarget
    sigma_in_signal: Optional[Callable] = None      # num_modes -> (2N,2N) array
    constraints: Sequence[Any] = field(default_factory=tuple)
    notes: str = ""
    suggested: Dict[str, Any] = field(default_factory=dict)

    def optimizer_kwargs(self):
        kwargs = {"sigma_in_signal": self.sigma_in_signal,
                  "constraints": tuple(self.constraints)}
        kwargs.update(self.suggested)
        return kwargs


def _thermal_input(occupations):
    """Build a ``num_modes -> sigma_in`` callable from per-port occupations."""
    def builder(num_modes):
        n = np.zeros(num_modes)
        n[: len(occupations)] = np.asarray(occupations, dtype=float)
        return channel_covariance(n=n, m=0.0, num_modes=num_modes)
    return builder


# --------------------------------------------------------------------------
# B.1 single-mode squeezer (validation)
# --------------------------------------------------------------------------

def single_mode_squeezer(variance=0.5, pin_antisqueezed=True):
    """3 dB squeezing in ``x`` at band centre; the conjugate quadrature is
    pinned to ``1/variance`` for a *pure* output when requested."""
    target = CovarianceTarget(num_ports=1, name="single-mode squeezer")
    target.pin((qidx(0, X), qidx(0, X)), variance)
    target.pin((qidx(0, X), qidx(0, P)), 0.0)
    if pin_antisqueezed:
        target.pin((qidx(0, P), qidx(0, P)), 1.0 / variance)
    return GalleryProblem(
        target,
        notes="B.1  loss pins both diagonals (pure state) or just sigma_xx.",
        suggested={"optimize_gauge": False},
    )


# --------------------------------------------------------------------------
# B.2 EPR / two-mode-squeezed source
# --------------------------------------------------------------------------

def epr_source(local=1.25, correlation=0.75):
    """Local ports thermal, pinned correlations squeeze the joint quadratures.

    ``local = 1.25``, ``correlation = 0.75`` gives joint variances 0.5, i.e.
    Duan sum 1.0 < 2 -- certified entangled.
    """
    target = CovarianceTarget(num_ports=2, name="EPR source")
    matrix = [[local, 0.0, correlation, 0.0],
              [0.0, local, 0.0, -correlation],
              [correlation, 0.0, local, 0.0],
              [0.0, -correlation, 0.0, local]]
    target.pin_matrix(matrix)
    return GalleryProblem(
        target,
        notes="B.2  two-mode squeezed vacuum; local thermal, joint squeezed.",
        suggested={"optimize_gauge": False},
    )


# --------------------------------------------------------------------------
# B.3 directional squeezed source
# --------------------------------------------------------------------------

def directional_squeezed_source(variance=0.5, forward_transmission=None,
                                isolate_backward=False):
    """Port 1 pinned to vacuum, port 2 to squeezing -- the *asymmetry between
    the diagonal blocks is the target*.

    Achievable at finite cooperativity.  Pass ``forward_transmission`` to forbid
    the trivial decoupled solution (Sec. 11); pass ``isolate_backward=True`` to
    additionally silence the backward direction -- *that* is the part which
    needs a symmetry-breaking dissipative auxiliary, since two directly coupled
    modes cannot be made directional by any coupling phase alone (App. F).
    """
    target = CovarianceTarget(num_ports=2, name="directional squeezed source")
    matrix = [[1.0, 0.0, None, None],
              [0.0, 1.0, None, None],
              [None, None, variance, 0.0],
              [None, None, 0.0, 1.0 / variance]]
    target.pin_matrix(matrix)
    constraints = []
    if forward_transmission is not None:
        constraints.append(TransmissionConstraint(port_out=1, port_in=0,
                                                  value=float(forward_transmission)))
    if isolate_backward:
        constraints.append(IsolationConstraint(port_out=0, port_in=1))
    constraints = tuple(constraints)
    return GalleryProblem(
        target, constraints=constraints,
        notes="B.3  finite-cooperativity directional source; add a transmission "
              "pin so the minimal solution is a single connected device.",
        suggested={"optimize_gauge": False},
    )


# --------------------------------------------------------------------------
# B.4 broadband squeezer (point + derivatives)
# --------------------------------------------------------------------------

def broadband_squeezer(variance=0.5, num_derivatives=2, port=0, bandwidth=None,
                       num_band_points=4):
    """Flat-band condition: pin ``sigma_xx(0)`` and its first ``k``
    ``Omega``-derivatives to zero.

    **Pin a ``bandwidth`` as well.**  Derivative pins alone constrain the
    spectrum only *at* ``Omega = 0``, and the optimiser can satisfy them by
    shrinking ``kappa~`` until the curvature vanishes on a linewidth far
    narrower than the band you meant -- flat at a point, not flat across a
    band.  Passing ``bandwidth=W`` additionally pins ``sigma_xx = variance`` on
    a grid out to ``+-W``, which is the condition you actually want (the
    "full-spectrum" flavour of Sec. 1.1).
    """
    target = CovarianceTarget(num_ports=port + 1, name="broadband squeezer")
    entry = (qidx(port, X), qidx(port, X))
    target.pin(entry, variance)
    for order in range(1, int(num_derivatives) + 1):
        target.pin_derivative(entry, order=order, value=0.0)
    if bandwidth is not None:
        grid = np.linspace(-float(bandwidth), float(bandwidth), int(num_band_points) + 1)
        grid = grid[grid != 0.0]
        target.pin_spectrum(entry, grid, variance)
    return GalleryProblem(
        target,
        notes="B.4  derivative pins flatten the dip; decay ratios kappa~_i are "
              "the parameters that do the flattening.  Pin a bandwidth too, or "
              "the flatness is only local.",
        suggested={"optimize_gauge": False, "free_decay_ratios": True},
    )


# --------------------------------------------------------------------------
# B.5 CV graph-state emitter
# --------------------------------------------------------------------------

def cv_graph_state(edges=((0, 1),), epsilon=0.1, num_ports=2):
    """Pin one nullifier variance per graph edge:
    ``Var(p_i - x_j) = epsilon``."""
    target = CovarianceTarget(num_ports=num_ports, name="CV graph state")
    for i, j in edges:
        vector = np.zeros(2 * num_ports)
        vector[qidx(i, P)] = 1.0
        vector[qidx(j, X)] = -1.0
        target.pin_form(vector, epsilon)
    return GalleryProblem(
        target,
        notes="B.5  nullifier variances -> emitted cluster state.",
        suggested={"optimize_gauge": False},
    )


# --------------------------------------------------------------------------
# B.6 backaction-evading readout
# --------------------------------------------------------------------------

def backaction_evading_readout(signal_port=0, probe_port=1):
    """Silence one cross-term: ``sigma_{p_signal, x_probe} = 0``, rest free."""
    num_ports = max(signal_port, probe_port) + 1
    target = CovarianceTarget(num_ports=num_ports, name="backaction-evading readout")
    target.pin((qidx(signal_port, P), qidx(probe_port, X)), 0.0)
    return GalleryProblem(
        target,
        constraints=(MinimumTransmission(port_out=probe_port, port_in=signal_port,
                                         minimum=0.25),),
        notes="B.6  back-action noise into the probe silenced; the transmission "
              "floor keeps the readout coupled.",
    )


# --------------------------------------------------------------------------
# B.7 noise diode (needs non-vacuum input)
# --------------------------------------------------------------------------

def noise_diode(n_thermal=1.0):
    """Thermal input at port 2, port-1 output pinned to vacuum.

    Directionality shows up *only* because ``sigma_in != 1``.
    """
    target = CovarianceTarget(num_ports=2, name="noise diode")
    target.pin((qidx(0, X), qidx(0, X)), 1.0)
    target.pin((qidx(0, P), qidx(0, P)), 1.0)
    target.pin((qidx(0, X), qidx(0, P)), 0.0)
    return GalleryProblem(
        target,
        sigma_in_signal=_thermal_input([0.0, n_thermal]),
        constraints=(MinimumTransmission(port_out=1, port_in=0, minimum=0.1),),
        notes="B.7  declared thermal input at port 2; port 1 stays at the "
              "vacuum floor while forward transport survives.",
        suggested={"optimize_gauge": False},
    )


GALLERY = {
    "B.1": single_mode_squeezer,
    "B.2": epr_source,
    "B.3": directional_squeezed_source,
    "B.4": broadband_squeezer,
    "B.5": cv_graph_state,
    "B.6": backaction_evading_readout,
    "B.7": noise_diode,
}
