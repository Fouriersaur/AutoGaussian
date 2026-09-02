"""
Worked target gallery (App. B of the algorithm flow).

Every entry returns a :class:`GalleryProblem` bundling
``(target, sigma_in builder, extra constraints, notes)`` so it can be fed
straight into :class:`autogaussian.optimizer.CovarianceArchitectureOptimizer`.

Convention: a number is a *pinned* entry, ``None`` is free (``*``), quadrature
basis ``(x_1, p_1, x_2, p_2, ...)``, vacuum floor = 1.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp

from autogaussian.constraints import (
    IsolationConstraint,
    MinimumIntrinsicLoss,
    MinimumTransmission,
    PurityFloor,
    QuadratureSpectrum,
    TransmissionConstraint,
)
from autogaussian.forward import output_covariance_quadrature
from autogaussian.nambu import (
    build_H_bdg,
    channel_covariance,
    squeezed_bath,
    thermal_channel_covariance,
)
from autogaussian.target import PART_REAL, CovarianceTarget, P, X, qidx

__all__ = [
    "GalleryProblem",
    "single_mode_squeezer",
    "epr_source",
    "directional_squeezed_source",
    "thermal_directional_squeezer",
    "hot_channel_inflation",
    "broadband_squeezer",
    "cv_graph_state",
    "backaction_evading_readout",
    "noise_diode",
    # frequency-dependent (full-spectrum) targets, App. B.8 - B.12
    "filter_cavity_angle",
    "rotated_squeezing_block",
    "lorentzian_dip",
    "butterworth_dip",
    "notch_factor",
    "squeeze_angle_rotation",
    # rotation family, Sec. 8.7 (extensions of B.8)
    "rotation_device",
    "rotation_spectrum",
    "squeeze_angle",
    "rotation_purity",
    "accumulated_rotation",
    "single_mode_winding_ceiling",
    "impure_rotation",
    "non_monotone_rotation",
    "winding_rotation",
    "bandpass_squeezer",
    "flat_top_squeezer",
    "band_limited_epr",
    "duan_of_band_limited_epr",
    "epr_bandwidth",
    "notch_squeezer",
    "GALLERY",
    "SPECTRAL_GALLERY",
    "ROTATION_FAMILY",
]


@dataclass
class GalleryProblem:
    target: CovarianceTarget
    sigma_in_signal: Optional[Callable] = None      # num_modes -> (2N,2N) array
    sigma_in_noise: Optional[Callable] = None       # num_modes -> (2N,2N) array
    constraints: Sequence[Any] = field(default_factory=tuple)
    notes: str = ""
    suggested: Dict[str, Any] = field(default_factory=dict)
    kwargs_optimization: Dict[str, Any] = field(default_factory=dict)

    def optimizer_kwargs(self, **overrides):
        """Keyword arguments for :class:`CovarianceArchitectureOptimizer`.

        ``overrides`` are merged on top; a ``kwargs_optimization`` override is
        merged *into* the suggested one rather than replacing it, so a caller
        can add restarts without losing the fit tolerance a spectral target
        needs (the shape targets B.9-B.12 are matched to ~1e-8, not 1e-10).
        """
        kwargs = {"sigma_in_signal": self.sigma_in_signal,
                  "sigma_in_noise": self.sigma_in_noise,
                  "constraints": tuple(self.constraints)}
        kwargs.update(self.suggested)
        if self.kwargs_optimization:
            kwargs["kwargs_optimization"] = dict(self.kwargs_optimization)
        for key, value in overrides.items():
            if key == "kwargs_optimization":
                merged = dict(kwargs.get("kwargs_optimization", {}))
                merged.update(value)
                kwargs[key] = merged
            else:
                kwargs[key] = value
        return kwargs


def _thermal_input(occupations):
    """Build a ``num_modes -> sigma_in`` callable from per-port occupations."""
    def builder(num_modes):
        n = np.zeros(num_modes)
        n[: len(occupations)] = np.asarray(occupations, dtype=float)
        return channel_covariance(n=n, m=0.0, num_modes=num_modes)
    return builder


def _squeezed_input(r, phi=0.0, port=0):
    """Build a ``num_modes -> sigma_in`` callable with one squeezed channel."""
    def builder(num_modes):
        n = np.zeros(num_modes)
        m = np.zeros(num_modes, dtype=complex)
        n[port], m[port] = squeezed_bath(r, phi)
        return channel_covariance(n=n, m=m, num_modes=num_modes)
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
# B.3(h) thermal-resilient directional squeezer -- hot intrinsic-loss channel
# --------------------------------------------------------------------------

def thermal_directional_squeezer(
    r=0.8,
    n_thermal=0.5,
    hot_channel=2,
    monitored_port=1,
    vacuum_port=0,
    forward_transmission=1.3,
    isolate_backward=True,
    min_loss=0.1,
    bandwidth=None,
    num_band_points=4,
    intrinsic_losses=None,
):
    """Directional squeezer with a **hot intrinsic-loss channel** -- the
    ``sigma_out(Omega)``-native twin of the "quantum-limited amplifier with a hot
    output port" problem (AUTOSCATTER Sec. VI).

    The device is the B.3 directional squeezer: ``monitored_port`` emits
    squeezing of depth ``r``, ``vacuum_port`` sits at vacuum, transport is
    nonreciprocal.  The new content is the *adversary*: the intrinsic-loss
    channel of mode ``hot_channel`` is held at thermal occupation
    ``n_thermal > 0``, and the monitored port is nevertheless required to stay on
    the purity floor ``det sigma_out = 1``.

    Where the bath enters
    ---------------------
    Splitting the output covariance into its signal and noise contributions,

        sigma_out = S sigma_signal S^dag + N sigma_noise N^dag ,
        N(Omega)  = chi(Omega) sqrt(gamma) ,

    the squeezing lives in the first term and intrinsic-loss channels enter only
    through the second.  A cold channel contributes the identity; occupation
    ``n`` on channel ``k*`` inflates the monitored port block by

        d sigma_out,jj(Omega) = 2 n [ N(Omega) e_{k*} N(Omega)^dag ]_{jj} ,

    so the contamination is governed entirely by the susceptibility
    ``N_{j,k*}(Omega)`` -- the covariance-resolved form of AUTOSCATTER's
    added-noise sum.  Inspect it with
    :func:`autogaussian.forward.noise_response_block`; it is deliberately *not*
    a fit term.

    Conventions and reductions
    --------------------------
    * The squeezing is pinned **gauge-free**, as the eigenvalue pair
      ``(e^{-2r}, e^{+2r})`` of the port block (:class:`QuadratureSpectrum`),
      not as ``sigma_xx`` / ``sigma_pp`` separately.
    * ``det sigma_out = 1`` on the *monitored* block is declared explicitly
      (:class:`PurityFloor`) and is evaluated with the thermal occupation
      already present -- that is the whole point of the target.
    * The hot channel must exist *and stay open*: mode ``hot_channel`` gets a
      live intrinsic loss ``gamma``, floored at ``min_loss``.  The floor is not
      cosmetic -- with ``min_loss = 0`` the optimiser defeats the bath by
      sending ``gamma -> 0``, i.e. by disconnecting the channel instead of
      surviving it, and the target becomes the cold one in disguise.
    * ``n_thermal = 0, min_loss = 0`` reduces this exactly to the cold B.3
      directional squeezer with ``variance = e^{-2r}`` (up to the squeeze angle,
      which the gauge-free pin leaves free).  The honest *baseline* for "what
      does resilience cost" is instead ``n_thermal = 0`` at the same
      ``min_loss``: same open loss channel, cold bath.

    Parameters
    ----------
    bandwidth : float, optional
        **Off by default on purpose.**  Holding purity at a single frequency is
        generic; holding it across a finite band is not.  Only switch this on
        after a forward-map sweep shows ``det sigma_out(Omega) = 1`` is holdable
        at several grid points on some graph -- otherwise the band target is a
        conjecture wearing a fit tolerance.
    hot_channel : int
        Which intrinsic-loss channel is held at ``n_thermal``.  Defaults to
        mode 2, the first auxiliary -- see the obstruction note above before
        moving it onto a port mode.
    min_loss : float
        Floor on the hot channel's intrinsic loss (:class:`MinimumIntrinsicLoss`).
        Pass ``0.0`` to switch the floor off and reproduce the degenerate
        ``gamma -> 0`` escape.
    intrinsic_losses : bool or sequence of bool, optional
        Which modes carry a live ``gamma``.  Defaults to the hot channel alone.
    """
    r = float(r)
    n_thermal = float(n_thermal)
    hot_channel = int(hot_channel)
    monitored_port, vacuum_port = int(monitored_port), int(vacuum_port)
    if monitored_port == vacuum_port:
        raise ValueError("monitored and vacuum port must differ")

    if bandwidth is None:
        omegas = np.array([0.0])
    else:
        omegas = _spectral_grid(float(bandwidth), int(num_band_points))

    target = CovarianceTarget(
        num_ports=2,
        name="thermal-resilient directional squeezer (n=%.3g on channel %i)"
             % (n_thermal, hot_channel))

    # port sitting at vacuum -- pinned entry by entry, which also puts every
    # band frequency into the oracle's frequency set (the constraints below are
    # evaluated on that set)
    v0, v1 = qidx(vacuum_port, X), qidx(vacuum_port, P)
    for omega in omegas:
        target.pin((v0, v0), 1.0, omega=omega)
        target.pin((v1, v1), 1.0, omega=omega)
        target.pin((v0, v1), 0.0, omega=omega)
        target.pin((v1, v0), 0.0, omega=omega)

    constraints = []
    for omega in omegas:
        # squeezing depth, gauge-free ...
        constraints.append(QuadratureSpectrum.squeezed(monitored_port, r, omega=omega))
        # ... and the purity floor the hot bath is trying to break
        constraints.append(PurityFloor(port=monitored_port, omega=omega))
    if forward_transmission is not None:
        constraints.append(TransmissionConstraint(port_out=monitored_port,
                                                  port_in=vacuum_port,
                                                  value=float(forward_transmission)))
    if isolate_backward:
        constraints.append(IsolationConstraint(port_out=vacuum_port,
                                               port_in=monitored_port))
    if float(min_loss) > 0.0:
        constraints.append(MinimumIntrinsicLoss(mode=hot_channel,
                                                minimum=float(min_loss)))

    if intrinsic_losses is None:
        intrinsic_losses = [i == hot_channel for i in range(hot_channel + 1)]

    sigma_in_noise = None
    if n_thermal != 0.0:
        sigma_in_noise = lambda num_modes: thermal_channel_covariance(
            {hot_channel: n_thermal}, num_modes)

    return GalleryProblem(
        target,
        sigma_in_noise=sigma_in_noise,
        constraints=tuple(constraints),
        notes="B.3(h)  directional squeezer whose port-%i-side intrinsic-loss "
              "channel is held at n = %.3g; the monitored port %i must keep "
              "depth r = %.3g *and* det sigma_out = 1 despite it."
              % (vacuum_port, n_thermal, monitored_port, r),
        suggested={"optimize_gauge": False,
                   "intrinsic_losses": intrinsic_losses},
    )


def hot_channel_inflation(n_thermal, response_block):
    """``2 n * [N e_k N^dag]`` -- the predicted inflation of a monitored port
    block when its designated loss channel goes from vacuum to occupation ``n``.

    ``response_block`` is what
    :func:`autogaussian.forward.noise_response_block` returns.  Kept next to the
    target so the demonstration step ("the textbook graph must fail purity on
    purpose") can be written as one line.
    """
    return 2.0 * float(n_thermal) * np.asarray(response_block, dtype=float)


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

def cv_graph_state(edges=((0, 1),), epsilon=0.1, num_ports=2, nullifiers="per_node"):
    """Emitted CV cluster state defined by a graph ``edges``.

    A CV graph state with adjacency ``A`` is the state annihilated by one
    nullifier **per node**,

        n_i = p_i - sum_j A_ij x_j ,      Var(n_i) = epsilon  ->  0

    so an ``M``-edge graph on ``P`` modes contributes ``P`` pins, not ``M``.
    Pinning only *one* nullifier is badly under-specified: it is satisfiable by
    two **decoupled** squeezers (squeeze ``p_0`` in one mode and ``x_1`` in the
    other), which is not a cluster state at all -- the conjugate combination
    then blows up to keep the state pure.  See Sec. 11.

    ``nullifiers='per_edge'`` restores the older, looser behaviour of pinning
    one ``Var(p_i - x_j)`` per edge, for comparison.
    """
    target = CovarianceTarget(num_ports=num_ports, name="CV graph state")

    if nullifiers == "per_edge":
        for i, j in edges:
            vector = np.zeros(2 * num_ports)
            vector[qidx(i, P)] = 1.0
            vector[qidx(j, X)] = -1.0
            target.pin_form(vector, epsilon)
    else:
        adjacency = np.zeros((num_ports, num_ports))
        for i, j in edges:
            adjacency[i, j] = adjacency[j, i] = 1.0
        for i in range(num_ports):
            if not adjacency[i].any():
                continue
            vector = np.zeros(2 * num_ports)
            vector[qidx(i, P)] = 1.0
            for j in range(num_ports):
                if adjacency[i, j]:
                    vector[qidx(j, X)] -= 1.0
            target.pin_form(vector, epsilon)

    return GalleryProblem(
        target,
        notes="B.5  one nullifier per node -> emitted cluster state.",
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


# ==========================================================================
# Frequency-dependent (full-spectrum) targets -- App. B.8 - B.12
#
# B.1-B.7 pin entries at a single Omega (or, in B.4, a point plus derivatives),
# and for all of them the decay ratios kappa~_i are inert (App. A.4(c)).  The
# five targets below are pinned on a *grid* and carry genuine spectral shape,
# so kappa~_i -- and usually the detunings -- become live variables.  That is
# the whole point of them; every one of these problems is built with
# ``free_decay_ratios=True``.
#
# One convention worth stating once.  The symmetrised output spectrum obeys
#
#       V(-Omega) = V(Omega)^T ,
#
# so every *diagonal* entry sigma_xx, sigma_pp is automatically an even, real
# function of Omega, and so is the real part of sigma_xp.  Pinning a target at
# both +Omega and -Omega is therefore redundant, and pinning a shape that is
# *not* even (a Lorentzian centred at Omega_c > 0 and nothing at -Omega_c) asks
# for something no device can emit.  All grids below run over Omega >= 0 and
# the shapes are read as functions of the sideband frequency |Omega|.
# ==========================================================================


def filter_cavity_angle(omega, crossover=1.0):
    """Squeeze-angle rotation ``theta(Omega)`` of one detuned lossless mode.

    A single passive mode at detuning ``Delta/kappa = 1/2`` reflects a squeezed
    input with a *frequency-dependent* ellipse angle that sweeps exactly
    ``0 -> pi/2`` as ``|Omega|`` grows (App. B.8).  The reflection phase of the
    two sidebands is ``phi_pm = pi + 2 arctan(2(Omega +- Delta kappa~)/kappa~)``
    and the ellipse angle is half their difference, which for ``Delta = 1/2`` is

        theta(Omega) = pi/2 - arctan(t + 1) + arctan(t - 1) ,   t = 2 Omega/kappa~

    with ``theta(0) = 0``, ``theta(inf) = pi/2`` and ``theta = pi/4`` at the
    crossover ``Omega_c = kappa~/sqrt(2)`` -- the argument of this function.

    ``Delta/kappa = 1/2`` is not a free choice: ``theta(0) = 0`` requires
    ``4 arctan(2 Delta/kappa) = pi``.  Any other detuning gives a partial sweep.
    """
    t = np.sqrt(2.0) * np.asarray(omega, dtype=float) / float(crossover)
    return 0.5 * np.pi - np.arctan(t + 1.0) + np.arctan(t - 1.0)


def rotated_squeezing_block(r, theta):
    """``R(theta) diag(e^{-2r}, e^{+2r}) R(theta)^T`` entrywise.

    Returns ``(sigma_xx, sigma_pp, sigma_xp)``:

        sigma_xx = cosh 2r - sinh 2r cos 2theta
        sigma_pp = cosh 2r + sinh 2r cos 2theta
        sigma_xp =           sinh 2r sin 2theta

    The ellipse *eigenvalues* are ``(e^{-2r}, e^{+2r})`` for every ``theta``:
    the angle rotates the state, it does not change its depth.
    """
    theta = np.asarray(theta, dtype=float)
    return (np.cosh(2 * r) - np.sinh(2 * r) * np.cos(2 * theta),
            np.cosh(2 * r) + np.sinh(2 * r) * np.cos(2 * theta),
            np.sinh(2 * r) * np.sin(2 * theta))


def lorentzian_dip(omega, depth=0.5, center=0.0, bandwidth=1.0):
    """``1 - (1 - s_0) / (1 + ((Omega - Omega_c)/B)^2)`` (App. B.9)."""
    omega = np.asarray(omega, dtype=float)
    return 1.0 - (1.0 - depth) / (1.0 + ((omega - center) / bandwidth) ** 2)


def butterworth_dip(omega, depth=0.5, bandwidth=1.0, order=1):
    """``1 - (1 - s_0) / (1 + (Omega/B)^{2n})`` -- order-``n`` flat top (B.10).

    The first ``2n - 1`` Omega-derivatives vanish at ``Omega = 0``, so B.4's
    derivative pins are the low-order shadow of this shape.
    """
    omega = np.asarray(omega, dtype=float)
    return 1.0 - (1.0 - depth) / (1.0 + (omega / bandwidth) ** (2 * int(order)))


def notch_factor(omega, notch_center=1.0, notch_width=0.2):
    """``1 - gamma_n^2 / ((Omega - Omega_0)^2 + gamma_n^2)`` (App. B.12).

    Multiplies a squeezing *enhancement*: it is 0 at the notch line (no
    squeezing there, output back at the vacuum floor) and 1 far from it.
    """
    omega = np.asarray(omega, dtype=float)
    return 1.0 - notch_width ** 2 / ((omega - notch_center) ** 2 + notch_width ** 2)


def _spectral_grid(omega_max, num_points, include_zero=True):
    start = 0.0 if include_zero else omega_max / float(num_points)
    return np.linspace(start, float(omega_max), int(num_points))


# --------------------------------------------------------------------------
# B.8 frequency-dependent squeeze-angle rotation (filter cavity)
# --------------------------------------------------------------------------

def squeeze_angle_rotation(r=0.6, crossover=1.0, omega_max=None, num_points=9,
                           pin_cross_term=True, theta=None):
    """Fixed squeeze *magnitude*, rotating squeeze *axis* -- the GW-detector
    frequency-dependent-squeezing (FDS) target.

    The whole target *is* the rotation: it lives in the off-diagonal
    ``sigma_xp(Omega)`` (zero at both band edges, extremal at the crossover) and
    in the two diagonals swapping as ``theta`` crosses ``pi/4``.  It is
    **invisible to any phase-preserving (N x N) reduction**, which makes it the
    sharpest justification for the full Nambu treatment -- and, as a test, the
    regression guard that the forward map really is phase-sensitive.

    Realised by a *passive* device: one lossless mode at ``Delta/kappa = 1/2``
    reflecting the declared ``r``-squeezed input (the optical filter cavity).
    The output is then exactly pure (``det sigma_out = 1`` at every ``Omega``)
    with ``Omega``-independent ellipse eigenvalues ``(e^{-2r}, e^{+2r})`` -- a
    lossless passive element rotates the ellipse without touching its depth.

    Parameters
    ----------
    r : float
        Squeeze parameter of the *declared input* (App. B.8 works on a squeezed
        input; the same spectrum can also be built actively from vacuum).
    crossover : float
        ``Omega_c`` where ``theta = pi/4``, i.e. where the squeezed and
        anti-squeezed quadratures have swapped halfway.
    theta : callable, optional
        ``Omega -> theta(Omega)``; defaults to :func:`filter_cavity_angle`,
        the profile a single detuned mode actually produces.  The *closed form*
        of ``theta(Omega)`` is a symbolic-regression output (Sec. 7.2), never a
        fit input -- it is used here only to write down a realisable target.
    pin_cross_term : bool
        Pin ``sigma_xp`` as well as the two diagonals.  Leave it on: the
        rotation is exactly what the cross term encodes.
    """
    if omega_max is None:
        omega_max = 4.0 * crossover
    theta_of = filter_cavity_angle if theta is None else theta
    grid = _spectral_grid(omega_max, num_points)
    angles = np.asarray([theta_of(w, crossover) if theta is None else theta_of(w)
                         for w in grid], dtype=float)
    sigma_xx, sigma_pp, sigma_xp = rotated_squeezing_block(r, angles)

    target = CovarianceTarget(num_ports=1, name="frequency-dependent squeeze rotation")
    target.pin_spectrum((qidx(0, X), qidx(0, X)), grid, sigma_xx, part=PART_REAL)
    target.pin_spectrum((qidx(0, P), qidx(0, P)), grid, sigma_pp, part=PART_REAL)
    if pin_cross_term:
        target.pin_spectrum((qidx(0, X), qidx(0, P)), grid, sigma_xp, part=PART_REAL)

    return GalleryProblem(
        target,
        sigma_in_signal=_squeezed_input(r),
        notes="B.8  squeeze angle sweeps 0 -> pi/2 across the band at fixed "
              "depth; crossover Omega_c <- detuning, rotation rate <- kappa~. "
              "Passive and lossless, so the output stays pure at every Omega.",
        suggested={"optimize_gauge": False, "free_decay_ratios": True,
                   "intrinsic_losses": False},
    )



# --------------------------------------------------------------------------
# B.8-imp / B.8-nm / B.8-wind -- the squeeze-angle-rotation family (Sec. 8.7)
#
# Three extensions of B.8, each isolating a different way the rotation gets
# hard: the ellipse stops being pure (B.8-imp), the axis stops being monotone
# in |Omega| (B.8-nm), or the accumulated sweep exceeds what one mode can pack
# into the band (B.8-wind).
#
# B.8 could be written from the closed form theta(Omega) of a single detuned
# mode.  None of the three has such a closed form -- an impure or two-pole
# rotation is a ratio of quadratics whose angle and purity spectra are exactly
# the symbolic-regression *outputs* of Sec. 7.2, never inputs.  They are
# therefore written the only honest way round: as the spectrum of a declared
# *reference device*, evaluated through the forward map, with the prescribed
# quantity (purity spectrum / turning point / total winding) *measured* off it
# and asserted.  The fit never sees the reference, only the pinned numbers.
# --------------------------------------------------------------------------

def rotation_device(detunings=(0.5,), coupling=0.0, decay_ratios=None,
                    losses=None, r=0.6, squeezed_port=0):
    """Reference rotation device: passive modes reflecting an ``r``-squeezed input.

    Returns ``(H, gamma, kappa_tilde, sigma_in)`` ready for
    :func:`autogaussian.forward.output_covariance_quadrature`.

    ``detunings`` are ``Delta_i/kappa_i`` (so ``g_ii = -Delta_i/kappa_i``),
    ``coupling`` the beam-splitter element ``g_01`` of a two-mode device,
    ``losses`` the intrinsic loss ratios ``gamma_i``.  No squeezing element is
    used anywhere: all the squeezing is in the declared input, exactly as in
    B.8 -- what the device does is *rotate* (and, with ``losses``, dilute) it.
    """
    detunings = np.atleast_1d(np.asarray(detunings, dtype=float))
    num_modes = detunings.size
    g = np.diag(-detunings).astype(complex)
    if num_modes > 1 and coupling:
        g[0, 1] = complex(coupling)
        g[1, 0] = np.conj(complex(coupling))
    H = build_H_bdg(jnp.asarray(g), jnp.zeros((num_modes, num_modes), dtype=complex))

    ratios = (np.ones(num_modes) if decay_ratios is None
              else np.broadcast_to(np.asarray(decay_ratios, dtype=float), (num_modes,)))
    gammas = (np.zeros(num_modes) if losses is None
              else np.broadcast_to(np.asarray(losses, dtype=float), (num_modes,)))

    n, m = squeezed_bath(r)
    n_vec, m_vec = np.zeros(num_modes), np.zeros(num_modes, dtype=complex)
    n_vec[squeezed_port], m_vec[squeezed_port] = n, m
    sigma_in = channel_covariance(n=jnp.asarray(n_vec), m=jnp.asarray(m_vec),
                                  num_modes=num_modes)
    return H, jnp.asarray(gammas), jnp.asarray(ratios), sigma_in


def rotation_spectrum(omegas, detunings=(0.5,), coupling=0.0, decay_ratios=None,
                      losses=None, r=0.6):
    """``V(Omega)`` of :func:`rotation_device` on the monitored port -- a real
    ``(len(omegas), 2, 2)`` stack in the quadrature basis ``(x, p)``."""
    H, gammas, ratios, sigma_in = rotation_device(
        detunings, coupling=coupling, decay_ratios=decay_ratios, losses=losses, r=r)
    num_modes = H.shape[0] // 2
    thetas = jnp.zeros(num_modes)
    grid = jnp.asarray(np.atleast_1d(np.asarray(omegas, dtype=float)))
    # vmapped over the grid: the parameter scans below evaluate this on ~10^5
    # frequencies, which a Python loop over the forward map makes unbearable
    block = jax.vmap(lambda omega: output_covariance_quadrature(
        H, gammas, ratios, omega, sigma_in, thetas, 1))(grid)
    return np.real(np.asarray(block))


def squeeze_angle(V):
    """Squeeze-ellipse axis ``theta(Omega)`` of a stack of ``2x2`` blocks.

    ``tan 2theta = 2 sigma_xp / (sigma_xx - sigma_pp)``, unwrapped along the
    grid so that an accumulated sweep past ``pi/2`` stays readable (the raw
    ``arctan`` folds it back).
    """
    V = np.real(np.asarray(V))
    angle = 0.5 * np.arctan2(2.0 * V[:, 0, 1], V[:, 0, 0] - V[:, 1, 1])
    return 0.5 * np.unwrap(2.0 * angle)


def rotation_purity(V):
    """``mu(Omega) = sqrt(det V(Omega)) >= 1`` -- 1 for a pure ellipse, larger
    the more added noise the rotation has cost."""
    return np.sqrt(np.linalg.det(np.real(np.asarray(V))))


def accumulated_rotation(theta):
    """Total swept angle ``sum |Delta theta|`` -- the *winding* of B.8-wind.

    Total variation, not ``|theta(end) - theta(0)|``: a rotation that turns
    around (B.8-nm) still costs the device the angle it went out and back.
    """
    return float(np.sum(np.abs(np.diff(np.asarray(theta, dtype=float)))))


@lru_cache(maxsize=None)
def single_mode_winding_ceiling(omega_max=6.0, num_points=241, r=0.6,
                                detunings=None, decay_ratios=None):
    """Largest winding **one** detuned mode can produce on ``[0, omega_max]``.

    Cached, because the scan is a few thousand forward-map evaluations and the
    tests ask for the same ceiling repeatedly; pass ``detunings`` /
    ``decay_ratios`` as tuples if you override them.

    Found by scanning the whole single-mode family, because there is no
    closed-form budget to quote: the often-repeated ``pi/2`` of B.8 is the
    *useful in-band* rotation of one particular device, not the ceiling.  The
    measured ceiling is ``~pi`` (a single pole turns the sideband phase by
    ``pi``), approached only asymptotically in the band, so the winding that
    actually forces a second mode is band- and parameter-dependent and has to
    be found this way rather than hard-coded.
    """
    grid = _spectral_grid(omega_max, num_points)
    detunings = np.linspace(0.05, 8.0, 40) if detunings is None else np.asarray(detunings)
    decay_ratios = (np.exp(np.linspace(-2.0, 2.0, 21)) if decay_ratios is None
                    else np.asarray(decay_ratios))
    best = 0.0
    for detuning in detunings:
        for ratio in decay_ratios:
            V = rotation_spectrum(grid, [detuning], decay_ratios=[ratio], r=r)
            best = max(best, accumulated_rotation(squeeze_angle(V)))
    return best


def _rotation_target(name, grid, values, note, intrinsic_losses=False, r=0.6):
    target = CovarianceTarget(num_ports=1, name=name)
    target.pin_spectrum((qidx(0, X), qidx(0, X)), grid, values[:, 0, 0], part=PART_REAL)
    target.pin_spectrum((qidx(0, P), qidx(0, P)), grid, values[:, 1, 1], part=PART_REAL)
    target.pin_spectrum((qidx(0, X), qidx(0, P)), grid, values[:, 0, 1], part=PART_REAL)
    return GalleryProblem(
        target,
        sigma_in_signal=_squeezed_input(r),
        notes=note,
        suggested={"optimize_gauge": False, "free_decay_ratios": True,
                   "intrinsic_losses": intrinsic_losses},
        kwargs_optimization={"max_violation_success": 1.0e-8},
    )


def impure_rotation(r=0.6, crossover=1.0, loss=0.2, num_points=9, omega_max=None):
    """**B.8-imp** -- the same rotating axis as B.8, on an ellipse that is no
    longer pure: a prescribed purity spectrum ``mu(Omega) = sqrt(det V) >= 1``.

    B.8 lives entirely on the pure-state variety ``det V = 1``; this target
    deliberately pushes the fit **off** it, which couples the axis fit to a
    depth/purity fit.  It is the most faithful "as-measured" member of the
    family, because real filter cavities are lossy.

    **It forces the intrinsic losses live.**  Impurity means *added noise*: a
    lossless, vacuum-fed passive graph cannot leave ``det V = 1`` no matter how
    its detunings and decay ratios are tuned, so a build with
    ``intrinsic_losses=False`` provably cannot fit ``mu(Omega) > 1``.

    The reference device is B.8's mode with an intrinsic-loss channel of
    ``gamma = loss`` opened on it; the purity dips (``mu`` peaks) near the
    crossover, where the sideband dwells longest inside the cavity and so
    absorbs the most vacuum.
    """
    if omega_max is None:
        omega_max = 4.0 * crossover
    grid = _spectral_grid(omega_max, num_points)
    values = rotation_spectrum(grid, [0.5], decay_ratios=[np.sqrt(2.0) * crossover],
                               losses=[loss], r=r)
    purity = rotation_purity(values)
    return _rotation_target(
        "impure squeeze rotation", grid, values,
        "B.8-imp  rotating axis at prescribed purity mu(Omega) in [%.3f, %.3f] "
        "(most mixed near Omega_c = %.3g); det sigma_out > 1 needs a live "
        "intrinsic loss gamma_i, not just detuning."
        % (purity.min(), purity.max(), crossover),
        intrinsic_losses=True, r=r)


def non_monotone_rotation(r=0.6, detunings=(0.5, 0.5), coupling=1.0,
                          decay_ratios=(1.0, 0.8), num_points=17, omega_max=4.0):
    """**B.8-nm** -- a rotation with a *turning point*: ``theta(|Omega|)`` rises,
    turns over and comes back, instead of B.8's monotone sweep.

    **Provably not one mode (verified numerically over the whole single-mode
    family):** one detuned mode gives a strictly monotone ``theta(|Omega|)`` --
    single-pole phase.  A turning point needs the pole-zero interplay of *two
    coupled* modes.

    The trap this target exists to catch is a test that only counts modes: an
    *uncoupled* second mode is invisible to the monitored port (the dynamical
    matrix stays block diagonal), so it reproduces the single-mode angle
    exactly and stays monotone.  It is the hybridisation, not the mode count,
    that buys the turning point -- hence the default ``coupling`` larger than
    the detunings.  (Sec. 8.7 phrases this as "opposite-sign detunings"; in the
    coupled-mode parametrisation used here the sign of the *bare* detunings is
    not what decides it -- a strongly coupled same-sign pair hybridises into
    normal modes that straddle zero.  Measured over a wide random sweep, the
    normal-mode sign rule predicts monotonicity only ~70% of the time, so the
    tests assert what is actually verified: one mode never turns, this coupled
    pair does.)
    """
    grid = _spectral_grid(omega_max, num_points)
    values = rotation_spectrum(grid, detunings, coupling=coupling,
                               decay_ratios=decay_ratios, r=r)
    dense = _spectral_grid(omega_max, 241)
    angle = squeeze_angle(rotation_spectrum(dense, detunings, coupling=coupling,
                                            decay_ratios=decay_ratios, r=r))
    turn = dense[int(np.argmax(angle) if angle[1] > angle[0] else np.argmin(angle))]
    return _rotation_target(
        "non-monotone squeeze rotation", grid, values,
        "B.8-nm  theta(|Omega|) turns over at Omega ~ %.3g (sweep %.3g rad out, "
        "%.3g rad back); one mode is monotone, so this costs the coupled partner."
        % (turn, abs(angle.max() - angle[0]), abs(angle.max() - angle[-1])),
        r=r)


def winding_rotation(r=0.6, detunings=(1.0, 1.5), coupling=0.6,
                     decay_ratios=(0.9, 0.7), num_points=15, omega_max=6.0):
    """**B.8-wind** -- a prescribed *total* rotation across the band, larger
    than one mode can pack into it.

    Winding is a counted resource: demanding more accumulated sweep than the
    minimal graph supplies forces another mode, and the executable form of the
    claim is B.10's feasibility transition (one mode short -> ``L > tol`` on
    every restart; add the mode -> VALID).

    **No per-mode winding budget is claimed.**  The ``pi/2`` of B.8 is that
    device's useful in-band rotation, not a ceiling: a single detuned mode
    sweeps its axis by up to ``~pi`` (measured, asymptotically over an infinite
    band).  The threshold that actually forces the second mode is therefore
    band- and parameter-dependent and is *measured* by
    :func:`single_mode_winding_ceiling`, never hard-coded.
    """
    grid = _spectral_grid(omega_max, num_points)
    values = rotation_spectrum(grid, detunings, coupling=coupling,
                               decay_ratios=decay_ratios, r=r)
    dense = _spectral_grid(omega_max, 241)
    winding = accumulated_rotation(squeeze_angle(rotation_spectrum(
        dense, detunings, coupling=coupling, decay_ratios=decay_ratios, r=r)))
    return _rotation_target(
        "high-winding squeeze rotation", grid, values,
        "B.8-wind  accumulated sweep %.3g rad over |Omega| <= %.3g -- past the "
        "~pi ceiling of a single mode, so the winding itself costs the second "
        "one." % (winding, omega_max),
        r=r)


# --------------------------------------------------------------------------
# B.9 bandpass sideband squeezing
# --------------------------------------------------------------------------

def bandpass_squeezer(depth=0.5, center=1.0, bandwidth=0.4, num_points=11,
                      omega_max=None, pin_conjugate=False):
    """A squeezing dip localised in frequency, vacuum outside the band.

    ``sigma_xx(Omega) = 1 - (1 - s_0)/(1 + ((Omega - Omega_c)/B)^2)``, i.e. the
    honest full-spectrum form of B.4: pinned on a grid instead of through
    derivatives at a point.  Position ``Omega_c`` <- detuning, width ``B`` <-
    decay ratio ``kappa~``, depth ``1 - s_0`` <- squeezing cooperativity.

    ``pin_conjugate=True`` additionally pins ``sigma_pp = 1/sigma_xx`` (the
    "pure, aligned" reading of App. B.9).  That is a much harder target: a
    detuned mode *rotates* the ellipse (B.8), so demanding the axes stay aligned
    with ``x``/``p`` across a whole band fights the mechanism that makes the
    band in the first place.  Off by default; the squeezed quadrature is what
    the specification is about.
    """
    if omega_max is None:
        omega_max = center + 4.0 * bandwidth
    grid = _spectral_grid(omega_max, num_points)
    values = lorentzian_dip(grid, depth=depth, center=center, bandwidth=bandwidth)

    target = CovarianceTarget(num_ports=1, name="bandpass squeezer")
    target.pin_spectrum((qidx(0, X), qidx(0, X)), grid, values, part=PART_REAL)
    if pin_conjugate:
        target.pin_spectrum((qidx(0, P), qidx(0, P)), grid, 1.0 / values, part=PART_REAL)

    return GalleryProblem(
        target,
        notes="B.9  Lorentzian squeezing window of depth %.3g at Omega_c = %.3g, "
              "half-width %.3g." % (1.0 - depth, center, bandwidth),
        suggested={"optimize_gauge": False, "free_decay_ratios": True},
        kwargs_optimization={"max_violation_success": 1.0e-8},
    )


# --------------------------------------------------------------------------
# B.10 flat-top (maximally flat) broadband squeezing
# --------------------------------------------------------------------------

def flat_top_squeezer(depth=0.5, bandwidth=1.0, order=2, num_points=11,
                      omega_max=None):
    """Order-``n`` Butterworth squeezing plateau -- B.4's local flatness
    promoted to a prescribed roll-off order.

    This is the target for exhibiting the **spectral-degree -> mode-count bound
    at work**: prescribing a sharper band edge raises the McMillan degree of the
    target spectrum, so more modes (each carrying a ``kappa~_i`` knob) are
    needed as ``n`` grows.  Measured with this package: orders 1 and 2 are met
    by a *single* mode; order 3 is not, and needs a second one -- bandwidth
    flatness costs modes.
    """
    if omega_max is None:
        omega_max = 2.5 * bandwidth
    grid = _spectral_grid(omega_max, num_points)
    values = butterworth_dip(grid, depth=depth, bandwidth=bandwidth, order=order)

    target = CovarianceTarget(num_ports=1, name="flat-top squeezer (order %i)" % order)
    target.pin_spectrum((qidx(0, X), qidx(0, X)), grid, values, part=PART_REAL)

    return GalleryProblem(
        target,
        notes="B.10  order-%i maximally flat plateau of depth %.3g, band edge "
              "near |Omega| ~ %.3g." % (order, 1.0 - depth, bandwidth),
        suggested={"optimize_gauge": False, "free_decay_ratios": True},
        kwargs_optimization={"max_violation_success": 1.0e-8},
    )


# --------------------------------------------------------------------------
# B.11 band-limited EPR entanglement
# --------------------------------------------------------------------------

def band_limited_epr(depth=0.7, center=1.0, bandwidth=0.4, num_points=9,
                     omega_max=None, pin_correlations=False, correlation=None):
    """B.2 made spectral: EPR correlations that exist only inside a band.

    The target is written on the *joint* quadratures, because that is what the
    claim "EPR-certified inside the band, separable outside" is about:

        Var((x_1 - x_2)/sqrt2)(Omega) = Var((p_1 + p_2)/sqrt2)(Omega)
            = 1 - (1 - s_0) / (1 + ((Omega - Omega_c)/B)^2)

    (quadratic-form pins, :meth:`CovarianceTarget.pin_form` -- neither joint
    variance is a single entry of ``sigma_out``).  The halved Duan sum is the
    sum of the two, so it is ``2 - 2(1 - s_0) L(Omega)``: below the separability
    bound 2 precisely where the Lorentzian window has weight, and back at 2
    outside it.  The target *is* the entanglement bandwidth.

    ``pin_correlations=True`` writes App. B.11's other reading instead --
    ``sigma_{x1x2}(Omega) = -sigma_{p1p2}(Omega) = c(Omega)`` with the local
    variances pinned to the pure two-mode-squeezed value ``sqrt(1 + c^2)``.
    That form is far more heavily over-determined (six pinned entries per grid
    point instead of two) and in practice is only approached, not met; it is
    kept because it is the literal App. B.11 wording.

    Depth: the default ``s_0 = 0.7`` (about 1.5 dB of joint squeezing) is
    deliberately modest.  This is an *active* target, so the stability
    constraint binds (Sec. 5) -- witnesses come back with ``alpha`` close to
    zero, and asking for a deep band pushes the fit through the
    parametric-oscillation threshold.
    """
    if omega_max is None:
        omega_max = center + 4.0 * bandwidth
    grid = _spectral_grid(omega_max, num_points)
    window = 1.0 / (1.0 + ((grid - center) / bandwidth) ** 2)

    target = CovarianceTarget(num_ports=2, name="band-limited EPR source")
    if pin_correlations:
        if correlation is None:
            correlation = 0.75
        c = correlation * window
        local = np.sqrt(1.0 + c ** 2)                      # pure TMSV: cosh 2r
        for port in (0, 1):
            target.pin_spectrum((qidx(port, X), qidx(port, X)), grid, local, part=PART_REAL)
            target.pin_spectrum((qidx(port, P), qidx(port, P)), grid, local, part=PART_REAL)
        target.pin_spectrum((qidx(0, X), qidx(1, X)), grid, c, part=PART_REAL)
        target.pin_spectrum((qidx(0, P), qidx(1, P)), grid, -c, part=PART_REAL)
        note = ("B.11  correlation form: sigma_{x1x2} = -sigma_{p1p2} = c(Omega), "
                "local variances at the pure value sqrt(1+c^2).")
    else:
        joint = 1.0 - (1.0 - depth) * window
        minus, plus = np.zeros(4), np.zeros(4)
        minus[qidx(0, X)], minus[qidx(1, X)] = 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)
        plus[qidx(0, P)], plus[qidx(1, P)] = 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)
        for omega, value in zip(grid, joint):
            target.pin_form(minus, float(value), omega=float(omega))
            target.pin_form(plus, float(value), omega=float(omega))
        note = ("B.11  joint quadratures squeezed to %.3g inside |Omega - %.3g| ~ %.3g "
                "and at the vacuum floor outside; halved Duan sum = %.3g in band, "
                "2 out of band." % (depth, center, bandwidth, 2.0 * depth))

    return GalleryProblem(
        target,
        notes=note,
        suggested={"optimize_gauge": False, "free_decay_ratios": True},
        kwargs_optimization={"max_violation_success": 1.0e-8},
    )


def duan_of_band_limited_epr(omega, depth=0.7, center=1.0, bandwidth=0.4):
    """The halved Duan sum of the :func:`band_limited_epr` *target* itself
    (separability bound 2), as a function of ``Omega``."""
    omega = np.asarray(omega, dtype=float)
    window = 1.0 / (1.0 + ((omega - center) / bandwidth) ** 2)
    return 2.0 - 2.0 * (1.0 - depth) * window


def epr_bandwidth(depth=0.7, bandwidth=0.4, threshold=2.0 - 1.0e-2):
    """Half-width of the window in which the :func:`band_limited_epr` target is
    entangled by more than ``2 - threshold`` in the halved Duan sum.

    A Lorentzian window has infinite support, so the Duan sum only *approaches*
    2 off band; the certified bandwidth is therefore stated at a threshold.
    """
    margin = 2.0 - float(threshold)
    weight = margin / (2.0 * (1.0 - depth))
    if not 0.0 < weight <= 1.0:
        return 0.0
    return float(bandwidth * np.sqrt(1.0 / weight - 1.0))


# --------------------------------------------------------------------------
# B.12 notch (band-stop) squeezing
# --------------------------------------------------------------------------

def notch_squeezer(depth=0.5, bandwidth=1.5, notch_center=0.8, notch_width=0.25,
                   num_points=13, omega_max=None, band_order=1):
    """Broadband squeezing with a hole punched at a chosen line.

    Used to skip a mechanical resonance or to leave a readout window at the
    vacuum floor:

        sigma_xx(Omega) = 1 - (1 - s_0) * band(Omega) * notch(Omega)

    with ``notch`` the destructive-interference zero of App. B.12 and ``band`` a
    (Butterworth) envelope of half-width ``bandwidth``.  App. B.12 writes the
    notch alone; the envelope is added because a bare notch formula asks for
    squeezing out to ``|Omega| -> inf``, which no device does -- ``sigma_out``
    returns to the input covariance once the sidebands run off resonance.

    The point of the target is the **minimal-mode contrast with B.9**: a dip
    needs one resonance, a dip with a hole in it needs the interfering partner,
    so the notch cannot be fitted on B.9's single-mode graph.
    """
    if omega_max is None:
        omega_max = max(2.0 * bandwidth, notch_center + 4.0 * notch_width)
    grid = _spectral_grid(omega_max, num_points)
    band = 1.0 / (1.0 + (grid / bandwidth) ** (2 * int(band_order)))
    values = 1.0 - (1.0 - depth) * band * notch_factor(
        grid, notch_center=notch_center, notch_width=notch_width)

    target = CovarianceTarget(num_ports=1, name="notch squeezer")
    target.pin_spectrum((qidx(0, X), qidx(0, X)), grid, values, part=PART_REAL)

    return GalleryProblem(
        target,
        notes="B.12  broadband dip of depth %.3g with a hole at Omega_0 = %.3g "
              "(width %.3g); the hole is what costs the extra detuned mode."
              % (1.0 - depth, notch_center, notch_width),
        suggested={"optimize_gauge": False, "free_decay_ratios": True},
        kwargs_optimization={"max_violation_success": 1.0e-8},
    )


GALLERY = {
    "B.1": single_mode_squeezer,
    "B.2": epr_source,
    "B.3": directional_squeezed_source,
    "B.4": broadband_squeezer,
    "B.5": cv_graph_state,
    "B.6": backaction_evading_readout,
    "B.7": noise_diode,
    "B.8": squeeze_angle_rotation,
    "B.9": bandpass_squeezer,
    "B.10": flat_top_squeezer,
    "B.11": band_limited_epr,
    "B.12": notch_squeezer,
    "B.8-imp": impure_rotation,
    "B.8-nm": non_monotone_rotation,
    "B.8-wind": winding_rotation,
}

#: the subset whose pins live on a frequency *grid* -- the targets that make
#: the decay ratios kappa~_i live optimisation variables (Sec. 3, App. A.4(c))
SPECTRAL_GALLERY = {key: GALLERY[key] for key in
                    ("B.8", "B.9", "B.10", "B.11", "B.12",
                     "B.8-imp", "B.8-nm", "B.8-wind")}

#: the squeeze-angle-rotation family of Sec. 8.7 -- B.8 plus the three
#: extensions that make the rotation impure, non-monotone, or over-wound
ROTATION_FAMILY = {key: GALLERY[key] for key in
                   ("B.8", "B.8-imp", "B.8-nm", "B.8-wind")}
