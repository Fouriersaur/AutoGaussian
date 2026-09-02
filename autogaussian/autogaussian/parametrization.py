"""
Continuous parametrisation (Sec. 2 / App. A.4 of the algorithm flow).

The learnable dimensionless parameters are exactly

    { C_ij , Phi , Delta_i/kappa_i , gamma_i , kappa~_i }  (device)
    +  sigma_in  (declared)   mod  theta_j  (gauge)

stored here as one flat real vector ``x``.  Each coupling contributes a
magnitude and a phase (so ``C_ij = 4 |H_ij|^2`` and the loop phases ``Phi``
follow from them); decay-rate ratios are stored as ``log kappa~_i`` so the
optimiser stays unconstrained and ``kappa~_i > 0`` is automatic.

Just like AUTOSCATTER, *every* variable exists for *every* graph; a graph
switches elements off by freezing the corresponding variables at 0 (all "off"
values are 0: ``|g| = 0``, ``phase = 0``, ``Delta = 0``, ``gamma = 0``,
``log kappa~ = 0`` i.e. ``kappa~ = 1``, ``theta = 0``).  This way the whole
search compiles a single jitted forward map.
"""

import numpy as np
import jax.numpy as jnp

from autogaussian.graph import (
    NO_COUPLING,
    COUPLING_WITH_PHASE,
    SLOT_BEAMSPLITTER,
    SLOT_DETUNING,
    SLOT_ONSITE_SQUEEZING,
    SLOT_TWO_MODE_SQUEEZING,
)

VAR_ABS = "abs_variable"
VAR_PHASE = "phase_variable"
VAR_LOSS = "intrinsic_loss_variable"
VAR_LOG_KAPPA = "log_decay_ratio_variable"
VAR_USER = "user_defined"

__all__ = ["Parametrization", "VAR_ABS", "VAR_PHASE", "VAR_LOSS", "VAR_LOG_KAPPA", "VAR_USER"]


class Parametrization:
    """Variable bookkeeping + the ``x -> (H, gamma, kappa~, theta)`` map.

    Parameters
    ----------
    space : :class:`autogaussian.graph.GraphSpace`
    num_ports : int
    intrinsic_losses : bool or sequence of bool
        Which modes carry a free intrinsic loss ``gamma_i`` (App. A.4(b)).
    free_decay_ratios : bool or sequence of bool
        Which modes carry a free decay-rate ratio ``kappa~_i``.  These are
        *inert at Omega = 0* and only matter for a spectral target -- the
        headline difference to AUTOSCATTER (App. A.4(c)).
    optimize_gauge : bool
        Optimise the per-port quadrature phases ``theta_j``, or fix them at 0
        when the target names a quadrature (Sec. 1.5).
    target_symbols : list of sympy.Symbol
        Free symbols appearing in the target values.
    asymptotic_bus_modes : sequence of int
        Modes treated as ideal mediators: all their couplings are scaled by
        ``sqrt(bus_cooperativity)`` so that ``C_bus -> infinity`` at fixed
        ratios (Sec. 7.3).  Error of the finite surrogate ~ ``1/C_bus``.
    bus_cooperativity : float
    """

    def __init__(
        self,
        space,
        num_ports,
        intrinsic_losses=False,
        free_decay_ratios=True,
        optimize_gauge=True,
        target_symbols=(),
        asymptotic_bus_modes=(),
        bus_cooperativity=1.0e6,
        log_decay_ratio_bound=np.log(100.0),
    ):
        self.space = space
        self.log_decay_ratio_bound = (None if log_decay_ratio_bound is None
                                      else float(log_decay_ratio_bound))
        self.num_modes = space.num_modes
        self.num_ports = int(num_ports)
        self.target_symbols = list(target_symbols)
        self.asymptotic_bus_modes = set(int(b) for b in asymptotic_bus_modes)
        self.bus_cooperativity = float(bus_cooperativity)

        self.intrinsic_losses = self._expand_flags(intrinsic_losses)
        self.free_decay_ratios = self._expand_flags(free_decay_ratios)
        self.optimize_gauge = bool(optimize_gauge)

        self.names = []
        self.types = []
        self._index = {}

        # -- slot variables ------------------------------------------------
        # slot_variables[k] = list of variable indices belonging to slot k
        self.slot_variables = []
        self.slot_phase_variable = []
        for kind, i, j in space.slots:
            if kind == SLOT_DETUNING:
                idx = self._new("Delta_%i" % i, VAR_ABS)
                self.slot_variables.append([idx])
                self.slot_phase_variable.append(None)
            elif kind == SLOT_BEAMSPLITTER:
                a = self._new("|g_{%i,%i}|" % (i, j), VAR_ABS)
                p = self._new("arg(g_{%i,%i})" % (i, j), VAR_PHASE)
                self.slot_variables.append([a, p])
                self.slot_phase_variable.append(p)
            elif kind == SLOT_ONSITE_SQUEEZING:
                a = self._new("|nu_{%i,%i}|" % (i, i), VAR_ABS)
                p = self._new("arg(nu_{%i,%i})" % (i, i), VAR_PHASE)
                self.slot_variables.append([a, p])
                self.slot_phase_variable.append(p)
            else:
                a = self._new("|nu_{%i,%i}|" % (i, j), VAR_ABS)
                p = self._new("arg(nu_{%i,%i})" % (i, j), VAR_PHASE)
                self.slot_variables.append([a, p])
                self.slot_phase_variable.append(p)

        # -- per-mode variables --------------------------------------------
        self.gamma_idx = np.array(
            [self._new("gamma_%i" % i, VAR_LOSS) for i in range(self.num_modes)]
        )
        self.log_kappa_idx = np.array(
            [self._new("log_kappa_%i" % i, VAR_LOG_KAPPA) for i in range(self.num_modes)]
        )
        self.theta_idx = np.array(
            [self._new("theta_%i" % i, VAR_PHASE) for i in range(self.num_ports)]
        )

        # -- free target symbols -------------------------------------------
        self.target_symbol_idx = np.array(
            [self._new(str(s), VAR_USER) for s in self.target_symbols], dtype=int
        )

        self.num_variables = len(self.names)
        self.types = list(self.types)

        # variables permanently frozen by the configuration (not by the graph)
        frozen = []
        for i in range(self.num_modes):
            if not self.intrinsic_losses[i]:
                frozen.append(self.gamma_idx[i])
            if not self.free_decay_ratios[i]:
                frozen.append(self.log_kappa_idx[i])
        if not self.optimize_gauge:
            frozen.extend(list(self.theta_idx))
        self.config_frozen = set(int(k) for k in frozen)

        self._build_index_arrays()

    # ------------------------------------------------------------------ #

    def _expand_flags(self, flags):
        if flags is True:
            return np.ones(self.num_modes, dtype=bool)
        if flags is False:
            return np.zeros(self.num_modes, dtype=bool)
        flags = np.asarray(flags, dtype=bool)
        if flags.size > self.num_modes:
            if flags[self.num_modes:].any():
                raise ValueError(
                    "flag set for mode %i, but the device only has %i modes"
                    % (int(np.argmax(flags[self.num_modes:])) + self.num_modes,
                       self.num_modes))
            flags = flags[: self.num_modes]
        out = np.zeros(self.num_modes, dtype=bool)
        out[: flags.size] = flags
        return out

    def _new(self, name, vartype):
        idx = len(self.names)
        self.names.append(name)
        self.types.append(vartype)
        self._index[name] = idx
        return idx

    def index(self, name):
        return self._index[name]

    def _build_index_arrays(self):
        """Static index arrays for the jitted ``x -> H`` map."""
        space = self.space

        def bus_scale(i, j):
            """Asymptotic building block (Sec. 7.3).

            A mediating bus mode ``b`` is pushed to ``C_bus -> infinity`` at
            fixed ratios by scaling its couplings with ``sqrt(Lambda)`` per bus
            endpoint -- so a self term (detuning / on-site squeezing) scales
            with ``Lambda`` and a link to a signal mode with ``sqrt(Lambda)``.
            That keeps ``C_ajb / sqrt(C_bb)`` finite, virtually occupies the
            bus, and leaves an error ``~ 1/C_bus``.
            """
            endpoints = int(i in self.asymptotic_bus_modes) + int(j in self.asymptotic_bus_modes)
            return self.bus_cooperativity ** (0.5 * endpoints)

        det_rows, det_vars, det_scale = [], [], []
        bs_rows, bs_cols, bs_abs, bs_phase, bs_scale = [], [], [], [], []
        nu_rows, nu_cols, nu_abs, nu_phase, nu_scale = [], [], [], [], []

        for slot_idx, (kind, i, j) in enumerate(space.slots):
            vars_ = self.slot_variables[slot_idx]
            scale = bus_scale(i, j)
            if kind == SLOT_DETUNING:
                det_rows.append(i)
                det_vars.append(vars_[0])
                det_scale.append(scale)
            elif kind == SLOT_BEAMSPLITTER:
                bs_rows.append(i)
                bs_cols.append(j)
                bs_abs.append(vars_[0])
                bs_phase.append(vars_[1])
                bs_scale.append(scale)
            else:  # on-site or two-mode squeezing -> anomalous block
                nu_rows.append(i)
                nu_cols.append(j)
                nu_abs.append(vars_[0])
                nu_phase.append(vars_[1])
                nu_scale.append(scale)

        self._det_rows = jnp.array(det_rows, dtype=int)
        self._det_vars = jnp.array(det_vars, dtype=int)
        self._det_scale = jnp.array(det_scale)
        self._bs_rows = jnp.array(bs_rows, dtype=int)
        self._bs_cols = jnp.array(bs_cols, dtype=int)
        self._bs_abs = jnp.array(bs_abs, dtype=int)
        self._bs_phase = jnp.array(bs_phase, dtype=int)
        self._bs_scale = jnp.array(bs_scale)
        self._nu_rows = jnp.array(nu_rows, dtype=int)
        self._nu_cols = jnp.array(nu_cols, dtype=int)
        self._nu_abs = jnp.array(nu_abs, dtype=int)
        self._nu_phase = jnp.array(nu_phase, dtype=int)
        self._nu_scale = jnp.array(nu_scale)

        self._gamma_idx = jnp.array(self.gamma_idx, dtype=int)
        self._log_kappa_idx = jnp.array(self.log_kappa_idx, dtype=int)
        self._theta_idx = jnp.array(self.theta_idx, dtype=int)

    # ------------------------------------------------------------------ #
    # x -> physical quantities
    # ------------------------------------------------------------------ #

    def blocks(self, x):
        """Return the normal (``g``, Hermitian) and anomalous (``nu``,
        symmetric) blocks of the dimensionless BdG Hamiltonian."""
        n = self.num_modes
        g = jnp.zeros((n, n), dtype=jnp.complex128)
        nu = jnp.zeros((n, n), dtype=jnp.complex128)

        # detunings:  Delta_i / kappa_i = -H_ii   (Sec. 0)
        if self._det_rows.shape[0] > 0:
            g = g.at[self._det_rows, self._det_rows].add(
                -(self._det_scale * x[self._det_vars]).astype(jnp.complex128)
            )
        # beam-splitter couplings (Hermitian)
        if self._bs_rows.shape[0] > 0:
            vals = (self._bs_scale * x[self._bs_abs]) * jnp.exp(1j * x[self._bs_phase])
            g = g.at[self._bs_rows, self._bs_cols].add(vals)
            g = g.at[self._bs_cols, self._bs_rows].add(jnp.conj(vals))
        # squeezing couplings (symmetric; the diagonal is on-site squeezing)
        if self._nu_rows.shape[0] > 0:
            vals = (self._nu_scale * x[self._nu_abs]) * jnp.exp(1j * x[self._nu_phase])
            nu = nu.at[self._nu_rows, self._nu_cols].add(vals)
            off_diagonal = (self._nu_rows != self._nu_cols).astype(vals.dtype)
            nu = nu.at[self._nu_cols, self._nu_rows].add(vals * off_diagonal)
        return g, nu

    def hamiltonian(self, x):
        from autogaussian.nambu import build_H_bdg

        g, nu = self.blocks(x)
        return build_H_bdg(g, nu)

    def gamma(self, x):
        return jnp.abs(x[self._gamma_idx])

    def kappa_tilde(self, x):
        """Decay-rate ratios ``kappa~_i = exp(log kappa~_i)``.

        The exponent is squashed smoothly into ``[-bound, bound]`` so the
        optimiser cannot run the ratios off to 0 or infinity.  That is not
        cosmetic: an unbounded ``kappa~`` lets a derivative-only broadband
        target (App. B.4) be "solved" by shrinking the linewidth until the
        curvature at ``Omega = 0`` vanishes, instead of widening the band.
        ``tanh`` keeps ``x = 0 -> kappa~ = 1`` and is the identity for small
        ``x``, so nothing else changes.
        """
        exponent = x[self._log_kappa_idx]
        if self.log_decay_ratio_bound is not None:
            bound = self.log_decay_ratio_bound
            exponent = bound * jnp.tanh(exponent / bound)
        return jnp.exp(exponent)

    def thetas(self, x):
        return x[self._theta_idx]

    def unpack(self, x):
        """``x -> (H, gamma, kappa~, theta)``."""
        return self.hamiltonian(x), self.gamma(x), self.kappa_tilde(x), self.thetas(x)

    # ------------------------------------------------------------------ #
    # graph <-> frozen variables
    # ------------------------------------------------------------------ #

    def frozen_indices(self, graph):
        """Variable indices held at 0 for this graph (Sec. 6 conditions)."""
        frozen = set(self.config_frozen)
        for slot_idx, value in enumerate(np.asarray(graph)):
            vars_ = self.slot_variables[slot_idx]
            if value == NO_COUPLING:
                frozen.update(vars_)
            elif value != COUPLING_WITH_PHASE:
                phase = self.slot_phase_variable[slot_idx]
                if phase is not None:
                    frozen.add(phase)
        return frozen

    def free_indices(self, graph):
        frozen = self.frozen_indices(graph)
        return np.array([i for i in range(self.num_variables) if i not in frozen], dtype=int)

    def embed(self, x_free, free_idxs):
        x = np.zeros(self.num_variables)
        x[free_idxs] = x_free
        return x

    # ------------------------------------------------------------------ #
    # readout
    # ------------------------------------------------------------------ #

    def solution_dict(self, x):
        return {name: float(value) for name, value in zip(self.names, np.asarray(x))}

    def physical_report(self, x):
        """Cooperativities, fluxes, detunings, losses and decay ratios of a
        solution -- the human-readable form of Sec. 8's output."""
        x = np.asarray(x)
        g, nu = self.blocks(jnp.asarray(x))
        g = np.asarray(g)
        nu = np.asarray(nu)
        report = {"cooperativities": {}, "phases": {}, "detunings": {},
                  "intrinsic_losses": {}, "decay_ratios": {}, "gauge_phases": {}}
        n = self.num_modes
        for i in range(n):
            if abs(g[i, i]) > 1e-12:
                report["detunings"]["Delta_%i/kappa_%i" % (i, i)] = float(-np.real(g[i, i]))
            if abs(nu[i, i]) > 1e-12:
                report["cooperativities"]["C_{%i,%i}" % (i, i)] = float(4 * abs(nu[i, i]) ** 2)
                report["phases"]["arg(nu_{%i,%i})" % (i, i)] = float(np.angle(nu[i, i]))
            for j in range(i + 1, n):
                if abs(g[i, j]) > 1e-12:
                    report["cooperativities"]["C^{BS}_{%i,%i}" % (i, j)] = float(
                        4 * abs(g[i, j]) ** 2)
                    report["phases"]["arg(g_{%i,%i})" % (i, j)] = float(np.angle(g[i, j]))
                if abs(nu[i, j]) > 1e-12:
                    report["cooperativities"]["C^{SQZ}_{%i,%i}" % (i, j)] = float(
                        4 * abs(nu[i, j]) ** 2)
                    report["phases"]["arg(nu_{%i,%i})" % (i, j)] = float(np.angle(nu[i, j]))
        kappa_tilde = np.asarray(self.kappa_tilde(jnp.asarray(x)))
        for i in range(n):
            if self.intrinsic_losses[i]:
                report["intrinsic_losses"]["gamma_%i" % i] = float(abs(x[self.gamma_idx[i]]))
            report["decay_ratios"]["kappa~_%i" % i] = float(kappa_tilde[i])
        for j in range(self.num_ports):
            report["gauge_phases"]["theta_%i" % j] = float(x[self.theta_idx[j]])
        return report
