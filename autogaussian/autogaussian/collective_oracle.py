"""
The oracle and architecture optimiser wired to the **collective** forward map
(Addendum Sec. 3, Sec. 4).

Only four methods of :class:`~autogaussian.oracle.CovarianceOracle` touch the
forward map at all -- ``_covariance``, ``_responses``, ``_covariance_derivative``
and ``_dynamical_matrix``.  Everything else (the compiled residual, the hinge,
the restarts, the constrained stability search, the verdict) is generic in ``x``.
So the collective engine is a subclass that swaps those four, and the whole
Sec. 4 / Sec. 5 machine comes along unchanged: fit stays the hard constraint,
stability stays the objective, and the oracle still never certifies.

Fixed channel topology
----------------------
Every slot of the :class:`~autogaussian.hypergraph.ChannelSpace` contributes a
row for *every* graph; a graph switches a channel off by freezing its amplitudes
at zero, not by removing the row.  This keeps ``sigma_out(x)`` a pure function of
``x`` -- jittable and differentiable -- and is physically identical on the
monitored ports: a zero row in ``K_kappa`` is an unmonitored vacuum port and a
zero row in ``K_Gamma`` a zero column of ``N``.  Both are asserted inert in the
D0 and D3 suites.

Ports
-----
``sigma_out`` is indexed by **controlled channel**, and the private controlled
slots are emitted in mode order ahead of every collective one, so port ``j`` is
mode ``j`` exactly as in the base spec.  The extra controlled channels (the
collective ones) are outputs the target does not pin; they are real emission
ports, simply unmonitored.

Scale gauge
-----------
The base spec fixed ``kappa_i = 1`` and measured everything against it.  Here the
collected rate *is* a free amplitude, so the map has a one-parameter scale
redundancy ``(H, K) -> (sH, sK)`` -- a choice of time unit.  It is harmless (the
target only constrains dimensionless ratios) but it does mean the reported
amplitudes are fixed only up to that scale; read cooperativities, not rates.
"""

import numpy as np
import jax
import jax.numpy as jnp

from autogaussian.channel_parametrization import (
    CollectiveParametrization,
    VAR_CHANNEL_ABS,
    VAR_CHANNEL_PHASE,
)
from autogaussian.channels import Access, nambu_adjoint
from autogaussian.graph import COUPLING_WITH_PHASE, COUPLING_WITHOUT_PHASE, GraphSpace, NO_COUPLING
from autogaussian.hypergraph import ChannelSpace, TwoColouredSpace
from autogaussian.nambu import nambu_to_quadrature, quadrature_matrix, vacuum_covariance
from autogaussian.optimizer import CovarianceArchitectureOptimizer
from autogaussian.oracle import CovarianceOracle
from autogaussian.parametrization import Parametrization

__all__ = ["CollectiveOracle", "CollectiveArchitectureOptimizer",
           "collective_cooperativities"]

# An amplitude seeded at ~0 is an undamped device: singular susceptibility and a
# drift that is not Hurwitz.  Start the channels somewhere alive.
CHANNEL_INIT_RANGES = {
    VAR_CHANNEL_ABS: (0.3, 1.4),
    VAR_CHANNEL_PHASE: (-np.pi, np.pi),
}
CHANNEL_PSO_RANGES = {
    VAR_CHANNEL_ABS: (0.0, 3.0),
    VAR_CHANNEL_PHASE: (-np.pi, np.pi),
}


class CollectiveOracle(CovarianceOracle):
    """:class:`CovarianceOracle` with the Addendum Sec. 3 forward map."""

    def __init__(self, parametrization, target, sigma_in_signal=None,
                 sigma_in_noise=None, init_ranges=None, pso_ranges=None, **kwargs):
        self.num_controlled = parametrization.channels.num_controlled
        self.num_uncontrolled = parametrization.channels.num_uncontrolled

        if sigma_in_signal is None:
            sigma_in_signal = vacuum_covariance(self.num_controlled)
        elif callable(sigma_in_signal):
            sigma_in_signal = sigma_in_signal(self.num_controlled)
        if sigma_in_noise is None:
            sigma_in_noise = vacuum_covariance(max(self.num_uncontrolled, 1))
        elif callable(sigma_in_noise):
            sigma_in_noise = sigma_in_noise(self.num_uncontrolled)

        merged_init = dict(CHANNEL_INIT_RANGES)
        merged_init.update(init_ranges or {})
        merged_pso = dict(CHANNEL_PSO_RANGES)
        merged_pso.update(pso_ranges or {})

        super().__init__(parametrization, target,
                         sigma_in_signal=sigma_in_signal,
                         sigma_in_noise=sigma_in_noise,
                         init_ranges=merged_init, pso_ranges=merged_pso, **kwargs)

    # -- forward map -------------------------------------------------------

    def _thetas_full(self, x):
        """One quadrature reference phase per controlled channel."""
        thetas = self.param.coherent.thetas(x)
        pad = self.num_controlled - thetas.shape[0]
        if pad > 0:
            thetas = jnp.concatenate([thetas, jnp.zeros(pad)])
        return thetas[: self.num_controlled]

    def _pieces(self, x, Omega):
        H_bdg = self.param.hamiltonian(x)
        K_kappa, K_Gamma, M_diss = self.param.channels.nambu_pieces(x)
        from autogaussian.nambu import pauli_z

        sz = pauli_z(self.num_modes)
        M = -1j * (sz @ H_bdg) + M_diss
        chi = jnp.linalg.inv(M + 1j * Omega * jnp.eye(2 * self.num_modes,
                                                      dtype=jnp.complex128))
        S = (jnp.eye(2 * self.num_controlled, dtype=jnp.complex128)
             + K_kappa @ chi @ nambu_adjoint(K_kappa, self.num_controlled,
                                             self.num_modes))
        N = K_kappa @ chi @ nambu_adjoint(K_Gamma, self.num_uncontrolled,
                                          self.num_modes)
        return S, N, M

    def _sigma_out(self, x, Omega):
        S, N, _ = self._pieces(x, Omega)
        out = S @ self.sigma_in_signal @ jnp.conj(S).T
        if self.num_uncontrolled:
            out = out + N @ self.sigma_in_noise @ jnp.conj(N).T
        return out

    def _covariance(self, x, Omega):
        sigma_out = self._sigma_out(x, Omega)
        W = quadrature_matrix(self._thetas_full(x), self.num_controlled)
        return nambu_to_quadrature(sigma_out, W)

    def _responses(self, x, Omega):
        S, N, _ = self._pieces(x, Omega)
        return S, N, self._covariance(x, Omega)

    def _dynamical_matrix(self, x):
        _, _, M = self._pieces(x, 0.0)
        return M


class CollectiveArchitectureOptimizer(CovarianceArchitectureOptimizer):
    """Architecture search over the **two-coloured** space (Addendum Sec. 4).

    Same surface as :class:`CovarianceArchitectureOptimizer`, so
    :func:`autogaussian.search.discover` works unmodified; the space is a
    :class:`~autogaussian.hypergraph.TwoColouredSpace` and the oracle is a
    :class:`CollectiveOracle`.

    ``channel_kwargs`` is forwarded to
    :class:`~autogaussian.hypergraph.ChannelSpace` and is how the dissipative
    colour is restricted -- ``max_support``, which tags are enumerated, and
    ``forbidden_supports``.  Note a hardware constraint of the form "these two
    modes may not couple" is a statement about *both* colours: forbidding the
    coherent edge with ``forbidden_couplings`` leaves a shared bath between the
    same two modes perfectly legal, and a shared bath is a coupling.  Pass the
    pair to ``forbidden_supports`` as well when the constraint is meant to be
    physical rather than merely coherent.

    ``asymptotic_bus_modes`` / ``bus_cooperativity`` behave exactly as in the
    base optimiser: they scale the *coherent* couplings that touch a bus mode.
    They do not touch the channel amplitudes -- a bus is a mediator, not a
    detector.
    """

    def __init__(
        self,
        target,
        num_auxiliary_modes=0,
        sigma_in_signal=None,
        sigma_in_noise=None,
        optimize_gauge=True,
        constraints=(),
        graph_space=None,
        channel_space=None,
        stability_margin=1.0e-3,
        stability_weight=1.0,
        asymptotic_bus_modes=(),
        bus_cooperativity=1.0e6,
        kwargs_optimization=None,
        make_initial_test=True,
        seed=None,
        channel_kwargs=None,
        **graph_space_kwargs
    ):
        self.target = target
        self.num_ports = target.num_ports
        self.num_auxiliary_modes = int(num_auxiliary_modes)
        self.num_modes = self.num_ports + self.num_auxiliary_modes

        if graph_space is None:
            graph_space = GraphSpace(self.num_modes, **graph_space_kwargs)
        elif graph_space_kwargs:
            raise ValueError("pass either graph_space or the graph-space keyword arguments")
        if channel_space is None:
            options = dict(max_support=2, collective_controlled=True,
                           collective_uncontrolled=False,
                           collective_bogoliubov=False)
            options.update(channel_kwargs or {})
            channel_space = ChannelSpace(self.num_modes, **options)

        self.space = TwoColouredSpace(graph_space, channel_space)
        self.space.validate()

        # the rate knobs are subsumed by the channel amplitudes (Sec. 4), so the
        # coherent block is used unrescaled: kappa~ and gamma are frozen away
        coherent_param = Parametrization(
            graph_space, self.num_ports,
            intrinsic_losses=False, free_decay_ratios=False,
            optimize_gauge=optimize_gauge, target_symbols=target.free_symbols,
            asymptotic_bus_modes=asymptotic_bus_modes,
            bus_cooperativity=bus_cooperativity,
        )
        self.param = CollectiveParametrization(self.space, coherent_param,
                                               num_ports=self.num_ports)
        self.param.num_modes = self.num_modes
        self.param.num_ports = self.num_ports
        self.param.types = (list(coherent_param.types)
                            + list(self.param.channels.types))

        self.oracle = CollectiveOracle(
            self.param, target,
            sigma_in_signal=sigma_in_signal, sigma_in_noise=sigma_in_noise,
            constraints=constraints, stability_margin=stability_margin,
            stability_weight=stability_weight,
        )

        self.kwargs_optimization = {
            "num_tests": 10,
            "method": "BFGS",
            "max_violation_success": 1.0e-10,
            "interrupt_if_successful": True,
        }
        if kwargs_optimization:
            self.kwargs_optimization.update(kwargs_optimization)

        self.rng = np.random.default_rng(seed)
        self.libraries = None
        self.valid_combinations = []
        self.invalid_combinations = []
        self.tested_complexities = []
        self.num_tested_graphs = []
        self.num_tested_invalid_graphs = []
        self.solutions = {}

        if make_initial_test:
            success, infos = self.test_graph(self.space.fully_connected())
            if not success:
                raise Exception(
                    "fully connected two-coloured graph is invalid, interrupting "
                    "(best loss %.3g)"
                    % min(info["loss_reached"] for info in infos))
            print("fully connected two-coloured graph is a valid graph")

    # -- graph reduction over both colours ---------------------------------

    def reduce_graph(self, x, tolerance=None):
        """Read off which elements the witness actually used, in both colours.

        The coherent half reuses the base-spec rule.  A channel slot is dropped
        when every amplitude of the channel came out below tolerance, and loses
        its phase freedom when every phase came out at zero -- the same
        "value 2 -> value 1 -> value 0" ladder the coherent slots use, which is
        what keeps the reduced graph a subgraph of the tested one.
        """
        if tolerance is None:
            tolerance = self.kwargs_optimization["max_violation_success"]
        x = np.asarray(x)
        coherent_reduced = self._reduce_coherent(x, tolerance)

        channel_reduced = np.zeros(self.space.channels.num_slots, dtype="int8")
        cp = self.param.channels
        for slot_idx in range(self.space.channels.num_slots):
            amplitudes = [abs(float(x[i])) for i in
                          (list(cp.slot_u_abs[slot_idx]) + list(cp.slot_v_abs[slot_idx]))]
            if not amplitudes or max(amplitudes) ** 2 / 2 < tolerance:
                channel_reduced[slot_idx] = NO_COUPLING
                continue
            phases = [float(x[i]) for i in
                      (list(cp.slot_u_phase[slot_idx]) + list(cp.slot_v_phase[slot_idx]))]
            live = [p for p, idx in zip(phases,
                                        list(cp.slot_u_phase[slot_idx])
                                        + list(cp.slot_v_phase[slot_idx]))
                    if idx not in cp.gauge_frozen]
            uses_phase = any(abs(p) ** 2 / 2 >= tolerance for p in live)
            channel_reduced[slot_idx] = (COUPLING_WITH_PHASE if uses_phase
                                         else COUPLING_WITHOUT_PHASE)
            if channel_reduced[slot_idx] not in self.space.channels.possible_values[slot_idx]:
                channel_reduced[slot_idx] = max(
                    v for v in self.space.channels.possible_values[slot_idx])
        return self.space.join(coherent_reduced, channel_reduced)

    def _reduce_coherent(self, x, tolerance):
        from autogaussian.graph import (
            SLOT_DETUNING,
        )

        g, nu = self.param.coherent.blocks(jnp.asarray(x))
        g = np.asarray(g)
        nu = np.asarray(nu)
        reduced = np.zeros(self.space.coherent.num_slots, dtype="int8")
        for slot_idx, (kind, i, j) in enumerate(self.space.coherent.slots):
            if kind == SLOT_DETUNING:
                element = -np.real(g[i, i])
                value = (NO_COUPLING if element ** 2 / 2 < tolerance
                         else COUPLING_WITHOUT_PHASE)
            else:
                element = g[i, j] if kind == "beamsplitter" else nu[i, j]
                if abs(element) ** 2 / 2 < tolerance:
                    value = NO_COUPLING
                elif abs(np.imag(element)) ** 2 / 2 < tolerance:
                    value = COUPLING_WITHOUT_PHASE
                else:
                    value = COUPLING_WITH_PHASE
            allowed = self.space.coherent.possible_values[slot_idx]
            while value not in allowed and value > 0:
                value -= 1
            reduced[slot_idx] = value
        return reduced


def collective_cooperativities(optimizer, graph, x, tolerance=1e-9):
    """Cooperativities of a collective-engine witness.

    **Do not use** :meth:`autogaussian.parametrization.Parametrization.physical_report`
    for this.  That method reports ``C_ij = 4 |H_ij|^2``, which is correct only
    in the base spec's normalisation where every ``kappa_i`` is pinned to 1.
    Here the collected rate is a free channel amplitude, so the cooperativity is

        C_ij = 4 |H_ij|^2 / (kappa_i kappa_j),      kappa_i = |u_i|^2

    and the two disagree by orders of magnitude whenever the fit moves a rate off
    1.  On the B.2' bus architecture ``physical_report`` prints 0.42 and 2.44
    where the true cooperativities are 7.0e3 and 2.4e5.

    Two things this has to get right, both learned the hard way:

    * **Honour the graph.**  A stored witness is a full parameter vector, and the
      variables its graph freezes may still hold optimiser dust.  Reading the
      blocks straight off ``x`` then reports elements the architecture does not
      contain -- e.g. a ``C_{0,0}`` of 1e-12 on a graph with no on-site squeezing
      at all.  The frozen variables are zeroed first.
    * **Threshold the cooperativity, not the amplitude.**  ``C`` goes as
      ``|H|^2``, so a cut on ``|H|`` admits values of order ``tolerance^2``.

    Returns ``{"kappa": {...}, "cooperativities": {...}, "linewidth_ratios": {...}}``.
    """
    x = np.asarray(x, dtype=float).copy()
    frozen = sorted(optimizer.param.frozen_indices(graph))
    if frozen:
        x[np.asarray(frozen, dtype=int)] = 0.0

    channels = optimizer.param.channel_set(graph, x)
    num_modes = optimizer.num_modes

    kappa = np.zeros(num_modes)
    for channel in channels:
        if channel.access is not Access.CONTROLLED or not channel.is_private:
            continue
        mode = channel.support[0]
        kappa[mode] += abs(channel.u[mode]) ** 2

    g, nu = [np.asarray(block) for block in optimizer.param.coherent.blocks(
        jnp.asarray(x))]

    def ratio(value, i, j):
        denominator = kappa[i] * kappa[j]
        return float("inf") if denominator <= 0 else 4 * abs(value) ** 2 / denominator

    cooperativities = {}

    def record(name, value, i, j):
        if abs(value) <= 0.0:
            return
        cooperativity = ratio(value, i, j)
        if cooperativity > tolerance or not np.isfinite(cooperativity):
            cooperativities[name] = cooperativity

    for i in range(num_modes):
        record("C_{%i,%i}" % (i, i), nu[i, i], i, i)
        for j in range(i + 1, num_modes):
            record("C^{BS}_{%i,%i}" % (i, j), g[i, j], i, j)
            record("C^{SQZ}_{%i,%i}" % (i, j), nu[i, j], i, j)

    reference = max(kappa) if max(kappa) > 0 else 1.0
    return {
        "kappa": {"kappa_%i" % i: float(kappa[i]) for i in range(num_modes)},
        "cooperativities": cooperativities,
        "linewidth_ratios": {"kappa_%i/kappa_max" % i: float(kappa[i] / reference)
                             for i in range(num_modes)},
    }
