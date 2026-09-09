"""
Continuous parameters of the dissipative colour (Addendum Sec. 4).

The new optimisation variables are the jump amplitudes and phases
``{u_{mu,i}, v_{mu,i}}``.  They **subsume** the base spec's rate knobs rather
than sitting beside them:

    a support-1 CONTROLLED   amplitude *is* sqrt(kappa~_i)   (the old live kappa~)
    a support-1 UNCONTROLLED amplitude *is* sqrt(Gamma_i)    (the old gamma_i)

so ``kappa~`` stays live exactly as the base spec Sec. 3 requires -- it is now a
channel amplitude instead of a separate scalar.  Because the rates live in the
channels, the coherent block is used **unrescaled**: the companion
:class:`~autogaussian.parametrization.Parametrization` is built with
``free_decay_ratios=False`` and ``intrinsic_losses=False``, which freezes the
knobs the channels have taken over.

Layout, per channel slot with support ``(i_0, ..., i_{k-1})``::

    |u_{i_0}| ... |u_{i_{k-1}}|        amplitudes  (always)
    arg(u_{i_1}) ... arg(u_{i_{k-1}})  phases      (the first is gauge, see below)
    |v_{i_0}| ... arg(v_{i_{k-1}})     anomalous   (BOGOLIUBOV slots only)

Slot values follow AUTOSCATTER's semantics so the search poset is untouched:
``0`` freezes every variable of the channel at zero (the channel is *off*),
``1`` frees the amplitudes and freezes the phases, ``2`` frees both.

Gauge: ``L_mu -> e^{i theta} L_mu`` leaves the dissipator invariant, so the
overall phase of each channel is unphysical; ``arg(u_{i_0})`` is permanently
frozen at 0.  This is a per-channel constraint, hence closed under hyperedge
deletion and safe for the Sec. 7 pruning argument.
"""

import numpy as np
import jax.numpy as jnp

from autogaussian.channels import Access, Channel, ChannelClass, ChannelSet
from autogaussian.graph import COUPLING_WITH_PHASE, NO_COUPLING
from autogaussian.parametrization import VAR_ABS, VAR_PHASE

__all__ = ["ChannelParametrization", "CollectiveParametrization",
           "VAR_CHANNEL_ABS", "VAR_CHANNEL_PHASE"]

# Channel amplitudes get their own variable types purely so the oracle can give
# them a sensible starting box: an amplitude seeded near zero is a device with
# no damping, whose susceptibility is singular and whose drift is not Hurwitz.
VAR_CHANNEL_ABS = "channel_amplitude_variable"
VAR_CHANNEL_PHASE = "channel_phase_variable"


class ChannelParametrization:
    """Variable bookkeeping for a :class:`~autogaussian.hypergraph.ChannelSpace`."""

    def __init__(self, channel_space, offset=0):
        self.space = channel_space
        self.num_modes = channel_space.num_modes
        self.offset = int(offset)

        self.names = []
        self.types = []
        # per slot: dicts of variable indices
        self.slot_u_abs = []
        self.slot_u_phase = []
        self.slot_v_abs = []
        self.slot_v_phase = []
        self.gauge_frozen = set()

        for slot in channel_space.channel_slots:
            tag = "%s%s" % ("".join(str(i) for i in slot.support),
                            "K" if slot.access is Access.CONTROLLED else "G")
            u_abs, u_phase = [], []
            for position, mode in enumerate(slot.support):
                u_abs.append(self._new("|u_{%s,%i}|" % (tag, mode), VAR_CHANNEL_ABS))
                phase = self._new("arg(u_{%s,%i})" % (tag, mode), VAR_CHANNEL_PHASE)
                u_phase.append(phase)
                if position == 0:
                    # overall channel phase is gauge -- freeze it permanently
                    self.gauge_frozen.add(phase)
            v_abs, v_phase = [], []
            if slot.channel_class is ChannelClass.BOGOLIUBOV:
                for mode in slot.support:
                    v_abs.append(self._new("|v_{%s,%i}|" % (tag, mode), VAR_CHANNEL_ABS))
                    v_phase.append(self._new("arg(v_{%s,%i})" % (tag, mode), VAR_CHANNEL_PHASE))
            self.slot_u_abs.append(u_abs)
            self.slot_u_phase.append(u_phase)
            self.slot_v_abs.append(v_abs)
            self.slot_v_phase.append(v_phase)

        self.num_variables = len(self.names)
        self._build_index_arrays()

    # -- jax-traceable assembly (fixed channel topology) -------------------

    def _build_index_arrays(self):
        """Static index arrays for a jitted ``x -> (U, V)`` map.

        The topology is **fixed**: every slot of the space contributes a row,
        and a graph switches a channel off by freezing its amplitudes at zero
        rather than by removing the row.  That keeps the forward map a pure
        function of ``x`` -- jit-friendly, and differentiable -- while staying
        physically identical on the monitored ports (a zero row in ``K_kappa``
        is an unmonitored vacuum port, a zero row in ``K_Gamma`` a zero column
        of ``N``; both verified inert in the D0/D3 tests).
        """
        self.controlled_slots = [k for k, s in enumerate(self.space.channel_slots)
                                 if s.access is Access.CONTROLLED]
        self.uncontrolled_slots = [k for k, s in enumerate(self.space.channel_slots)
                                   if s.access is Access.UNCONTROLLED]
        self.num_controlled = len(self.controlled_slots)
        self.num_uncontrolled = len(self.uncontrolled_slots)

        def gather(slot_indices):
            u_rows, u_cols, u_abs, u_phase = [], [], [], []
            v_rows, v_cols, v_abs, v_phase = [], [], [], []
            for row, slot_idx in enumerate(slot_indices):
                slot = self.space.channel_slots[slot_idx]
                for position, mode in enumerate(slot.support):
                    u_rows.append(row)
                    u_cols.append(mode)
                    u_abs.append(self.slot_u_abs[slot_idx][position])
                    u_phase.append(self.slot_u_phase[slot_idx][position])
                for position, mode in enumerate(
                        slot.support if self.slot_v_abs[slot_idx] else ()):
                    v_rows.append(row)
                    v_cols.append(mode)
                    v_abs.append(self.slot_v_abs[slot_idx][position])
                    v_phase.append(self.slot_v_phase[slot_idx][position])
            to = lambda a: jnp.array(a, dtype=int)
            return (to(u_rows), to(u_cols), to(u_abs), to(u_phase),
                    to(v_rows), to(v_cols), to(v_abs), to(v_phase))

        self._ctrl_index = gather(self.controlled_slots)
        self._unctrl_index = gather(self.uncontrolled_slots)

    def _rows_from(self, x, index, num_rows):
        u_rows, u_cols, u_abs, u_phase, v_rows, v_cols, v_abs, v_phase = index
        shape = (num_rows, self.num_modes)
        U = jnp.zeros(shape, dtype=jnp.complex128)
        V = jnp.zeros(shape, dtype=jnp.complex128)
        if u_rows.shape[0]:
            U = U.at[u_rows, u_cols].set(
                jnp.abs(x[u_abs]) * jnp.exp(1j * x[u_phase]))
        if v_rows.shape[0]:
            V = V.at[v_rows, v_cols].set(
                jnp.abs(x[v_abs]) * jnp.exp(1j * x[v_phase]))
        return U, V

    def coupling_rows(self, x):
        """``(U_kappa, V_kappa, U_Gamma, V_Gamma)`` as jax arrays."""
        x = jnp.asarray(x)
        Uk, Vk = self._rows_from(x, self._ctrl_index, self.num_controlled)
        Ug, Vg = self._rows_from(x, self._unctrl_index, self.num_uncontrolled)
        return Uk, Vk, Ug, Vg

    def nambu_pieces(self, x):
        """``(K_kappa, K_Gamma, M_diss)`` -- everything the drift and the
        response need, assembled in one traceable pass."""
        from autogaussian.channels import dissipative_drift, nambu_coupling

        Uk, Vk, Ug, Vg = self.coupling_rows(x)
        K_kappa = nambu_coupling(Uk, Vk)
        K_Gamma = nambu_coupling(Ug, Vg)
        M_diss = dissipative_drift(jnp.concatenate([Uk, Ug], axis=0),
                                   jnp.concatenate([Vk, Vg], axis=0))
        return K_kappa, K_Gamma, M_diss

    def _new(self, name, vartype):
        idx = self.offset + len(self.names)
        self.names.append(name)
        self.types.append(vartype)
        return idx

    # -- graph -> frozen variables ----------------------------------------

    def slot_variables(self, slot_idx):
        return (list(self.slot_u_abs[slot_idx]) + list(self.slot_u_phase[slot_idx])
                + list(self.slot_v_abs[slot_idx]) + list(self.slot_v_phase[slot_idx]))

    def frozen_indices(self, channel_values):
        """Variables held at 0 for this channel vector."""
        frozen = set(self.gauge_frozen)
        for slot_idx, value in enumerate(np.asarray(channel_values)):
            if value == NO_COUPLING:
                frozen.update(self.slot_variables(slot_idx))
            elif value != COUPLING_WITH_PHASE:
                frozen.update(self.slot_u_phase[slot_idx])
                frozen.update(self.slot_v_phase[slot_idx])
        return frozen

    # -- x -> ChannelSet ---------------------------------------------------

    def channel_set(self, channel_values, x, drop_inactive=True):
        """Build the :class:`~autogaussian.channels.ChannelSet` for one graph.

        Slots at ``NO_COUPLING`` are dropped entirely when ``drop_inactive``;
        keeping them (as zero-amplitude rows) is physically equivalent on the
        monitored ports and is what the Sec. 7 monotonicity test exercises.

        Controlled channels come first, in slot order, so channel index ==
        output-port index; with only the private controlled slots active and no
        collective ones, that reproduces the base spec's mode ordering.
        """
        x = np.asarray(x)
        values = np.asarray(channel_values)
        controlled, uncontrolled = [], []
        for slot_idx, (value, slot) in enumerate(zip(values, self.space.channel_slots)):
            if value == NO_COUPLING and drop_inactive:
                continue
            u = np.zeros(self.num_modes, dtype=complex)
            v = np.zeros(self.num_modes, dtype=complex)
            if value != NO_COUPLING:
                for position, mode in enumerate(slot.support):
                    magnitude = abs(float(x[self.slot_u_abs[slot_idx][position]]))
                    phase = float(x[self.slot_u_phase[slot_idx][position]])
                    u[mode] = magnitude * np.exp(1j * phase)
                if slot.channel_class is ChannelClass.BOGOLIUBOV:
                    for position, mode in enumerate(slot.support):
                        magnitude = abs(float(x[self.slot_v_abs[slot_idx][position]]))
                        phase = float(x[self.slot_v_phase[slot_idx][position]])
                        v[mode] = magnitude * np.exp(1j * phase)
            channel = Channel(u=u, v=v, access=slot.access)
            (controlled if slot.access is Access.CONTROLLED
             else uncontrolled).append((channel, slot_idx))
        ordered = controlled + uncontrolled
        channel_set = ChannelSet([c for c, _ in ordered], num_modes=self.num_modes)
        # provenance: which space slot each emitted row came from.  Reports must
        # key off this rather than off support containment -- a private channel
        # sits *inside* a collective slot's support and would otherwise be
        # mistaken for that slot having collapsed to support-1.
        channel_set.slot_origin = [k for _, k in ordered]
        return channel_set


class CollectiveParametrization:
    """The two-coloured parameter vector: coherent block + channel hypergraph.

    ``unpack(x)`` returns ``(H_BdG, ChannelSet, thetas)`` ready for
    :func:`autogaussian.forward.collective_output_covariance_quadrature`.  The
    Hamiltonian is **unrescaled** (see the module docstring): all rates live in
    the channels.
    """

    def __init__(self, space, coherent_param, num_ports=None):
        self.space = space
        self.coherent = coherent_param
        self.channels = ChannelParametrization(space.channels,
                                               offset=coherent_param.num_variables)
        self.num_modes = space.num_modes
        self.num_ports = (space.channels.num_modes if num_ports is None
                          else int(num_ports))
        self.num_variables = coherent_param.num_variables + self.channels.num_variables
        self.names = list(coherent_param.names) + list(self.channels.names)

    def __getattr__(self, name):
        """Delegate anything coherent-only to the coherent parametrisation.

        The oracle reads a handful of attributes off ``param`` that belong to
        the coherent block alone (``target_symbols``, the symbol index arrays,
        ``blocks``, ...).  ``__getattr__`` fires only when normal lookup fails,
        so the collective overrides above are never shadowed.
        """
        if name.startswith("__") or name == "coherent":
            raise AttributeError(name)
        try:
            return getattr(object.__getattribute__(self, "coherent"), name)
        except AttributeError:
            raise AttributeError(
                "%r has no attribute %r (and neither does its coherent block)"
                % (type(self).__name__, name))

    # -- x -> physics ------------------------------------------------------

    def hamiltonian(self, x):
        """Unrescaled ``H_BdG``.

        The coherent parametrisation already emits ``H`` in units of
        ``kappa_ref`` with ``kappa~ = 1`` frozen, so no rescaling is applied.
        """
        return self.coherent.hamiltonian(jnp.asarray(x))

    def channel_set(self, graph, x, drop_inactive=True):
        _, channel_values = self.space.split(graph)
        return self.channels.channel_set(channel_values, x,
                                         drop_inactive=drop_inactive)

    def thetas(self, x, num_controlled):
        """Per-output-channel gauge phases, padded/truncated to the number of
        controlled channels the graph actually switched on."""
        base = np.asarray(self.coherent.thetas(jnp.asarray(x)))
        out = np.zeros(int(num_controlled))
        take = min(out.size, base.size)
        out[:take] = base[:take]
        return jnp.asarray(out)

    def unpack(self, graph, x, drop_inactive=True):
        channels = self.channel_set(graph, x, drop_inactive=drop_inactive)
        return (self.hamiltonian(x), channels,
                self.thetas(x, channels.num_controlled))

    # -- graph <-> frozen variables ---------------------------------------

    def frozen_indices(self, graph):
        coherent_values, channel_values = self.space.split(graph)
        frozen = set(self.coherent.frozen_indices(coherent_values))
        frozen |= self.channels.frozen_indices(channel_values)
        return frozen

    def free_indices(self, graph):
        frozen = self.frozen_indices(graph)
        return np.array([i for i in range(self.num_variables) if i not in frozen],
                        dtype=int)

    def embed(self, x_free, free_idxs):
        x = np.zeros(self.num_variables)
        x[np.asarray(free_idxs, dtype=int)] = x_free
        return x
