"""
THE DISSIPATIVE COLOUR: a searchable channel hypergraph (Addendum Sec. 4, Sec. 7).

The coherent graph of :mod:`autogaussian.graph` is unchanged.  Beside it now sits
a second colour whose elements are **hyperedges**: each candidate dissipative
channel is a slot carrying a mode subset (its support) plus a ``(class, access)``
tag.  A private channel is the degenerate support-1 hyperedge, so "private vs
collective" is *discovered* by the search rather than assumed.

Slot values keep AUTOSCATTER's semantics exactly, which is what preserves the
poset the breadth-first search runs on::

    0 = channel off        (all amplitudes frozen at 0)
    1 = real amplitudes    (phases frozen at 0)
    2 = complex amplitudes (phases free)

so ``g1 <= g2`` elementwise still means "``g1`` is a subgraph of ``g2``", across
both colours at once.  :class:`TwoColouredSpace` concatenates the coherent slot
vector and the channel slot vector into one integer vector, and exposes the same
surface :func:`autogaussian.search.discover` already consumes -- so the search
engine needs no changes to run two-coloured.

Monotonicity (Addendum Sec. 7)
------------------------------
A dissipative hyperedge switches off *continuously* as its amplitudes go to
zero, and it takes its bundled bath noise with it: a zero row in ``K_Gamma``
contributes a zero column to ``N``, and a zero row in ``K_kappa`` contributes an
unmonitored vacuum port.  Neither touches ``sigma_out`` on the monitored ports,
so a subgraph witness is reproduced exactly by the supergraph.  VALID stays
upward-closed and certified-INVALID downward-closed.

**This holds only for constraints closed under hyperedge deletion.**  Caps on
support size, on which tags are allowed, on which modes may host a channel --
all fine, all enforced here.  *Connectivity-type* constraints ("at least one
collective channel") are **not** closed under deletion and would break pruning
soundness; :meth:`ChannelSpace.validate_deletion_closure` checks that the space
actually admits the empty channel set, which is the operative test.

Canonical form (Addendum Sec. 5)
--------------------------------
The space never emits a ``BOGOLIUBOV CONTROLLED`` slot.  The guard lives in
:class:`~autogaussian.channels.Channel` as well, but enforcing it here too means
the degenerate specs are never enumerated in the first place.
"""

from itertools import combinations

import numpy as np

from autogaussian.channels import Access, Channel, ChannelClass, ChannelSet
from autogaussian.graph import COUPLING_WITH_PHASE, COUPLING_WITHOUT_PHASE, NO_COUPLING

__all__ = [
    "ChannelSlot",
    "ChannelSpace",
    "TwoColouredSpace",
    "collective_usage_report",
    "spectral_fingerprint",
]

SLOT_CHANNEL = "channel"


class ChannelSlot:
    """One candidate channel: a mode subset plus a ``(class, access)`` tag."""

    __slots__ = ("support", "channel_class", "access")

    def __init__(self, support, channel_class, access):
        support = tuple(int(i) for i in sorted(support))
        channel_class = ChannelClass(channel_class)
        access = Access(access)
        if not support:
            raise ValueError("a channel slot needs at least one mode in its support")
        if (access is Access.CONTROLLED
                and channel_class is ChannelClass.BOGOLIUBOV):
            raise ValueError(
                "BOGOLIUBOV CONTROLLED is forbidden by the Sec. 5 canonical form")
        self.support = support
        self.channel_class = channel_class
        self.access = access

    @property
    def is_private(self):
        return len(self.support) == 1

    @property
    def is_collective(self):
        return len(self.support) >= 2

    @property
    def has_free_phase(self):
        """Whether ``COUPLING_WITH_PHASE`` is a *distinct* device for this slot.

        ``L_mu -> e^{i theta} L_mu`` leaves the dissipator invariant, so the
        overall phase of every channel is gauge and
        :class:`~autogaussian.channel_parametrization.ChannelParametrization`
        freezes ``arg(u_{i_0})`` permanently.  A support-1 particle-conserving
        channel has *only* that phase, so its "real" and "complex" slot values
        describe the same physical device -- offering both would put duplicate
        graphs in the library and inflate the completeness statement with
        phantom false negatives.  A phase is genuinely free only when the
        channel spans more than one mode (relative phases across the support,
        which is what makes the D1 directional cell work) or carries anomalous
        amplitudes (the squeeze angle of ``v``).
        """
        return (len(self.support) > 1
                or self.channel_class is ChannelClass.BOGOLIUBOV)

    @property
    def num_amplitudes(self):
        """Free amplitude count: ``|support|`` for PC, ``2|support|`` for Bogoliubov."""
        k = len(self.support)
        return k if self.channel_class is ChannelClass.PARTICLE_CONSERVING else 2 * k

    def key(self):
        return (self.support, self.channel_class.value, self.access.value)

    def __eq__(self, other):
        return isinstance(other, ChannelSlot) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        return "ChannelSlot(%s, %s, %s)" % (list(self.support),
                                            self.channel_class.value,
                                            self.access.value)

    def describe(self, value=COUPLING_WITH_PHASE):
        kind = "private" if self.is_private else "collective"
        phase = "complex" if value == COUPLING_WITH_PHASE else "real"
        return "%s %s %s channel on %s (%s)" % (
            kind, self.access.value, self.channel_class.value,
            list(self.support), phase)


class ChannelSpace:
    """Enumerates the candidate channels of an ``N``-mode device.

    Parameters
    ----------
    num_modes : int
    max_support : int
        Largest collective support to enumerate (1 = the base-spec private
        scaffold only, which is the D0 corner).
    private_controlled, private_uncontrolled : bool
        Emit the support-1 slots.  The controlled ones are the output ports and
        carry the old ``kappa~_i``; the uncontrolled ones carry the old
        ``gamma_i``.
    collective_controlled, collective_uncontrolled : bool
        Emit particle-conserving collective slots of each access.
    collective_bogoliubov : bool
        Emit ``BOGOLIUBOV UNCONTROLLED`` collective slots (the only place
        anomalous jumps are allowed, per Sec. 5).
    private_bogoliubov : bool
        Emit ``BOGOLIUBOV UNCONTROLLED`` support-1 slots.
    forbidden_supports : iterable of iterable of int
        Mode subsets that may not host a channel.  Closed under deletion, hence
        safe for pruning.

    Notes
    -----
    Every constraint here removes slots outright; none of them can force a slot
    to be *present*.  That is what keeps the space closed under hyperedge
    deletion -- see the module docstring.
    """

    def __init__(
        self,
        num_modes,
        max_support=2,
        private_controlled=True,
        private_uncontrolled=True,
        private_bogoliubov=False,
        collective_controlled=True,
        collective_uncontrolled=False,
        collective_bogoliubov=False,
        allow_phases=True,
        forbidden_supports=(),
    ):
        self.num_modes = int(num_modes)
        self.max_support = max(1, min(int(max_support), self.num_modes))
        self.allow_phases = bool(allow_phases)
        self.forbidden_supports = set(frozenset(int(i) for i in s)
                                      for s in forbidden_supports)

        slots = []
        for i in range(self.num_modes):
            if private_controlled:
                slots.append(ChannelSlot((i,), ChannelClass.PARTICLE_CONSERVING,
                                         Access.CONTROLLED))
            if private_uncontrolled:
                slots.append(ChannelSlot((i,), ChannelClass.PARTICLE_CONSERVING,
                                         Access.UNCONTROLLED))
            if private_bogoliubov:
                slots.append(ChannelSlot((i,), ChannelClass.BOGOLIUBOV,
                                         Access.UNCONTROLLED))
        for size in range(2, self.max_support + 1):
            for support in combinations(range(self.num_modes), size):
                if frozenset(support) in self.forbidden_supports:
                    continue
                if collective_controlled:
                    slots.append(ChannelSlot(support, ChannelClass.PARTICLE_CONSERVING,
                                             Access.CONTROLLED))
                if collective_uncontrolled:
                    slots.append(ChannelSlot(support, ChannelClass.PARTICLE_CONSERVING,
                                             Access.UNCONTROLLED))
                if collective_bogoliubov:
                    slots.append(ChannelSlot(support, ChannelClass.BOGOLIUBOV,
                                             Access.UNCONTROLLED))

        self.channel_slots = [s for s in slots
                              if frozenset(s.support) not in self.forbidden_supports]
        self.slots = [(SLOT_CHANNEL, s, None) for s in self.channel_slots]
        self.possible_values = []
        for slot in self.channel_slots:
            allowed = [NO_COUPLING, COUPLING_WITHOUT_PHASE]
            if self.allow_phases and slot.has_free_phase:
                allowed.append(COUPLING_WITH_PHASE)
            self.possible_values.append(allowed)

        self.num_slots = len(self.channel_slots)
        self.max_values = np.array([max(v) for v in self.possible_values]
                                   if self.possible_values else [], dtype="int8")
        self.max_complexity = int(self.max_values.sum()) if self.num_slots else 0

    # -- views -------------------------------------------------------------

    def __len__(self):
        return self.num_slots

    def private_slot_indices(self):
        return [k for k, s in enumerate(self.channel_slots) if s.is_private]

    def collective_slot_indices(self):
        return [k for k, s in enumerate(self.channel_slots) if s.is_collective]

    def controlled_slot_indices(self):
        return [k for k, s in enumerate(self.channel_slots)
                if s.access is Access.CONTROLLED]

    def index_of(self, support, channel_class, access):
        probe = ChannelSlot(support, channel_class, access)
        for k, slot in enumerate(self.channel_slots):
            if slot == probe:
                return k
        raise KeyError("no such channel slot: %r" % (probe,))

    def describe(self, values):
        out = []
        for value, slot in zip(np.asarray(values), self.channel_slots):
            if value != NO_COUPLING:
                out.append(slot.describe(value))
        return out

    # -- soundness guards ---------------------------------------------------

    def validate_deletion_closure(self):
        """The Sec. 7 caveat, made executable.

        Pruning soundness needs the admissible set to be closed under hyperedge
        deletion.  Every constraint this class accepts only *removes* slots, so
        the empty channel vector is always admissible -- which is the operative
        statement.  A future connectivity-type constraint ("at least one
        collective channel") would break this, and would break pruning with it.
        """
        empty = np.zeros(self.num_slots, dtype="int8")
        for value, allowed in zip(empty, self.possible_values):
            if value not in allowed:
                raise ValueError(
                    "the channel space is not closed under hyperedge deletion: "
                    "the empty channel set is inadmissible.  A constraint of "
                    "connectivity type would break the pruning soundness of "
                    "Sec. 7; only deletion-closed constraints are allowed.")
        return True

    def no_collected_bogoliubov(self):
        """Sec. 5, enforced at the level of the *space*: the degenerate specs
        are never enumerated, not merely rejected later."""
        return all(not (s.access is Access.CONTROLLED
                        and s.channel_class is ChannelClass.BOGOLIUBOV)
                   for s in self.channel_slots)


class TwoColouredSpace:
    """Coherent graph + dissipative hypergraph as one slot vector.

    The vector is ``concat(coherent slots, channel slots)``; monotonicity is
    elementwise on the whole thing, so the search poset is unchanged and
    :func:`autogaussian.search.discover` runs on this object unmodified.
    """

    def __init__(self, coherent, channels):
        self.coherent = coherent
        self.channels = channels
        self.num_modes = coherent.num_modes
        self.num_coherent_slots = coherent.num_slots
        self.num_channel_slots = channels.num_slots
        self.num_slots = self.num_coherent_slots + self.num_channel_slots
        self.possible_values = list(coherent.possible_values) + list(channels.possible_values)
        self.slots = list(coherent.slots) + list(channels.slots)
        self.max_values = np.array([max(v) for v in self.possible_values], dtype="int8")
        self.max_complexity = int(self.max_values.sum())

    # -- splitting ---------------------------------------------------------

    def split(self, graph):
        graph = np.asarray(graph, dtype="int8")
        return graph[: self.num_coherent_slots], graph[self.num_coherent_slots:]

    def join(self, coherent_part, channel_part):
        return np.concatenate([
            np.asarray(coherent_part, dtype="int8"),
            np.asarray(channel_part, dtype="int8"),
        ]).astype("int8")

    # -- the surface search.discover consumes ------------------------------

    def fully_connected(self):
        return self.max_values.copy()

    def empty(self):
        return np.zeros(self.num_slots, dtype="int8")

    def num_possible_graphs(self):
        n = 1
        for values in self.possible_values:
            n *= len(values)
        return n

    def iter_graphs_at_complexity(self, level):
        """Same suffix-sum enumeration as :class:`~autogaussian.graph.GraphSpace`,
        run over the concatenated slot list."""
        values = self.possible_values
        n = self.num_slots
        suffix_max = np.zeros(n + 1, dtype=int)
        for idx in range(n - 1, -1, -1):
            suffix_max[idx] = suffix_max[idx + 1] + max(values[idx])
        current = np.zeros(n, dtype="int8")

        def rec(idx, remaining):
            if idx == n:
                if remaining == 0:
                    yield current.copy()
                return
            for v in values[idx]:
                if v <= remaining and remaining - v <= suffix_max[idx + 1]:
                    current[idx] = v
                    for out in rec(idx + 1, remaining - v):
                        yield out
            current[idx] = 0

        for out in rec(0, int(level)):
            yield out

    def describe(self, graph):
        coherent_part, channel_part = self.split(graph)
        return (self.coherent.describe(coherent_part)
                + self.channels.describe(channel_part))

    # -- guards ------------------------------------------------------------

    def validate(self):
        self.channels.validate_deletion_closure()
        if not self.channels.no_collected_bogoliubov():
            raise ValueError("the channel space emits a BOGOLIUBOV CONTROLLED slot")
        return True


# ---------------------------------------------------------------------------
# Sec. 4 -- "did a collective channel actually earn its place?"
# ---------------------------------------------------------------------------

def collective_usage_report(space, graph, channel_set, tolerance=1e-8):
    """Report, per active collective slot, whether the fit actually used it.

    The addendum's claim is that "private vs collective" is *discovered*: a
    shared channel the fit drives to support-1, or to zero amplitude, is the
    engine reporting that the local scaffold sufficed.  This turns that into a
    readable verdict.

    Returns a dict with ``"private_sufficed"`` (True when no collective channel
    survived the fit with support >= 2) and a per-slot breakdown with verdicts
    ``"off"``, ``"collapsed_to_private"`` or ``"collective"``.
    """
    coherent_part, channel_part = space.split(graph)
    channel_space = space.channels

    # Provenance, not support containment: a private channel's support is a
    # *subset* of a collective slot's, so matching by containment would report a
    # live private channel as the collective slot having collapsed.
    origin = getattr(channel_set, "slot_origin", None)
    if origin is None:
        raise ValueError(
            "channel_set carries no slot provenance; build it with "
            "ChannelParametrization.channel_set so each row records the space "
            "slot it came from")
    realised_by_slot = {}
    for channel, slot_idx in zip(channel_set, origin):
        amplitude = float(np.max(np.abs(np.concatenate([channel.u, channel.v]))))
        realised_by_slot[slot_idx] = (channel.support if amplitude > tolerance
                                      else ())

    entries = []
    for slot_idx, (value, slot) in enumerate(zip(channel_part,
                                                 channel_space.channel_slots)):
        if not slot.is_collective:
            continue
        if value == NO_COUPLING:
            entries.append({"slot": repr(slot), "verdict": "off",
                            "realised_support": ()})
            continue
        realised = realised_by_slot.get(slot_idx, ())
        if len(realised) == 0:
            verdict = "off"
        elif len(realised) == 1:
            verdict = "collapsed_to_private"
        else:
            verdict = "collective"
        entries.append({"slot": repr(slot), "verdict": verdict,
                        "realised_support": realised})

    used = [e for e in entries if e["verdict"] == "collective"]
    return {
        "private_sufficed": len(used) == 0,
        "num_collective_used": len(used),
        "channels": entries,
    }


# ---------------------------------------------------------------------------
# Sec. 8, D4 -- dedup over the emitted spectrum
# ---------------------------------------------------------------------------

def spectral_fingerprint(H_bdg, channels, thetas, omegas, num_ports,
                         sigma_sig=None, sigma_noise=None, decimals=9):
    """A hashable digest of ``sigma_out(Omega)`` over the grid.

    Two library entries that agree on this produce the *same* emitted field and
    are therefore the same device as far as the target is concerned -- which is
    what "minimal setup" has to be well-defined against once the Sec. 5
    canonical form has removed the collected-Bogoliubov degeneracy.
    """
    import numpy as _np
    from autogaussian.forward import collective_output_covariance_quadrature
    from autogaussian.nambu import vacuum_covariance

    if sigma_sig is None:
        sigma_sig = vacuum_covariance(channels.num_controlled)
    rows = []
    for omega in _np.atleast_1d(_np.asarray(omegas, dtype=float)):
        V = _np.asarray(collective_output_covariance_quadrature(
            H_bdg, channels, float(omega), sigma_sig, thetas,
            num_ports=num_ports, sigma_noise=sigma_noise))
        rows.append(_np.round(_np.real(V), decimals).ravel())
        rows.append(_np.round(_np.imag(V), decimals).ravel())
    return tuple(_np.concatenate(rows).tolist())
