"""D3 -- two-coloured search and hyperedge monotonicity (Addendum Sec. 8, D3).

The dissipative colour is a hypergraph whose slots follow AUTOSCATTER's value
semantics, so the concatenated vector keeps the same poset and
:func:`autogaussian.search.discover` runs on it **unmodified**.  What has to be
checked is that nothing about the poset argument breaks when the new colour is a
hypergraph rather than a graph:

  * the space enumerates coherent edges *and* channel partitions, and slot
    dominance is still elementwise;
  * the Sec. 5 canonical form is enforced at the level of the space, so
    ``BOGOLIUBOV CONTROLLED`` specs are never enumerated;
  * the space is closed under hyperedge deletion (the Sec. 7 caveat);
  * switching a hyperedge's amplitude to zero reproduces the subgraph witness
    **exactly** on the monitored ports -- the continuity the monotonicity
    argument rests on;
  * a shared channel the fit drives to support-1 or to zero is reported as
    "private sufficed";
  * base-spec search invariants 1-4 survive, and only ``certified=True``
    invalids prune.

Certificates are stubbed inconclusive here, exactly as in base-spec milestone M4.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp

from autogaussian.channel_parametrization import CollectiveParametrization
from autogaussian.channels import Access, ChannelClass
from autogaussian.forward import collective_output_covariance_quadrature
from autogaussian.graph import GraphSpace, any_subgraph_of, is_subgraph_of_any
from autogaussian.hypergraph import (
    ChannelSlot,
    ChannelSpace,
    TwoColouredSpace,
    collective_usage_report,
)
from autogaussian.nambu import vacuum_covariance
from autogaussian.parametrization import Parametrization
from autogaussian.search import discover, one_step_reductions
from autogaussian.types import Verdict, graph_key


def build_space(num_modes=2, **channel_kwargs):
    coherent = GraphSpace(num_modes, allow_onsite_squeezing=True)
    options = dict(max_support=2, collective_controlled=True,
                   collective_bogoliubov=False)
    options.update(channel_kwargs)
    channels = ChannelSpace(num_modes, **options)
    space = TwoColouredSpace(coherent, channels)
    space.validate()
    return space


def build_small_space(num_modes=2):
    """A deliberately small two-coloured space for the enumeration and search
    tests: no detunings, no on-site squeezing, and channel amplitudes without
    phases.  288 graphs instead of ~10^5, so the invariants can be checked
    exhaustively rather than sampled."""
    coherent = GraphSpace(num_modes, allow_detunings=False,
                          allow_onsite_squeezing=False)
    channels = ChannelSpace(num_modes, max_support=2, collective_controlled=True,
                            allow_phases=False)
    space = TwoColouredSpace(coherent, channels)
    space.validate()
    return space


def build_param(space):
    coherent = Parametrization(space.coherent, num_ports=space.num_modes,
                               intrinsic_losses=False, free_decay_ratios=False)
    return CollectiveParametrization(space, coherent)


# ---------------------------------------------------------------------------
# the space: two colours, one poset
# ---------------------------------------------------------------------------

def test_space_is_two_coloured_and_splits_cleanly():
    space = build_space()
    graph = space.fully_connected()
    coherent_part, channel_part = space.split(graph)
    assert coherent_part.size == space.coherent.num_slots
    assert channel_part.size == space.channels.num_slots
    assert np.array_equal(space.join(coherent_part, channel_part), graph)
    assert space.num_slots == coherent_part.size + channel_part.size


def test_channel_slots_cover_private_and_collective():
    space = build_space(num_modes=3, max_support=2)
    slots = space.channels.channel_slots
    private = [s for s in slots if s.is_private]
    collective = [s for s in slots if s.is_collective]
    assert len(private) == 3 * 2                     # controlled + uncontrolled per mode
    assert len(collective) == 3                      # the three mode pairs, controlled
    assert all(s.channel_class is ChannelClass.PARTICLE_CONSERVING for s in collective)


def test_enumeration_covers_both_colours():
    """Complexity levels mix coherent edges and channel partitions."""
    space = build_small_space()
    seen_coherent, seen_channel = False, False
    total = 0
    for level in range(space.max_complexity + 1):
        for graph in space.iter_graphs_at_complexity(level):
            total += 1
            assert int(np.sum(graph)) == level
            coherent_part, channel_part = space.split(graph)
            seen_coherent |= bool(np.any(coherent_part))
            seen_channel |= bool(np.any(channel_part))
    assert seen_coherent and seen_channel
    assert total == space.num_possible_graphs()


def test_subgraph_dominance_is_still_elementwise_across_both_colours():
    space = build_space()
    full = space.fully_connected()
    for child in one_step_reductions(full, space):
        assert is_subgraph_of_any(child, np.atleast_2d(full))
        assert any_subgraph_of(full, np.atleast_2d(child))
        assert int(np.sum(child)) < int(np.sum(full))


def test_describe_reports_both_colours():
    space = build_space()
    text = " ".join(space.describe(space.fully_connected()))
    assert "channel" in text
    assert "squeezing" in text or "beam-splitter" in text


# ---------------------------------------------------------------------------
# Sec. 5 at the level of the space
# ---------------------------------------------------------------------------

def test_collected_bogoliubov_slot_cannot_be_constructed():
    with pytest.raises(ValueError, match="canonical form"):
        ChannelSlot((0, 1), ChannelClass.BOGOLIUBOV, Access.CONTROLLED)


def test_space_never_enumerates_a_collected_bogoliubov_slot():
    space = build_space(num_modes=3, collective_bogoliubov=True,
                        private_bogoliubov=True)
    assert space.channels.no_collected_bogoliubov()
    for slot in space.channels.channel_slots:
        assert not (slot.access is Access.CONTROLLED
                    and slot.channel_class is ChannelClass.BOGOLIUBOV)
    # ...and Bogoliubov slots do exist, so the check is not vacuous
    assert any(s.channel_class is ChannelClass.BOGOLIUBOV
               for s in space.channels.channel_slots)


# ---------------------------------------------------------------------------
# Sec. 7 -- deletion closure and monotonicity
# ---------------------------------------------------------------------------

def test_space_is_closed_under_hyperedge_deletion():
    """The operative form of the Sec. 7 caveat: the empty channel set is always
    admissible, so deleting a hyperedge never leaves the space."""
    space = build_space(num_modes=3, max_support=3, collective_uncontrolled=True,
                        collective_bogoliubov=True)
    assert space.channels.validate_deletion_closure()
    empty = space.empty()
    assert int(np.sum(empty)) == 0
    reachable = list(space.iter_graphs_at_complexity(0))
    assert len(reachable) == 1 and np.array_equal(reachable[0], empty)


def test_every_constraint_only_removes_slots():
    """Constraints are deletion-closed by construction: forbidding a support
    shrinks the slot list, it never forces a slot on."""
    unconstrained = build_space(num_modes=3, max_support=2)
    constrained = build_space(num_modes=3, max_support=2,
                              forbidden_supports=[(0, 1)])
    assert constrained.channels.num_slots < unconstrained.channels.num_slots
    keys = {s.key() for s in unconstrained.channels.channel_slots}
    assert {s.key() for s in constrained.channels.channel_slots} < keys


def test_switching_a_hyperedge_off_reproduces_the_subgraph_exactly():
    """**The Sec. 7 monotonicity check.**

    Take a supergraph with a live collective channel, drive that channel's
    amplitudes to zero, and compare ``sigma_out`` on the monitored ports against
    the subgraph in which the hyperedge slot is simply absent.  They must agree
    to machine precision: the hyperedge removes its coupling *and* its bundled
    bath noise together, so the subgraph witness is reproduced by the
    supergraph and VALID stays upward-closed.
    """
    space = build_space(num_modes=2)
    param = build_param(space)
    rng = np.random.default_rng(4)

    collective_slot = space.channels.index_of(
        (0, 1), ChannelClass.PARTICLE_CONSERVING, Access.CONTROLLED)

    supergraph = space.fully_connected()
    subgraph = supergraph.copy()
    subgraph[space.num_coherent_slots + collective_slot] = 0

    x = rng.normal(scale=0.4, size=param.num_variables)
    x[np.asarray(param.channels.slot_u_abs[collective_slot])] = 0.0   # amplitude -> 0
    # keep the private controlled amplitudes healthy so the device is stable
    for slot_idx, slot in enumerate(space.channels.channel_slots):
        if slot.is_private and slot.access is Access.CONTROLLED:
            x[param.channels.slot_u_abs[slot_idx][0]] = 1.0

    def out(graph):
        H_bdg, channels, thetas = param.unpack(graph, x)
        return np.asarray(collective_output_covariance_quadrature(
            H_bdg, channels, 0.37, vacuum_covariance(channels.num_controlled),
            thetas, num_ports=2))

    assert np.allclose(out(supergraph), out(subgraph), atol=1e-12)


def test_keeping_a_zero_amplitude_row_is_also_inert():
    """The same statement with the dead row retained rather than dropped -- the
    two conventions must agree on the monitored ports."""
    space = build_space(num_modes=2)
    param = build_param(space)
    rng = np.random.default_rng(11)
    graph = space.fully_connected()
    collective_slot = space.channels.index_of(
        (0, 1), ChannelClass.PARTICLE_CONSERVING, Access.CONTROLLED)
    x = rng.normal(scale=0.3, size=param.num_variables)
    x[np.asarray(param.channels.slot_u_abs[collective_slot])] = 0.0
    for slot_idx, slot in enumerate(space.channels.channel_slots):
        if slot.is_private and slot.access is Access.CONTROLLED:
            x[param.channels.slot_u_abs[slot_idx][0]] = 1.0

    off = graph.copy()
    off[space.num_coherent_slots + collective_slot] = 0

    def out(graph_, drop):
        H_bdg, channels, thetas = param.unpack(graph_, x, drop_inactive=drop)
        return np.asarray(collective_output_covariance_quadrature(
            H_bdg, channels, 0.2, vacuum_covariance(channels.num_controlled),
            thetas, num_ports=2))

    assert np.allclose(out(off, True), out(off, False), atol=1e-12)


# ---------------------------------------------------------------------------
# Sec. 4 -- "private sufficed" is a reported verdict, not an assumption
# ---------------------------------------------------------------------------

def test_collective_channel_driven_to_zero_reports_private_sufficed():
    space = build_space(num_modes=2)
    param = build_param(space)
    graph = space.fully_connected()
    collective_slot = space.channels.index_of(
        (0, 1), ChannelClass.PARTICLE_CONSERVING, Access.CONTROLLED)

    x = np.zeros(param.num_variables)
    for slot_idx, slot in enumerate(space.channels.channel_slots):
        if slot.is_private and slot.access is Access.CONTROLLED:
            x[param.channels.slot_u_abs[slot_idx][0]] = 1.0
    channel_set = param.channel_set(graph, x)
    report = collective_usage_report(space, graph, channel_set)
    assert report["private_sufficed"]
    assert report["num_collective_used"] == 0
    assert any(entry["verdict"] == "off" for entry in report["channels"])


def test_collective_channel_collapsing_to_support_one_is_reported():
    space = build_space(num_modes=2)
    param = build_param(space)
    graph = space.fully_connected()
    collective_slot = space.channels.index_of(
        (0, 1), ChannelClass.PARTICLE_CONSERVING, Access.CONTROLLED)
    x = np.zeros(param.num_variables)
    # only the first leg of the shared channel survives the fit
    x[param.channels.slot_u_abs[collective_slot][0]] = 0.9
    x[param.channels.slot_u_abs[collective_slot][1]] = 0.0
    channel_set = param.channel_set(graph, x)
    report = collective_usage_report(space, graph, channel_set)
    assert report["private_sufficed"]
    assert any(entry["verdict"] == "collapsed_to_private"
               for entry in report["channels"])


def test_a_genuinely_collective_fit_is_reported_as_collective():
    space = build_space(num_modes=2)
    param = build_param(space)
    graph = space.fully_connected()
    collective_slot = space.channels.index_of(
        (0, 1), ChannelClass.PARTICLE_CONSERVING, Access.CONTROLLED)
    x = np.zeros(param.num_variables)
    x[np.asarray(param.channels.slot_u_abs[collective_slot])] = 0.7
    channel_set = param.channel_set(graph, x)
    report = collective_usage_report(space, graph, channel_set)
    assert not report["private_sufficed"]
    assert report["num_collective_used"] == 1
    assert any(entry["verdict"] == "collective" for entry in report["channels"])


def test_private_amplitudes_are_the_old_rate_knobs():
    """Sec. 4: a support-1 CONTROLLED amplitude *is* ``sqrt(kappa~_i)``."""
    space = build_space(num_modes=2)
    param = build_param(space)
    graph = space.fully_connected()
    x = np.zeros(param.num_variables)
    controlled = [k for k, s in enumerate(space.channels.channel_slots)
                  if s.is_private and s.access is Access.CONTROLLED]
    uncontrolled = [k for k, s in enumerate(space.channels.channel_slots)
                    if s.is_private and s.access is Access.UNCONTROLLED]
    x[param.channels.slot_u_abs[controlled[0]][0]] = np.sqrt(0.4)
    x[param.channels.slot_u_abs[uncontrolled[1]][0]] = np.sqrt(0.9)
    channels = param.channel_set(graph, x)
    assert np.isclose(abs(channels.controlled[0].u[0]), np.sqrt(0.4))
    assert np.isclose(abs(channels.uncontrolled[1].u[1]), np.sqrt(0.9))


# ---------------------------------------------------------------------------
# search invariants 1-4 on the two-coloured poset (certificates stubbed, M4)
# ---------------------------------------------------------------------------

class StubOptimizer:
    """Minimal surface :func:`discover` consumes, with a deterministic oracle.

    A graph is VALID iff it switches on at least one coherent element **and**
    at least one controlled channel -- a rule that is genuinely upward-closed on
    the two-coloured poset, so the monotonicity the search assumes really holds
    and any invariant violation is the engine's fault, not the stub's.
    """

    def __init__(self, space):
        self.space = space
        self.kwargs_optimization = {"num_tests": 1}
        self.solutions = {}
        self.tested_complexities = []
        self.num_tested_graphs = []
        self.num_tested_invalid_graphs = []
        self.tested = []

    def _is_valid(self, graph):
        coherent_part, channel_part = self.space.split(graph)
        controlled = [k for k, s in enumerate(self.space.channels.channel_slots)
                      if s.access is Access.CONTROLLED]
        return bool(np.any(coherent_part)) and bool(np.any(channel_part[controlled]))

    def test_graph(self, graph, **kwargs):
        self.tested.append(np.asarray(graph, dtype="int8").copy())
        ok = self._is_valid(graph)
        info = {
            "success": ok,
            "loss_reached": 0.0 if ok else 1.0,
            "loss_below_tolerance": ok,
            "max_real_eigenvalue": -1.0 if ok else 0.5,
            "x": np.asarray(graph, dtype=float),
        }
        return ok, [info]

    def reduce_graph(self, x):
        return np.asarray(x, dtype="int8")


def run_stub_search(**kwargs):
    space = build_small_space(num_modes=2)
    optimizer = StubOptimizer(space)
    libraries = discover(optimizer, progress=False, verbose=False,
                         use_certificates=False, perform_graph_reduction=False,
                         **kwargs)
    return space, optimizer, libraries


def test_two_coloured_search_runs_and_finds_both_colours_in_witnesses():
    space, _, libraries = run_stub_search()
    assert libraries.valid
    for key in libraries.valid:
        graph = np.frombuffer(key, dtype="int8")
        coherent_part, channel_part = space.split(graph)
        assert np.any(coherent_part) and np.any(channel_part)


def test_invariant_3_every_tested_graph_lands_in_exactly_one_library():
    _, _, libraries = run_stub_search()
    assert set(libraries.valid) & set(libraries.invalid) == set()


def test_invariant_4_valid_entries_carry_a_witness():
    _, _, libraries = run_stub_search()
    for witness in libraries.valid.values():
        assert witness["loss_below_tolerance"]
        assert witness["max_real_eigenvalue"] < 0.0


def test_invariant_2_uncertified_invalids_never_condemn_subgraphs():
    """The correctness-critical one, now on the two-coloured poset."""
    space, _, libraries = run_stub_search()
    assert libraries.n_certified() == 0
    assert libraries.n_uncertified() == len(libraries.invalid)
    valid_graphs = libraries.valid_graphs()
    for entry in libraries.invalid.values():
        assert not entry.certified
        for child in one_step_reductions(entry.graph, space):
            covered = valid_graphs.size and any_subgraph_of(child, valid_graphs)
            assert (graph_key(child) in libraries.valid
                    or graph_key(child) in libraries.invalid
                    or covered)


def test_minimal_valid_elements_are_incomparable():
    _, _, libraries = run_stub_search()
    minimal = libraries.minimal_valid()
    assert len(minimal) >= 1
    for i, row in enumerate(minimal):
        others = np.delete(minimal, i, axis=0)
        assert not any_subgraph_of(row, others)


def test_completeness_statement_is_emitted():
    space, _, libraries = run_stub_search()
    statement = libraries.completeness_statement(space)
    assert "complete" in statement


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
