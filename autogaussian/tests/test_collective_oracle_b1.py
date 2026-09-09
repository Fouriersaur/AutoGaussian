"""The collective engine run on B.1, the single-mode squeezer.

This is the end-to-end check that the Addendum machinery reproduces the base
spec where it must, and reports honestly where the two genuinely differ.

Three claims are locked in here.

**Reduction at the level of the whole discovery loop.**  With one mode there is
no subset of size >= 2, so the channel hypergraph has only private slots and the
two-coloured search *is* the base-spec search.  It must return the same
irreducible architectures.

**The Sec. 7.2 construction rule survives the scale gauge.**  The base spec fixed
``kappa = 1``; here the collected rate is a free channel amplitude, so the map
gained a one-parameter scale redundancy ``(H, K) -> (sH, sK)``.  The
*dimensionless* cooperativity must still land on
``C_00(v) = ((1-sqrt v)/(1+sqrt v))^2`` even though ``kappa`` comes out far from 1.

**A behavioural change the base spec could not have.**  Because ``kappa_i`` was
hard-wired to 1 on every mode, the base spec could never propose a device with an
undamped mode.  Now that it is a searchable amplitude, a graph may leave an
auxiliary mode untouched by every channel -- a genuine dark mode, correctly
certified by PBH.  Reaching it with *either* a private or a collective channel
repairs it, and the engine reports which one the fit actually used.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogaussian import CovarianceArchitectureOptimizer
from autogaussian.certificates import pbh_dark_mode_collective
from autogaussian.channels import Access, ChannelClass
from autogaussian.collective_oracle import CollectiveArchitectureOptimizer
from autogaussian.gallery import single_mode_squeezer
from autogaussian.hypergraph import collective_usage_report
from autogaussian.search import discover

VARIANCE = 0.5


def build(num_auxiliary_modes=0, num_tests=20):
    problem = single_mode_squeezer(variance=VARIANCE)
    return problem, CollectiveArchitectureOptimizer(
        problem.target, num_auxiliary_modes=num_auxiliary_modes, seed=0,
        make_initial_test=False, kwargs_optimization={"num_tests": num_tests},
        **problem.optimizer_kwargs())


def coherent_signature(space, graphs, coherent_only=True):
    """The coherent half of each graph, as a comparable set."""
    out = set()
    for graph in graphs:
        coherent_part, _ = space.split(graph)
        out.add(tuple(int(v) for v in coherent_part))
    return out


def test_one_mode_space_has_no_collective_slots():
    """With a single mode there is nothing to share: the hypergraph degenerates
    to the base-spec private scaffold, which is why D0's reduction is what
    governs this target."""
    _, optimizer = build(num_auxiliary_modes=0)
    slots = optimizer.space.channels.channel_slots
    assert slots and all(slot.is_private for slot in slots)


def test_b1_finds_the_same_irreducibles_as_the_base_algorithm():
    """Reduction, at the level of the discovery loop rather than the forward map."""
    problem, collective = build(num_auxiliary_modes=0)
    collective_libs = discover(collective, progress=False, verbose=False,
                               use_certificates=False)

    base = CovarianceArchitectureOptimizer(
        problem.target, num_auxiliary_modes=0, seed=0, make_initial_test=False,
        **problem.optimizer_kwargs())
    base_libs = discover(base, progress=False, verbose=False, use_certificates=False)

    base_graphs = {tuple(int(v) for v in g) for g in base_libs.minimal_valid()}
    collective_graphs = coherent_signature(collective.space,
                                           collective_libs.minimal_valid())
    assert base_graphs == collective_graphs
    # the known B.1 answer is in there: on-site squeezing alone
    assert (0, 2) in base_graphs


def test_every_valid_witness_carries_a_live_collected_port():
    """A device with no collected channel emits nothing; such graphs must not
    reach the valid library."""
    _, optimizer = build(num_auxiliary_modes=0)
    libs = discover(optimizer, progress=False, verbose=False, use_certificates=False)
    assert libs.valid
    for key in libs.valid:
        graph = np.frombuffer(key, dtype="int8")
        _, channel_part = optimizer.space.split(graph)
        controlled = [k for k, s in enumerate(optimizer.space.channels.channel_slots)
                      if s.access is Access.CONTROLLED]
        assert np.any(channel_part[controlled])


def test_b1_reproduces_the_construction_rule_under_a_free_kappa():
    """``C_00(v) = ((1-sqrt v)/(1+sqrt v))^2`` on the on-site-squeezing family,
    with ``kappa`` free rather than pinned to 1."""
    _, optimizer = build(num_auxiliary_modes=0)
    libs = discover(optimizer, progress=False, verbose=False, use_certificates=False)

    space = optimizer.space
    onsite = space.coherent.slots.index(("onsite_squeezing", 0, 0))
    target_graph = None
    for graph in libs.minimal_valid():
        coherent_part, _ = space.split(graph)
        # the pure on-site-squeezing family: squeezing on, no detuning
        if coherent_part[onsite] and not coherent_part[
                space.coherent.slots.index(("detuning", 0, 0))]:
            target_graph = graph
            break
    assert target_graph is not None

    solution = optimizer.solution_of(target_graph)
    assert solution is not None
    x = solution["x"]

    covariance = np.real(np.asarray(optimizer.oracle.covariance(x, 0.0)))
    assert np.isclose(covariance[0, 0], VARIANCE, atol=1e-5)
    assert np.isclose(covariance[1, 1], 1.0 / VARIANCE, atol=1e-4)

    nu = np.asarray(optimizer.param.coherent.blocks(x)[1])[0, 0]
    channels = optimizer.param.channel_set(target_graph, x)
    kappa = abs(channels.controlled[0].u[0]) ** 2
    cooperativity = 4 * abs(nu / kappa) ** 2
    analytic = ((1 - np.sqrt(VARIANCE)) / (1 + np.sqrt(VARIANCE))) ** 2
    assert np.isclose(cooperativity, analytic, rtol=1e-4)
    # ...and the scale really was free: kappa did not come back at 1
    assert not np.isclose(kappa, 1.0, atol=0.2)


# ---------------------------------------------------------------------------
# the auxiliary-mode case: a dark mode the base spec could not express
# ---------------------------------------------------------------------------

def _aux_graphs(optimizer):
    space = optimizer.space
    offset = space.num_coherent_slots
    onsite = space.coherent.slots.index(("onsite_squeezing", 0, 0))
    port0 = space.channels.index_of((0,), ChannelClass.PARTICLE_CONSERVING,
                                    Access.CONTROLLED)
    port1 = space.channels.index_of((1,), ChannelClass.PARTICLE_CONSERVING,
                                    Access.CONTROLLED)
    shared = space.channels.index_of((0, 1), ChannelClass.PARTICLE_CONSERVING,
                                     Access.CONTROLLED)

    def make(extra=None, value=2):
        graph = space.empty()
        graph[onsite] = 2
        graph[offset + port0] = 1
        if extra is not None:
            graph[offset + extra] = value
        return graph

    return {"private_only": make(), "collective": make(shared),
            "private_aux": make(port1, value=1)}


def test_idle_auxiliary_mode_is_a_certified_dark_mode():
    """The fit succeeds and the device is still not Hurwitz: mode 1 is touched
    by no channel at all, so its eigenvalue sits exactly at zero."""
    _, optimizer = build(num_auxiliary_modes=1)
    graph = _aux_graphs(optimizer)["private_only"]
    success, infos = optimizer.test_graph(graph)
    best = min(infos, key=lambda info: info["loss_reached"])

    assert not success
    assert best["loss_reached"] < 1e-10          # the *fit* was fine
    assert min(info["max_real_eigenvalue"] for info in infos) >= -1e-9

    H_bdg = optimizer.param.hamiltonian(best["x"])
    channels = optimizer.param.channel_set(graph, best["x"], drop_inactive=False)
    fired, detail = pbh_dark_mode_collective(H_bdg, channels)
    assert fired
    assert detail["rank"] < detail["dimension"]


@pytest.mark.parametrize("case", ["collective", "private_aux"])
def test_reaching_the_dark_mode_repairs_it(case):
    """*Either* a collective or a private channel touching mode 1 works."""
    _, optimizer = build(num_auxiliary_modes=1)
    graph = _aux_graphs(optimizer)[case]
    success, infos = optimizer.test_graph(graph)
    assert success
    witness = next(info for info in infos if info["success"])
    assert witness["max_real_eigenvalue"] < 0.0

    H_bdg = optimizer.param.hamiltonian(witness["x"])
    channels = optimizer.param.channel_set(graph, witness["x"], drop_inactive=False)
    assert not pbh_dark_mode_collective(H_bdg, channels)[0]


def test_private_sufficed_is_reported_correctly_on_b1():
    """B.1 gives the collective channel nothing to do, and the engine says so.

    Repairing the dark mode with a private channel is reported as
    ``private_sufficed``; doing it with the shared channel is reported as a
    genuinely collective solution.  Neither is *required* -- which is the honest
    finding for this target, and exactly the Sec. 4 claim that private-vs-
    collective is discovered rather than assumed.
    """
    _, optimizer = build(num_auxiliary_modes=1)
    graphs = _aux_graphs(optimizer)

    success, infos = optimizer.test_graph(graphs["private_aux"])
    assert success
    x = next(info for info in infos if info["success"])["x"]
    report = collective_usage_report(optimizer.space, graphs["private_aux"],
                                     optimizer.param.channel_set(graphs["private_aux"], x))
    assert report["private_sufficed"]
    assert report["num_collective_used"] == 0

    success, infos = optimizer.test_graph(graphs["collective"])
    assert success
    x = next(info for info in infos if info["success"])["x"]
    report = collective_usage_report(optimizer.space, graphs["collective"],
                                     optimizer.param.channel_set(graphs["collective"], x))
    assert not report["private_sufficed"]
    assert report["num_collective_used"] == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# soundness of the graph-reduction step (regression)
# ---------------------------------------------------------------------------

def test_reduction_that_removes_a_port_is_rejected():
    """A reduction may not switch off a channel the witness was using.

    ``discover`` stores a reduced graph as VALID carrying the parent's witness
    without re-testing it.  That is sound only while the reduction drops
    elements the witness left at zero.  Switching off a *collected* channel is
    never such an element: it deletes the output port, turning that port's
    scattering row into the identity so the detector reads back its own vacuum
    input.  ``reduction_is_sound`` has to catch that.

    Regression for a false VALID observed in the B.2' library, where an entry
    with no collected channel on port 1 was reported irreducible and then failed
    re-testing with loss 1.25 (its ports emitting plain vacuum).
    """
    from autogaussian.search import reduction_is_sound

    _, optimizer = build(num_auxiliary_modes=0)
    space = optimizer.space
    onsite = space.coherent.slots.index(("onsite_squeezing", 0, 0))
    port0 = space.channels.index_of((0,), ChannelClass.PARTICLE_CONSERVING,
                                    Access.CONTROLLED)
    full = space.empty()
    full[onsite] = 2
    full[space.num_coherent_slots + port0] = 1

    success, infos = optimizer.test_graph(full)
    assert success
    witness = next(info for info in infos if info["success"])

    # the graph the witness actually satisfies is sound
    assert reduction_is_sound(optimizer, full, witness)

    # ...but a "reduction" that removes the only collected channel is not
    stripped = full.copy()
    stripped[space.num_coherent_slots + port0] = 0
    assert not reduction_is_sound(optimizer, stripped, witness)


def test_revalidate_flags_a_planted_false_valid():
    """``revalidate_valid_library`` catches an entry that cannot fit."""
    from autogaussian.search import revalidate_valid_library
    from autogaussian.types import Libraries

    _, optimizer = build(num_auxiliary_modes=0)
    space = optimizer.space
    onsite = space.coherent.slots.index(("onsite_squeezing", 0, 0))

    # on-site squeezing but no collected channel at all: emits vacuum, so it
    # cannot meet a squeezing target under any parameters
    bogus = space.empty()
    bogus[onsite] = 2

    libraries = Libraries()
    libraries.add_valid(bogus, {"loss_reached": 0.0, "max_real_eigenvalue": -1.0,
                                "x": np.zeros(optimizer.param.num_variables)})
    # witness mode is deterministic: the stored x cannot satisfy a graph that
    # emits vacuum, so this fails every time rather than sometimes
    failures = revalidate_valid_library(optimizer, libraries)
    assert len(failures) == 1
    assert np.array_equal(failures[0], bogus)

    revalidate_valid_library(optimizer, libraries, graphs=[bogus], remove=True)
    assert len(libraries.valid) == 0


def test_reoptimising_audit_is_not_a_soundness_test():
    """Guard the distinction the two-library design rests on.

    A genuine VALID entry always passes the witness audit.  Re-optimising can
    fail it purely because the restarts landed badly -- an existential failure
    of the optimiser, not a proof that no witness exists.  Witness mode must
    accept what a weaker re-optimisation might reject.
    """
    from autogaussian.search import revalidate_valid_library
    from autogaussian.types import Libraries

    _, optimizer = build(num_auxiliary_modes=0)
    space = optimizer.space
    onsite = space.coherent.slots.index(("onsite_squeezing", 0, 0))
    port0 = space.channels.index_of((0,), ChannelClass.PARTICLE_CONSERVING,
                                    Access.CONTROLLED)
    graph = space.empty()
    graph[onsite] = 2
    graph[space.num_coherent_slots + port0] = 1

    success, infos = optimizer.test_graph(graph)
    assert success
    witness = next(info for info in infos if info["success"])

    libraries = Libraries()
    libraries.add_valid(graph, witness)
    assert revalidate_valid_library(optimizer, libraries) == []
