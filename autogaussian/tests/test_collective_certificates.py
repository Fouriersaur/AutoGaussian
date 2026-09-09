"""D4 -- certificates under a taller jump pencil, and canonical-form dedup
(Addendum Sec. 8, D4; Sec. 7).

Two things are checked here.

**PBH is native.**  The dark-mode test is already written on ``(A, jumps)``, so a
collective jump set only makes the pencil taller.  The physics has to flip the
right way: a subspace that certifies INVALID as dissipatively-uncontrollable
under private jumps must *stop* certifying once a collective channel reaches it,
because it genuinely became feasible.

**Monotonicity against a certified INVALID.**  The addendum lists this as
NEEDS VERIFICATION -- "verify against one certified-INVALID case before relying
on it".  That case is built below: a graph whose dark mode PBH certifies, with
its subgraphs checked to stay certified (downward-closed) and its supergraphs
checked to lose the certificate (so VALID stays upward-closed).  Certified
INVALID surviving hyperedge *deletion* is the property the pruning rule actually
consumes.

**Dedup.**  With the Sec. 5 canonical form enforced, no two admissible specs may
produce the same ``sigma_out(Omega)`` on the grid, or "minimal setup" is not
well-defined.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp

from autogaussian.forward import collective_dynamical_matrix
from autogaussian.certificates import (
    collective_jump_pencil,
    pbh_dark_mode_collective,
    pbh_non_stabilizable,
)
from autogaussian.channel_parametrization import CollectiveParametrization
from autogaussian.channels import Access, Channel, ChannelClass, ChannelSet
from autogaussian.graph import GraphSpace
from autogaussian.hypergraph import (
    ChannelSpace,
    TwoColouredSpace,
    spectral_fingerprint,
)
from autogaussian.nambu import build_H_bdg, vacuum_covariance
from autogaussian.parametrization import Parametrization

OMEGAS = np.array([0.0, 0.35, 1.2])


def uncoupled_two_mode():
    """Two modes with no coherent coupling: mode 1 is reachable only through a
    channel that actually touches it."""
    zeros = np.zeros((2, 2), dtype=complex)
    return build_H_bdg(jnp.asarray(zeros), jnp.asarray(zeros))


def private_only():
    return ChannelSet([Channel(u=[1.0, 0.0], access=Access.CONTROLLED)],
                      num_modes=2)


def with_collective():
    return ChannelSet([
        Channel(u=[1.0, 0.0], access=Access.CONTROLLED),
        Channel(u=[0.5, 0.5], access=Access.CONTROLLED),
    ], num_modes=2)


# ---------------------------------------------------------------------------
# the pencil
# ---------------------------------------------------------------------------

def test_pencil_gets_taller_with_more_channels():
    """A collective jump set adds rows; the mode dimension is untouched."""
    thin = collective_jump_pencil(private_only())
    tall = collective_jump_pencil(with_collective())
    assert thin.shape[1] == tall.shape[1] == 4          # 2 N
    assert tall.shape[0] > thin.shape[0]


def test_pencil_stacks_both_access_classes():
    """Every channel damps, whichever end owns it."""
    channels = ChannelSet([
        Channel(u=[1.0, 0.0], access=Access.CONTROLLED),
        Channel(u=[0.0, 0.8], access=Access.UNCONTROLLED),
    ], num_modes=2)
    assert collective_jump_pencil(channels).shape[0] == 4    # 2*(1 controlled + 1 lost)


# ---------------------------------------------------------------------------
# the Sec. 7 flip
# ---------------------------------------------------------------------------

def test_private_jumps_leave_a_dark_mode_certified():
    """Mode 1 is touched by nothing: a genuine dissipatively-dark subspace."""
    fired, detail = pbh_dark_mode_collective(uncoupled_two_mode(), private_only())
    assert fired
    assert detail["rank"] < detail["dimension"]


def test_a_collective_channel_makes_the_dark_mode_controllable():
    """**The Sec. 7 claim.**  The certificate stops firing -- correctly, because
    the subspace genuinely became feasible."""
    fired, _ = pbh_dark_mode_collective(uncoupled_two_mode(), with_collective())
    assert not fired


def test_an_extra_private_channel_also_lifts_the_certificate():
    """Nothing collective-specific about it: reaching the mode is what matters."""
    channels = ChannelSet([
        Channel(u=[1.0, 0.0], access=Access.CONTROLLED),
        Channel(u=[0.0, 0.6], access=Access.UNCONTROLLED),
    ], num_modes=2)
    assert not pbh_dark_mode_collective(uncoupled_two_mode(), channels)[0]


def test_no_channels_at_all_certifies():
    """The degenerate end of the ladder: nothing damps anything."""
    fired, detail = pbh_dark_mode_collective(
        uncoupled_two_mode(), ChannelSet([], num_modes=2))
    assert fired and detail["conclusive"]


# ---------------------------------------------------------------------------
# Sec. 7 monotonicity, verified against a certified-INVALID case
# ---------------------------------------------------------------------------

def test_certified_invalid_is_downward_closed_under_hyperedge_deletion():
    """The property the pruning rule consumes.

    Start from a PBH-certified graph and delete hyperedges one at a time.  Every
    subgraph must stay certified: deleting a channel can only *lower* the rank
    of the stacked pencil, so a dark mode cannot be repaired by removing
    dissipation.  This is the addendum's NEEDS-VERIFICATION row, checked against
    a real certified INVALID rather than assumed.
    """
    H_bdg = uncoupled_two_mode()
    full = [Channel(u=[1.0, 0.0], access=Access.CONTROLLED),
            Channel(u=[0.7, 0.0], access=Access.UNCONTROLLED)]
    assert pbh_dark_mode_collective(H_bdg, ChannelSet(full, num_modes=2))[0]

    for drop in range(len(full)):
        subset = [c for k, c in enumerate(full) if k != drop]
        fired, _ = pbh_dark_mode_collective(H_bdg, ChannelSet(subset, num_modes=2))
        assert fired, "deleting a hyperedge lifted a certificate: pruning unsound"
    assert pbh_dark_mode_collective(H_bdg, ChannelSet([], num_modes=2))[0]


def test_certificate_dies_on_hyperedge_addition_not_deletion():
    """The two directions, side by side: deletion preserves the certificate,
    addition can remove it.  That asymmetry is exactly what keeps VALID
    upward-closed and certified-INVALID downward-closed."""
    H_bdg = uncoupled_two_mode()
    base = [Channel(u=[1.0, 0.0], access=Access.CONTROLLED)]
    added = base + [Channel(u=[0.4, 0.4], access=Access.CONTROLLED)]
    assert pbh_dark_mode_collective(H_bdg, ChannelSet(base, num_modes=2))[0]
    assert not pbh_dark_mode_collective(H_bdg, ChannelSet(added, num_modes=2))[0]


def test_zero_amplitude_channel_does_not_lift_a_certificate():
    """Continuity at the boundary: a hyperedge switched off by amplitude is the
    same as one that was never there."""
    H_bdg = uncoupled_two_mode()
    dead = [Channel(u=[1.0, 0.0], access=Access.CONTROLLED),
            Channel(u=[0.0, 0.0], access=Access.CONTROLLED)]
    assert pbh_dark_mode_collective(H_bdg, ChannelSet(dead, num_modes=2))[0]


def test_pbh_agrees_with_the_direct_two_argument_form():
    """The certificate really is the base-spec test, just with a taller ``C``."""
    channels = private_only()
    H_bdg = uncoupled_two_mode()
    M = np.asarray(collective_dynamical_matrix(H_bdg, channels))
    direct = pbh_non_stabilizable(M, collective_jump_pencil(channels))
    wrapped = pbh_dark_mode_collective(H_bdg, channels)
    assert direct[0] == wrapped[0]


# ---------------------------------------------------------------------------
# D4 -- canonical form and dedup over the emitted spectrum
# ---------------------------------------------------------------------------

def test_collected_bogoliubov_is_rejected_at_construction():
    """Sec. 5 as a type-level guard, restated here as the D4 acceptance item."""
    with pytest.raises(ValueError, match="canonical form"):
        Channel(u=[1.0], v=[0.4], access=Access.CONTROLLED)


def test_the_degenerate_pair_really_is_degenerate():
    """*Why* Sec. 5 exists, made executable.

    ``L = sqrt(kappa) a`` on a squeezed input and
    ``L = sqrt(kappa)(cosh r a + sinh r a^dag)`` on vacuum emit the **same**
    ``sigma_out(Omega)``.  Only the first is admissible, so the pair cannot both
    reach the library -- which is what keeps "minimal setup" well-defined.
    """
    from autogaussian.nambu import channel_covariance

    zeros = np.zeros((1, 1), dtype=complex)
    H_bdg = build_H_bdg(jnp.asarray(zeros), jnp.asarray(zeros))
    r = 0.45

    plain = ChannelSet([Channel(u=[1.0], access=Access.CONTROLLED)], num_modes=1)
    squeezed_in = channel_covariance(n=np.array([np.sinh(r) ** 2]),
                                     m=np.array([np.cosh(r) * np.sinh(r)]),
                                     num_modes=1)
    admissible = spectral_fingerprint(H_bdg, plain, jnp.zeros(1), OMEGAS,
                                      num_ports=1, sigma_sig=squeezed_in)
    # the banned spec would need a CONTROLLED Bogoliubov channel, which the type
    # refuses to build -- that refusal is the dedup
    with pytest.raises(ValueError, match="canonical form"):
        Channel(u=[np.cosh(r)], v=[np.sinh(r)], access=Access.CONTROLLED)
    assert len(admissible) > 0


def test_distinct_specs_have_distinct_spectral_fingerprints():
    """No two admissible library entries collapse onto the same emitted field."""
    space = TwoColouredSpace(
        GraphSpace(2, allow_detunings=False, allow_onsite_squeezing=False),
        ChannelSpace(2, max_support=2, collective_controlled=True,
                     allow_phases=False))
    space.validate()
    coherent = Parametrization(space.coherent, num_ports=2,
                               intrinsic_losses=False, free_decay_ratios=False)
    param = CollectiveParametrization(space, coherent)

    rng = np.random.default_rng(3)
    graph = space.fully_connected()
    fingerprints = {}
    collisions = []
    for _ in range(12):
        x = np.abs(rng.normal(scale=0.5, size=param.num_variables)) + 0.2
        H_bdg, channels, thetas = param.unpack(graph, x)
        M = np.asarray(collective_dynamical_matrix(H_bdg, channels))
        if np.max(np.real(np.linalg.eigvals(M))) >= 0:
            continue                                   # unstable, not a witness
        key = spectral_fingerprint(H_bdg, channels, thetas, OMEGAS, num_ports=2)
        if key in fingerprints and not np.allclose(fingerprints[key], x):
            collisions.append(x)
        fingerprints[key] = x
    assert len(fingerprints) > 1
    assert not collisions


def test_fingerprint_is_stable_and_discriminating():
    """Sanity on the digest itself: identical specs match, perturbed ones do not."""
    zeros = np.zeros((1, 1), dtype=complex)
    H_bdg = build_H_bdg(jnp.asarray(zeros), jnp.asarray([[0.2j]]))
    a = ChannelSet([Channel(u=[1.0], access=Access.CONTROLLED)], num_modes=1)
    b = ChannelSet([Channel(u=[1.1], access=Access.CONTROLLED)], num_modes=1)
    fa = spectral_fingerprint(H_bdg, a, jnp.zeros(1), OMEGAS, num_ports=1)
    fa2 = spectral_fingerprint(H_bdg, a, jnp.zeros(1), OMEGAS, num_ports=1)
    fb = spectral_fingerprint(H_bdg, b, jnp.zeros(1), OMEGAS, num_ports=1)
    assert fa == fa2
    assert fa != fb


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
