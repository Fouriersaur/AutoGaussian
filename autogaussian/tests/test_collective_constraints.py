"""Sec. 10 constraints on the collective engine (Addendum Sec. 4).

The constraint block of :class:`~autogaussian.oracle.CovarianceOracle` reads the
forward map through a :class:`~autogaussian.constraints.ForwardContext`, which
the base spec builds from ``Parametrization.unpack(x)``.  The collective
parametrisation has no such four-tuple -- Sec. 4 promoted ``kappa~`` and
``gamma`` to jump amplitudes -- so the context is built by an overridable
``_forward_context`` hook instead, and this suite pins what that hook has to
deliver:

  * a constrained target *runs at all* on the collective engine (before the
    hook existed, every constrained fit died in ``unpack``);
  * the reconstructed per-mode rates are the rates the active channels add up
    to, and reduce to the base spec's ``kappa~_i`` / ``Gamma_i`` on a private
    scaffold;
  * ``S`` reaching the constraints is the collective scattering matrix, indexed
    so that ``S[j, i]`` is still port ``j`` <- port ``i`` for the monitored
    ports;
  * the B.3 device the notebook reports is real: at **zero** auxiliary modes the
    private scaffold cannot meet the directional target (the App. F counting)
    while a shared Bogoliubov bath can.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp

from autogaussian.collective_oracle import CollectiveArchitectureOptimizer
from autogaussian.gallery import directional_squeezed_source
from autogaussian.hypergraph import collective_usage_report

DIRECTIONAL_COLOUR = dict(max_support=2, collective_bogoliubov=True)


def problem():
    return directional_squeezed_source(variance=0.2, forward_transmission=1.3,
                                       isolate_backward=True)


def optimizer(channel_kwargs=None, num_auxiliary_modes=0, num_tests=40, seed=3):
    spec = problem()
    return CollectiveArchitectureOptimizer(
        spec.target, num_auxiliary_modes=num_auxiliary_modes, seed=seed,
        make_initial_test=False,
        channel_kwargs=DIRECTIONAL_COLOUR if channel_kwargs is None else channel_kwargs,
        kwargs_optimization={"num_tests": num_tests},
        **spec.optimizer_kwargs())


@pytest.fixture(scope="module")
def directional_witness():
    """A converged B.3 device on two modes and a shared Bogoliubov bath."""
    opt = optimizer()
    graph = opt.space.fully_connected()
    success, infos = opt.test_graph(graph)
    assert success, ("no valid witness on the collective colour (best loss %.2e)"
                     % min(info["loss_reached"] for info in infos))
    witness = [info for info in infos if info["success"]][-1]
    return opt, graph, witness["x"]


def test_constrained_fit_runs_on_the_collective_engine(directional_witness):
    """The regression: a Sec. 10 constraint used to raise inside ``unpack``."""
    opt, _, x = directional_witness
    assert opt.oracle.fit_loss(x) < 1.0e-9


def test_the_transmission_and_isolation_pins_are_actually_met(directional_witness):
    opt, _, x = directional_witness
    S, _ = opt.oracle.scattering(x, 0.0)
    assert abs(S[1, 0]) ** 2 == pytest.approx(1.3, abs=1.0e-4)   # forward pin
    assert abs(S[0, 1]) ** 2 < 1.0e-8                            # backward silenced


def test_the_target_itself_is_met_on_the_monitored_ports(directional_witness):
    opt, _, x = directional_witness
    V = np.real(np.asarray(opt.oracle.covariance(x, 0.0)))
    assert V[0, 0] == pytest.approx(1.0, abs=1.0e-4)
    assert V[1, 1] == pytest.approx(1.0, abs=1.0e-4)
    assert V[2, 2] == pytest.approx(0.2, abs=1.0e-4)
    assert V[3, 3] == pytest.approx(5.0, abs=1.0e-3)


def test_scattering_helper_agrees_with_the_collective_forward_map(directional_witness):
    """``oracle.scattering`` must route through ``_responses``, not through the
    base spec's ``response_matrices`` -- the two forward maps differ."""
    opt, _, x = directional_witness
    for omega in (0.0, 0.35, -1.2):
        S, N = opt.oracle.scattering(x, omega)
        S_ref, N_ref, _ = opt.oracle._responses(jnp.asarray(x), float(omega))
        assert np.allclose(S, np.asarray(S_ref))
        assert np.allclose(N, np.asarray(N_ref))


def test_context_rates_are_the_rates_the_channels_add_up_to(directional_witness):
    """``kappa~_i`` / ``Gamma_i`` no longer exist as scalars; the hook has to
    rebuild them as ``sum_mu |u_{mu,i}|^2 + |v_{mu,i}|^2``."""
    opt, _, x = directional_witness
    ctx = opt.oracle._forward_context(jnp.asarray(x), None, None, None)
    Uk, Vk, Ug, Vg = opt.param.channels.coupling_rows(jnp.asarray(x))
    expected_kappa = np.sum(np.abs(np.asarray(Uk)) ** 2 + np.abs(np.asarray(Vk)) ** 2, axis=0)
    expected_gamma = np.sum(np.abs(np.asarray(Ug)) ** 2 + np.abs(np.asarray(Vg)) ** 2, axis=0)
    assert np.allclose(np.asarray(ctx.kappa_tilde), expected_kappa)
    assert np.allclose(np.asarray(ctx.gamma), expected_gamma)
    assert np.asarray(ctx.H).shape == (2 * opt.num_modes, 2 * opt.num_modes)


def test_private_scaffold_reduces_to_the_base_spec_rates():
    """On the support-1 corner the reconstruction is the base spec's own
    ``kappa~_i`` and ``Gamma_i``, one channel per mode."""
    opt = optimizer(channel_kwargs=dict(max_support=1), num_tests=1)
    graph = opt.space.fully_connected()
    rng = np.random.default_rng(0)
    x = rng.normal(size=opt.param.num_variables)
    ctx = opt.oracle._forward_context(jnp.asarray(x), None, None, None)

    channels = opt.param.channel_set(graph, x)
    kappa = np.zeros(opt.num_modes)
    gamma = np.zeros(opt.num_modes)
    for channel in channels:
        mode = channel.support[0]
        rate = abs(channel.u[mode]) ** 2 + abs(channel.v[mode]) ** 2
        if channel.access.name == "CONTROLLED":
            kappa[mode] += rate
        else:
            gamma[mode] += rate
    assert np.allclose(np.asarray(ctx.kappa_tilde), kappa)
    assert np.allclose(np.asarray(ctx.gamma), gamma)


def test_private_scaffold_cannot_do_B3_without_an_auxiliary():
    """App. F on the base spec's scaffold: two directly coupled modes, private
    channels only, cannot be made directional.  The fit stalls far above the
    success threshold rather than merely missing it by a restart."""
    opt = optimizer(channel_kwargs=dict(max_support=1), num_tests=40)
    success, infos = opt.test_graph(opt.space.fully_connected())
    assert not success
    assert min(info["loss_reached"] for info in infos) > 1.0e-3


def test_a_shared_bogoliubov_bath_removes_the_auxiliary(directional_witness):
    """What the notebook's B.3 port claims: the same target closes on two modes
    once the dissipative scaffold is searchable, and the shared channel is
    genuinely used."""
    opt, graph, x = directional_witness
    assert opt.num_modes == 2
    channels = opt.param.channel_set(graph, x)
    report = collective_usage_report(opt.space, graph, channels)
    assert not report["private_sufficed"]
    assert report["num_collective_used"] >= 1
