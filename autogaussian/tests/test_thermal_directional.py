"""B.3(h) -- thermal-resilient directional squeezer (hot intrinsic-loss channel).

The target is the cold B.3 directional squeezer plus one adversary: a designated
intrinsic-loss channel held at occupation ``n > 0``.  What is asserted here, in
the order the brief establishes it:

1.  the plumbing is a no-op when the bath is cold (regression guard);
2.  the cold witness *breaks on purpose* when its channel is heated, by exactly
    the ``2 n [N e_k N^dag]`` inflation formula;
3.  purity under a hot bath is equivalent to exact noise evasion, and is
    therefore *obstructed* -- certifiably, at every mode count -- when the
    target itself forces the bath onto the monitored port;
4.  where evasion is possible, a resilient witness exists and costs one
    auxiliary mode more than the cold baseline.

No exact mode count is claimed anywhere; step 4 asserts the *transition*.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogaussian import (
    CovarianceArchitectureOptimizer,
    MinimumIntrinsicLoss,
    PurityFloor,
    QuadratureSpectrum,
    certify_bath_unhosted,
    certify_hot_purity_obstruction,
    certify_passive_range,
    noise_response_block,
)
from autogaussian.certificates import certify_fit_infeasible
from autogaussian.graph import (
    NO_COUPLING,
    SLOT_ONSITE_SQUEEZING,
    SLOT_TWO_MODE_SQUEEZING,
)
from autogaussian.nambu import stack_input_covariance, thermal_channel_covariance
from autogaussian.gallery import (
    directional_squeezed_source,
    hot_channel_inflation,
    thermal_directional_squeezer,
)

VARIANCE = 0.2
R = -0.5 * np.log(VARIANCE)
HOT = 2                     # the first auxiliary mode
N_THERMAL = 0.5
TRANSMISSION = 1.3
GAMMA_MIN = 0.1


def build(problem, num_auxiliary_modes, seed=1, num_tests=60):
    return CovarianceArchitectureOptimizer(
        problem.target, num_auxiliary_modes=num_auxiliary_modes, seed=seed,
        make_initial_test=False, kwargs_optimization={"num_tests": num_tests},
        **problem.optimizer_kwargs())


def port_block(V, port):
    rows = slice(2 * port, 2 * port + 2)
    return np.real(np.asarray(V))[rows, rows]


# ---------------------------------------------------------------------------
# 0. plumbing: the thermal noise block is opt-in
# ---------------------------------------------------------------------------

def test_noise_block_defaults_to_vacuum():
    """The regression guard: nothing changes unless a bath is declared."""
    signal = np.eye(4, dtype=complex)
    default = np.asarray(stack_input_covariance(signal, 2))
    explicit = np.asarray(stack_input_covariance(signal, 2, sigma_noise=np.eye(4)))
    assert np.allclose(default, np.eye(8))
    assert np.allclose(default, explicit)


def test_thermal_channel_covariance_is_2n_plus_1_on_the_hot_channel():
    sigma = np.asarray(thermal_channel_covariance({1: 0.75}, 3))
    assert np.allclose(np.diag(sigma).real, [1.0, 2.5, 1.0, 1.0, 2.5, 1.0])
    assert np.allclose(sigma - np.diag(np.diag(sigma)), 0.0)   # no anomalous part


def test_oracle_reads_the_declared_occupations_back():
    problem = thermal_directional_squeezer(r=R, n_thermal=N_THERMAL, hot_channel=HOT)
    optimizer = build(problem, num_auxiliary_modes=1, num_tests=1)
    occupations = optimizer.oracle.channel_occupations
    assert optimizer.oracle.has_hot_channel
    assert occupations[HOT] == pytest.approx(N_THERMAL)
    assert np.allclose(np.delete(occupations, HOT), 0.0)


# ---------------------------------------------------------------------------
# 1. cold baseline -- one-parameter deformation of a target that already passes
# ---------------------------------------------------------------------------

def test_cold_limit_reproduces_the_B3_target():
    """``n = 0, min_loss = 0`` is the cold B.3 target, gauge-free.

    The cold fixture pins ``sigma_xx`` and ``sigma_pp`` of the squeezed port
    separately; the hot-channel target pins the same block's *eigenvalues*.  A
    cold B.3 witness must therefore satisfy the deformed target's constraints to
    the same tolerance.
    """
    cold = directional_squeezed_source(variance=VARIANCE,
                                       forward_transmission=TRANSMISSION,
                                       isolate_backward=True)
    optimizer = build(cold, num_auxiliary_modes=1, seed=3, num_tests=30)
    success, infos = optimizer.test_graph(optimizer.space.fully_connected())
    assert success, "the cold B.3 baseline itself must pass"

    deformed = thermal_directional_squeezer(
        r=R, n_thermal=0.0, min_loss=0.0, hot_channel=HOT,
        forward_transmission=TRANSMISSION, isolate_backward=True)
    check = build(deformed, num_auxiliary_modes=1, seed=3, num_tests=1)
    residual = np.asarray(check.oracle.residual_func(infos[-1]["x"]))
    # the invariants (trace, det) are nonlinear in the entries the cold fit
    # drove to ~1e-5, so the deformed loss sits a couple of orders above the
    # cold one -- what matters is that it is the *same* solution, not a new one
    assert 0.5 * float(np.sum(residual ** 2)) < 1.0e-7
    assert float(np.max(np.abs(residual))) < 1.0e-4


def test_cold_baseline_needs_one_auxiliary_mode():
    """Same open (but cold) loss channel: one auxiliary mode is enough."""
    problem = thermal_directional_squeezer(
        r=R, n_thermal=0.0, min_loss=GAMMA_MIN, hot_channel=HOT,
        forward_transmission=TRANSMISSION)
    optimizer = build(problem, num_auxiliary_modes=1)
    success, infos = optimizer.test_graph(optimizer.space.fully_connected())
    assert success
    V = optimizer.oracle.covariance(infos[-1]["x"], 0.0)
    block = port_block(V, 1)
    assert np.linalg.det(block) == pytest.approx(1.0, abs=1.0e-5)
    assert sorted(np.linalg.eigvalsh(block)) == pytest.approx(
        [VARIANCE, 1.0 / VARIANCE], abs=1.0e-4)


# ---------------------------------------------------------------------------
# 2. the break -- heat the cold witness and re-evaluate, without re-fitting
# ---------------------------------------------------------------------------

def test_heating_a_cold_witness_breaks_purity_by_the_inflation_formula():
    problem = thermal_directional_squeezer(
        r=R, n_thermal=0.0, min_loss=GAMMA_MIN, hot_channel=HOT,
        forward_transmission=TRANSMISSION)
    optimizer = build(problem, num_auxiliary_modes=1)
    success, infos = optimizer.test_graph(optimizer.space.fully_connected())
    assert success
    x = infos[-1]["x"]
    cold_block = port_block(optimizer.oracle.covariance(x, 0.0), 1)

    hot_problem = thermal_directional_squeezer(
        r=R, n_thermal=N_THERMAL, min_loss=GAMMA_MIN, hot_channel=HOT,
        forward_transmission=TRANSMISSION)
    hot = build(hot_problem, num_auxiliary_modes=1, num_tests=1)
    hot_block = port_block(hot.oracle.covariance(x, 0.0), 1)

    H, gamma, kappa_tilde, _ = optimizer.param.unpack(x)
    predicted = hot_channel_inflation(
        N_THERMAL,
        noise_response_block(H, gamma, kappa_tilde, 0.0, port=1, channel=HOT,
                             thetas=np.zeros(optimizer.num_modes)))

    # the same graph, the same parameters, only the bath changed
    assert np.allclose(hot_block - cold_block, predicted, atol=1.0e-9)
    # ... and it fails purity on purpose
    assert np.linalg.det(cold_block) == pytest.approx(1.0, abs=1.0e-5)
    assert np.linalg.det(hot_block) > 1.0 + 1.0e-3


def test_inflation_is_positive_semidefinite():
    """``B = 2 n [N e_k N^dag]`` is PSD -- the reason purity can only be lost."""
    problem = thermal_directional_squeezer(
        r=R, n_thermal=0.0, min_loss=GAMMA_MIN, hot_channel=HOT,
        forward_transmission=TRANSMISSION)
    optimizer = build(problem, num_auxiliary_modes=1, num_tests=4)
    rng = np.random.default_rng(0)
    free = optimizer.param.free_indices(optimizer.space.fully_connected())
    for _ in range(5):
        x = np.zeros(optimizer.param.num_variables)
        x[free] = rng.normal(size=free.size)
        H, gamma, kappa_tilde, _ = optimizer.param.unpack(x)
        block = np.asarray(noise_response_block(
            H, gamma, kappa_tilde, 0.0, port=1, channel=HOT,
            thetas=np.zeros(optimizer.num_modes)))
        assert np.min(np.linalg.eigvalsh(block)) > -1.0e-12


# ---------------------------------------------------------------------------
# 3. the obstruction -- purity under a hot bath *is* noise evasion
# ---------------------------------------------------------------------------

def test_hot_channel_on_the_other_port_is_certified_infeasible():
    """``N_{jk} = S_{jk} sqrt(gamma_k)``: the transmission pin forces the bath
    onto the monitored port, so ``det = 1`` is unreachable at any mode count."""
    problem = thermal_directional_squeezer(
        r=R, n_thermal=N_THERMAL, hot_channel=0, min_loss=GAMMA_MIN,
        forward_transmission=TRANSMISSION, intrinsic_losses=[True])
    for num_auxiliary_modes in (1, 2):
        optimizer = build(problem, num_auxiliary_modes=num_auxiliary_modes,
                          num_tests=1)
        graph = optimizer.space.fully_connected()
        fired, detail = certify_hot_purity_obstruction(optimizer, graph)
        assert fired and detail["conclusive"]
        assert detail["hot_channel"] == 0 and detail["monitored_port"] == 1
        # and the general entry point reaches the same verdict
        assert certify_fit_infeasible(optimizer, graph, cache={})[0]


def test_obstruction_does_not_fire_when_evasion_is_possible():
    """Hot channel on an auxiliary: nothing is proved, so nothing is claimed."""
    problem = thermal_directional_squeezer(
        r=R, n_thermal=N_THERMAL, hot_channel=HOT, min_loss=GAMMA_MIN,
        forward_transmission=TRANSMISSION)
    optimizer = build(problem, num_auxiliary_modes=2, num_tests=1)
    fired, detail = certify_hot_purity_obstruction(
        optimizer, optimizer.space.fully_connected())
    assert not fired
    assert not detail["conclusive"]


def test_obstruction_is_silent_without_a_loss_floor():
    """With ``gamma_k`` free the optimiser may disconnect the bath instead of
    defeating it, so the argument does not apply and must not fire."""
    problem = thermal_directional_squeezer(
        r=R, n_thermal=N_THERMAL, hot_channel=0, min_loss=0.0,
        forward_transmission=TRANSMISSION, intrinsic_losses=[True])
    optimizer = build(problem, num_auxiliary_modes=1, num_tests=1)
    assert not certify_hot_purity_obstruction(
        optimizer, optimizer.space.fully_connected())[0]


def test_bath_needs_a_live_loss_channel():
    problem = thermal_directional_squeezer(
        r=R, n_thermal=N_THERMAL, hot_channel=HOT, min_loss=GAMMA_MIN,
        forward_transmission=TRANSMISSION, intrinsic_losses=False)
    optimizer = build(problem, num_auxiliary_modes=2, num_tests=1)
    fired, detail = certify_bath_unhosted(optimizer)
    assert fired and detail["conclusive"]
    assert detail["channels"] == [HOT]

    hosted = thermal_directional_squeezer(
        r=R, n_thermal=N_THERMAL, hot_channel=HOT, min_loss=GAMMA_MIN,
        forward_transmission=TRANSMISSION)
    assert not certify_bath_unhosted(build(hosted, 2, num_tests=1))[0]


def test_passive_graph_still_certified_under_a_hot_bath():
    """A hot bath enlarges the reachable set but cannot tilt it: a passive graph
    emits isotropic port blocks, so the squeezing pin stays unreachable."""
    problem = thermal_directional_squeezer(
        r=R, n_thermal=N_THERMAL, hot_channel=HOT, min_loss=GAMMA_MIN,
        forward_transmission=TRANSMISSION)
    optimizer = build(problem, num_auxiliary_modes=2, num_tests=1)
    graph = optimizer.space.fully_connected().copy()
    for slot, (kind, _, _) in enumerate(optimizer.space.slots):
        if kind in (SLOT_ONSITE_SQUEEZING, SLOT_TWO_MODE_SQUEEZING):
            graph[slot] = NO_COUPLING
    fired, detail = certify_passive_range(optimizer, graph)
    assert fired and detail["conclusive"]
    assert detail["test"] == "passive_output_is_isotropic"


# ---------------------------------------------------------------------------
# 4. thermal-resilient fit and the cost of resilience
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 2])
def test_one_auxiliary_mode_is_not_enough_when_the_bath_is_hot(seed):
    """A *provisional*, non-condemning invalid: the fit misses on every restart
    where the cold baseline passed comfortably."""
    problem = thermal_directional_squeezer(
        r=R, n_thermal=N_THERMAL, hot_channel=HOT, min_loss=GAMMA_MIN,
        forward_transmission=TRANSMISSION)
    optimizer = build(problem, num_auxiliary_modes=1, seed=seed)
    success, infos = optimizer.test_graph(optimizer.space.fully_connected())
    assert not success
    assert not any(info["loss_below_tolerance"] for info in infos)
    # nothing was proved -- no certificate may condemn this graph
    assert not certify_fit_infeasible(
        optimizer, optimizer.space.fully_connected(), cache={})[0]


def test_adding_one_auxiliary_mode_flips_it_valid():
    problem = thermal_directional_squeezer(
        r=R, n_thermal=N_THERMAL, hot_channel=HOT, min_loss=GAMMA_MIN,
        forward_transmission=TRANSMISSION)
    optimizer = build(problem, num_auxiliary_modes=2, seed=1)
    success, infos = optimizer.test_graph(optimizer.space.fully_connected())
    assert success, "no resilient witness found; the transition claim fails"

    witness = infos[-1]
    x = witness["x"]
    block = port_block(optimizer.oracle.covariance(x, 0.0), 1)
    assert np.linalg.det(block) == pytest.approx(1.0, abs=1.0e-4)
    assert sorted(np.linalg.eigvalsh(block)) == pytest.approx(
        [VARIANCE, 1.0 / VARIANCE], abs=1.0e-3)

    S, _ = optimizer.oracle.scattering(x, 0.0)
    assert abs(S[1, 0]) ** 2 == pytest.approx(TRANSMISSION, abs=1.0e-3)
    assert abs(S[0, 1]) ** 2 < 1.0e-6

    gamma = np.asarray(optimizer.param.gamma(x))
    assert gamma[HOT] >= GAMMA_MIN - 1.0e-4      # the bath stayed switched on
    assert witness["max_real_eigenvalue"] < 0.0  # and the device is stable


def test_resilient_witness_evades_the_hot_channel():
    """The theorem, measured: ``det = 1`` at the monitored port forces
    ``N_{j,k*} -> 0``.  Reported, not assumed."""
    problem = thermal_directional_squeezer(
        r=R, n_thermal=N_THERMAL, hot_channel=HOT, min_loss=GAMMA_MIN,
        forward_transmission=TRANSMISSION)
    optimizer = build(problem, num_auxiliary_modes=2, seed=1)
    success, infos = optimizer.test_graph(optimizer.space.fully_connected())
    assert success
    x = infos[-1]["x"]
    H, gamma, kappa_tilde, _ = optimizer.param.unpack(x)
    inflation = hot_channel_inflation(
        N_THERMAL,
        noise_response_block(H, gamma, kappa_tilde, 0.0, port=1, channel=HOT,
                             thetas=np.zeros(optimizer.num_modes)))
    block = port_block(optimizer.oracle.covariance(x, 0.0), 1)
    excess = np.linalg.det(block) - 1.0

    assert np.linalg.norm(inflation) < 1.0e-3
    # det(A + B) - det A is controlled by B; both must vanish together
    assert excess < 1.0e-4
    assert np.linalg.norm(inflation) >= 0.0


# ---------------------------------------------------------------------------
# 5. constraint algebra used by the target
# ---------------------------------------------------------------------------

def test_quadrature_spectrum_is_gauge_free():
    """Rotating the squeeze angle leaves the eigenvalue residual unchanged."""
    from autogaussian.constraints import ForwardContext

    def context(angle):
        eigen = np.diag([VARIANCE, 1.0 / VARIANCE])
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        block = rotation @ eigen @ rotation.T
        V = np.eye(4, dtype=complex)
        V[2:, 2:] = block
        return ForwardContext(x=None, H=None, gamma=None, kappa_tilde=None,
                              thetas=None, omegas=np.array([0.0]),
                              S=None, N=None, V=np.array([V]),
                              num_modes=2, num_ports=2)

    constraint = QuadratureSpectrum.squeezed(port=1, r=R)
    for angle in (0.0, 0.3, 1.1):
        assert np.allclose(np.asarray(constraint(context(angle))), 0.0, atol=1e-12)
    assert constraint.r == pytest.approx(R)

    purity = PurityFloor(port=1)
    assert np.allclose(np.asarray(purity(context(0.7))), 0.0, atol=1e-12)


def test_minimum_intrinsic_loss_is_one_sided():
    from autogaussian.constraints import ForwardContext

    def context(gamma):
        return ForwardContext(x=None, H=None, gamma=np.array([gamma]),
                              kappa_tilde=None, thetas=None,
                              omegas=np.array([0.0]), S=None, N=None, V=None,
                              num_modes=1, num_ports=1)

    constraint = MinimumIntrinsicLoss(mode=0, minimum=0.1)
    assert float(np.asarray(constraint(context(0.2)))[0]) == pytest.approx(0.0)
    assert float(np.asarray(constraint(context(0.04)))[0]) == pytest.approx(0.06)
