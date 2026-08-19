"""Frequency-dependent (full-spectrum) gallery targets -- App. B.8 - B.12.

These are the only golden tests that exercise the genuinely new code paths of
the sigma_out(Omega) formulation: a multi-point ``omega_grid``, the **live
decay ratios kappa~_i**, and the phase-sensitive 2N x 2N Nambu block (the
Omega-dependent x/p cross-spectrum).  A build that fixed kappa~, or that
reduced the forward map to the phase-preserving N x N block, fails them.

Two claims are checked structurally rather than numerically, because they
decide how the targets may be written at all:

* ``V(-Omega) = V(Omega)^T`` -- so every diagonal spectrum is even in Omega and
  the grids below run over ``Omega >= 0`` only;
* the spectral *degree* of the target sets the mode count (B.10, B.12): one
  mode short of the structural minimum, the fit never reaches tolerance.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp

from autogaussian import CovarianceArchitectureOptimizer, duan_sum
from autogaussian.forward import output_covariance_quadrature
from autogaussian.gallery import (
    SPECTRAL_GALLERY,
    band_limited_epr,
    bandpass_squeezer,
    butterworth_dip,
    duan_of_band_limited_epr,
    filter_cavity_angle,
    flat_top_squeezer,
    lorentzian_dip,
    notch_squeezer,
    rotated_squeezing_block,
    squeeze_angle_rotation,
)
from autogaussian.nambu import build_H_bdg, channel_covariance, squeezed_bath, vacuum_covariance

SEED = 7
NUM_TESTS = 15


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def filter_cavity(r, crossover=1.0, gamma=0.0):
    """One detuned lossless mode at Delta/kappa = 1/2 on an r-squeezed input."""
    H = build_H_bdg(jnp.array([[-0.5 + 0j]]), jnp.zeros((1, 1), dtype=complex))
    kappa_tilde = jnp.array([np.sqrt(2.0) * crossover])
    n, m = squeezed_bath(r)
    sigma_in = channel_covariance(n=jnp.array([n]), m=jnp.array([m]), num_modes=1)
    return H, jnp.array([gamma]), kappa_tilde, sigma_in


def covariance(H, gamma, kappa_tilde, omega, sigma_in, num_ports=1):
    num_modes = H.shape[0] // 2
    return np.asarray(output_covariance_quadrature(
        H, gamma, kappa_tilde, float(omega), sigma_in, jnp.zeros(num_modes), num_ports))


def fit(problem, num_auxiliary_modes, seed=SEED, num_tests=NUM_TESTS):
    """Run the oracle on the fully connected graph; return (optimizer, info)."""
    optimizer = CovarianceArchitectureOptimizer(
        problem.target, num_auxiliary_modes=num_auxiliary_modes, seed=seed,
        make_initial_test=False,
        **problem.optimizer_kwargs(kwargs_optimization={"num_tests": num_tests}))
    success, infos = optimizer.test_graph(optimizer.space.fully_connected())
    info = next((i for i in infos if i["success"]),
                min(infos, key=lambda i: i["loss_reached"]))
    return optimizer, success, info


# --------------------------------------------------------------------------
# B.8 -- forward-map golden values (runnable at M0, no optimiser involved)
# --------------------------------------------------------------------------

def test_b8_filter_cavity_rotates_the_ellipse_without_touching_its_depth():
    """A lossless passive mode may rotate the squeeze ellipse, nothing else."""
    r, crossover = 0.6, 1.0
    H, gamma, kappa_tilde, sigma_in = filter_cavity(r, crossover)
    expected = np.array([np.exp(-2 * r), np.exp(2 * r)])

    for omega in (0.0, 0.3, crossover, 2.0, 40.0):
        V = np.real(covariance(H, gamma, kappa_tilde, omega, sigma_in))
        # pure at every frequency
        assert np.isclose(np.linalg.det(V), 1.0, atol=1e-9)
        # Omega-independent block eigenvalues: the depth never changes
        assert np.allclose(np.sort(np.linalg.eigvalsh(V)), expected, atol=1e-9)


def test_b8_cross_spectrum_is_the_rotation_and_needs_the_full_nambu_block():
    """sigma_xp(Omega): even, zero at both band edges, extremal at Omega_c.

    This nonzero, Omega-dependent cross term is the regression guard that the
    forward map keeps the full 2N x 2N Nambu structure -- collapsing to the
    phase-preserving N x N block loses it entirely.
    """
    r, crossover = 0.6, 1.0
    H, gamma, kappa_tilde, sigma_in = filter_cavity(r, crossover)
    xp = lambda w: np.real(covariance(H, gamma, kappa_tilde, w, sigma_in))[0, 1]

    assert abs(xp(0.0)) < 1e-12                     # theta = 0
    assert abs(xp(500.0)) < 1e-2                    # theta -> pi/2
    assert np.isclose(xp(crossover), np.sinh(2 * r), atol=1e-9)   # theta = pi/4, extremal
    assert np.isclose(xp(0.37), xp(-0.37), atol=1e-12)            # even in Omega
    assert xp(crossover) >= max(xp(w) for w in (0.1, 0.5, 2.0, 8.0))

    # the diagonals swap as theta crosses pi/4
    low = np.real(covariance(H, gamma, kappa_tilde, 0.0, sigma_in))
    high = np.real(covariance(H, gamma, kappa_tilde, 60.0, sigma_in))
    assert np.isclose(low[0, 0], np.exp(-2 * r), atol=1e-9)
    assert np.isclose(low[1, 1], np.exp(2 * r), atol=1e-9)
    assert np.isclose(high[0, 0], np.exp(2 * r), atol=1e-3)
    assert np.isclose(high[1, 1], np.exp(-2 * r), atol=1e-3)


def test_b8_angle_sweeps_zero_to_half_pi_and_the_target_is_realisable():
    """The pinned target values are exactly what the filter cavity emits."""
    r, crossover = 0.6, 1.0
    angles = filter_cavity_angle(np.array([0.0, crossover, 1.0e6]), crossover)
    assert np.isclose(angles[0], 0.0, atol=1e-12)
    assert np.isclose(angles[1], 0.25 * np.pi, atol=1e-12)
    assert np.isclose(angles[2], 0.5 * np.pi, atol=1e-5)
    grid = np.linspace(0.0, 4.0, 17)
    assert np.all(np.diff(filter_cavity_angle(grid, crossover)) > 0)   # monotone in |Omega|

    problem = squeeze_angle_rotation(r=r, crossover=crossover)
    H, gamma, kappa_tilde, sigma_in = filter_cavity(r, crossover)
    for pin in problem.target.pins:
        V = np.real(covariance(H, gamma, kappa_tilde, pin.omega, sigma_in))
        assert np.isclose(V[pin.row, pin.col], float(np.real(pin.value)), atol=1e-9)


def test_b8_fit_recovers_the_detuning_and_the_decay_ratio():
    """Sec. 7.2 in miniature: the fit finds Delta/kappa = 1/2, kappa~ = sqrt2 Omega_c."""
    crossover = 1.0
    problem = squeeze_angle_rotation(r=0.6, crossover=crossover)
    optimizer, success, info = fit(problem, num_auxiliary_modes=0)
    assert success
    assert info["loss_reached"] <= 1.0e-10
    assert info["max_real_eigenvalue"] < 0.0
    parameters = info["parameters"]
    assert np.isclose(abs(parameters["detunings"]["Delta_0/kappa_0"]), 0.5, atol=1e-5)
    assert np.isclose(parameters["decay_ratios"]["kappa~_0"],
                      np.sqrt(2.0) * crossover, rtol=1e-5)
    # passive: no squeezing element is used at all
    assert all(value < 1e-12 for value in parameters["cooperativities"].values())


# --------------------------------------------------------------------------
# conventions shared by B.8 - B.12
# --------------------------------------------------------------------------

def test_symmetrised_spectrum_is_even_so_the_grids_start_at_zero():
    """V(-Omega) = V(Omega)^T -- pinning at -Omega would be redundant, and an
    Omega-asymmetric target would be unphysical."""
    g = jnp.array([[-0.3 + 0j, 0.4 * np.exp(0.7j)], [0.4 * np.exp(-0.7j), 0.15 + 0j]])
    nu = jnp.array([[0.1 * np.exp(1.1j), 0.25 * np.exp(-0.3j)],
                    [0.25 * np.exp(-0.3j), 0.05 + 0j]])
    H = build_H_bdg(g, nu)
    gamma, kappa_tilde = jnp.array([0.05, 0.0]), jnp.array([1.0, 2.3])
    sigma_in = vacuum_covariance(2)
    for omega in (0.3, 1.7):
        plus = covariance(H, gamma, kappa_tilde, omega, sigma_in, num_ports=2)
        minus = covariance(H, gamma, kappa_tilde, -omega, sigma_in, num_ports=2)
        assert np.allclose(minus, plus.T, atol=1e-12)
        assert np.allclose(np.real(np.diag(minus)), np.real(np.diag(plus)), atol=1e-12)

    for name, builder in SPECTRAL_GALLERY.items():
        omegas = builder().target.omegas
        assert omegas.min() >= 0.0, name
        assert len(omegas) > 1, name              # a *grid*, not a single point


def test_decay_ratios_are_live_for_every_spectral_target():
    """The defining Sec. 3 property: kappa~ is inert at Omega = 0 and shapes the
    spectrum off it, so these targets must leave it free."""
    H = build_H_bdg(jnp.array([[-0.2 + 0j]]), jnp.array([[0.3j]]))
    gamma, sigma_in = jnp.array([0.0]), vacuum_covariance(1)
    base = covariance(H, gamma, jnp.array([1.0]), 0.0, sigma_in)
    moved = covariance(H, gamma, jnp.array([2.7]), 0.0, sigma_in)
    assert np.allclose(base, moved, atol=1e-9)
    off_base = covariance(H, gamma, jnp.array([1.0]), 0.6, sigma_in)
    off_moved = covariance(H, gamma, jnp.array([2.7]), 0.6, sigma_in)
    assert not np.allclose(off_base, off_moved, atol=1e-3)

    for name, builder in SPECTRAL_GALLERY.items():
        problem = builder()
        assert problem.suggested.get("free_decay_ratios") is True, name
        optimizer = CovarianceArchitectureOptimizer(
            problem.target, num_auxiliary_modes=0, make_initial_test=False,
            **problem.optimizer_kwargs())
        free = set(optimizer.param.free_indices(optimizer.space.fully_connected()))
        assert set(int(i) for i in optimizer.param.log_kappa_idx) <= free, name


def test_frozen_decay_ratios_cannot_fit_a_spectral_target():
    """AUTOSCATTER's choice (kappa~ fixed hardware) fails on a spectrum -- the
    explicit xfail of Sec. 8.6."""
    problem = bandpass_squeezer()
    kwargs = problem.optimizer_kwargs(kwargs_optimization={"num_tests": 8})
    kwargs["free_decay_ratios"] = False
    optimizer = CovarianceArchitectureOptimizer(
        problem.target, num_auxiliary_modes=1, seed=SEED, make_initial_test=False,
        **kwargs)
    success, infos = optimizer.test_graph(optimizer.space.fully_connected())
    assert not success
    assert min(i["loss_reached"] for i in infos) > 1.0e-8


# --------------------------------------------------------------------------
# B.9 -- bandpass sideband squeezing
# --------------------------------------------------------------------------

def test_b9_bandpass_window_needs_a_second_mode():
    """One resonance is not enough for a Lorentzian window; two are."""
    depth, center, bandwidth = 0.5, 1.0, 0.4
    problem = bandpass_squeezer(depth=depth, center=center, bandwidth=bandwidth)

    _, success_one, info_one = fit(problem, num_auxiliary_modes=0)
    assert not success_one
    assert info_one["loss_reached"] > 1.0e-8

    optimizer, success, info = fit(problem, num_auxiliary_modes=1)
    assert success
    assert info["loss_reached"] <= 1.0e-8
    assert info["max_real_eigenvalue"] < 0.0

    # exact on the pins ...
    pinned = problem.target.omegas
    assert np.allclose(
        [np.real(optimizer.oracle.covariance(info["x"], w))[0, 0] for w in pinned],
        lorentzian_dip(pinned, depth, center, bandwidth), atol=1e-4)

    # ... and the right window between and beyond them: the fit interpolates
    # the shape rather than threading the pinned points
    grid = np.linspace(0.0, 2.6, 261)
    spectrum = np.array([np.real(optimizer.oracle.covariance(info["x"], w))[0, 0]
                         for w in grid])
    assert np.allclose(spectrum, lorentzian_dip(grid, depth, center, bandwidth), atol=1e-2)
    assert np.isclose(grid[int(np.argmin(spectrum))], center, atol=0.05)
    assert np.isclose(spectrum.min(), depth, atol=1e-2)
    half = 0.5 * (1.0 + depth)
    inside = grid[spectrum < half]
    assert np.isclose(0.5 * (inside.max() - inside.min()), bandwidth, atol=0.05)


# --------------------------------------------------------------------------
# B.10 -- flat-top: spectral order costs modes
# --------------------------------------------------------------------------

def test_b10_flat_top_order_two_fits_on_one_mode():
    problem = flat_top_squeezer(order=2)
    optimizer, success, info = fit(problem, num_auxiliary_modes=0)
    assert success
    assert info["loss_reached"] <= 1.0e-8
    grid = np.linspace(0.0, 2.5, 101)
    spectrum = np.array([np.real(optimizer.oracle.covariance(info["x"], w))[0, 0]
                         for w in grid])
    assert np.allclose(spectrum, butterworth_dip(grid, order=2), atol=5e-3)
    # a maximally flat plateau: curvature at the band centre is tiny compared
    # with the roll-off further out
    assert abs(spectrum[1] - spectrum[0]) < 0.1 * abs(spectrum[-1] - spectrum[0])


def test_b10_higher_order_band_edge_costs_a_mode():
    """Order 3 is out of reach one mode short; the extra mode flips it."""
    problem = flat_top_squeezer(order=3)
    _, success_one, info_one = fit(problem, num_auxiliary_modes=0)
    assert not success_one
    assert info_one["loss_reached"] > 1.0e-8
    _, success_two, info_two = fit(problem, num_auxiliary_modes=1)
    assert success_two
    assert info_two["loss_reached"] <= 1.0e-8


# --------------------------------------------------------------------------
# B.11 -- band-limited EPR entanglement
# --------------------------------------------------------------------------

def test_b11_target_is_entangled_only_inside_the_band():
    """The target itself: halved Duan sum < 2 in band, back at 2 outside."""
    depth, center, bandwidth = 0.7, 1.0, 0.4
    assert np.isclose(duan_of_band_limited_epr(center, depth, center, bandwidth),
                      2.0 * depth)
    assert duan_of_band_limited_epr(center + bandwidth, depth, center, bandwidth) < 2.0
    far = duan_of_band_limited_epr(center + 40 * bandwidth, depth, center, bandwidth)
    assert far < 2.0 and np.isclose(far, 2.0, atol=1e-3)


def test_b11_band_limited_epr_needs_an_auxiliary_mode():
    depth, center, bandwidth = 0.7, 1.0, 0.4
    problem = band_limited_epr(depth=depth, center=center, bandwidth=bandwidth)

    _, success_two, info_two = fit(problem, num_auxiliary_modes=0)
    assert not success_two
    assert info_two["loss_reached"] > 1.0e-8

    optimizer, success, info = fit(problem, num_auxiliary_modes=1)
    assert success
    assert info["loss_reached"] <= 1.0e-8
    assert info["max_real_eigenvalue"] < 0.0

    # the emitted field is EPR-certified exactly where the target says
    pinned = problem.target.omegas
    duan = np.array([duan_sum(np.real(optimizer.oracle.covariance(info["x"], w)), 0, 1) / 2.0
                     for w in pinned])
    assert np.allclose(duan, duan_of_band_limited_epr(pinned, depth, center, bandwidth),
                       atol=1e-3)
    in_band = duan[np.abs(pinned - center) <= bandwidth]
    out_of_band = duan[np.abs(pinned - center) >= 3.0 * bandwidth]
    assert in_band.max() < 2.0 - 0.25             # entangled inside the window
    assert out_of_band.min() > 2.0 - 0.1          # essentially separable outside


# --------------------------------------------------------------------------
# B.12 -- notch: a dip needs one resonance, a dip with a hole needs two
# --------------------------------------------------------------------------

def test_b12_notch_costs_the_interfering_partner():
    problem = notch_squeezer()
    for num_auxiliary_modes in (0, 1):
        _, success, info = fit(problem, num_auxiliary_modes=num_auxiliary_modes)
        assert not success
        assert info["loss_reached"] > 1.0e-8

    optimizer, success, info = fit(problem, num_auxiliary_modes=2)
    assert success
    assert info["loss_reached"] <= 1.0e-8
    assert info["max_real_eigenvalue"] < 0.0

    # the hole really is a hole: back at the vacuum floor on the notch line,
    # squeezed on both sides of it
    notch_center, notch_width = 0.8, 0.25
    at_notch = np.real(optimizer.oracle.covariance(info["x"], notch_center))[0, 0]
    below = np.real(optimizer.oracle.covariance(info["x"], notch_center - 3 * notch_width))[0, 0]
    above = np.real(optimizer.oracle.covariance(info["x"], notch_center + 3 * notch_width))[0, 0]
    assert np.isclose(at_notch, 1.0, atol=5e-3)
    assert below < 0.9 and above < 0.9


if __name__ == "__main__":
    failures = 0
    for name, func in sorted(list(globals().items())):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print("PASS  %s" % name)
            except AssertionError as error:
                failures += 1
                print("FAIL  %s: %s" % (name, error))
    print("%s" % ("all spectral-gallery tests passed" if not failures
                  else "%i failures" % failures))
    sys.exit(1 if failures else 0)
