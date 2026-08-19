"""The squeeze-angle-rotation family -- Sec. 8.7, three extensions of App. B.8.

B.8 asks for a rotating squeeze axis at *fixed* depth on a *pure* state, and one
lossless detuned mode delivers it.  Each target here breaks one of those
properties and asks what the break costs:

* **B.8-imp** -- same rotating axis, prescribed purity spectrum ``mu(Omega) >= 1``:
  pushes the fit **off** the pure-state variety, which needs *added noise*;
* **B.8-nm**  -- ``theta(|Omega|)`` with a turning point: rotation **shape** as
  the counted resource;
* **B.8-wind** -- prescribed accumulated sweep past what one mode supplies:
  rotation **magnitude** as the counted resource.

The last two are the rotation-family analogues of B.10 ("spectral structure
costs modes"), so they are asserted the same way: a *feasibility transition*
(one mode short -> ``L > tol`` on every restart, a provisional non-condemning
INVALID; add the mode -> VALID), never an exact closed-form count.

Two findings here differ from the Sec. 8.7 write-up and are asserted as
measured, not as written:

* the single-mode winding ceiling is ``~pi``, not ``pi/2`` (Sec. 8.7 says as
  much, and the tests measure it rather than hard-coding either number);
* the turning point of B.8-nm is bought by *hybridisation*, not by the sign of
  the bare detunings, and impurity can be bought either with an intrinsic-loss
  channel **or** with an auxiliary mode's vacuum port.  Both are tested.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogaussian import CovarianceArchitectureOptimizer
from autogaussian.gallery import (
    ROTATION_FAMILY,
    SPECTRAL_GALLERY,
    accumulated_rotation,
    impure_rotation,
    non_monotone_rotation,
    rotation_purity,
    rotation_spectrum,
    single_mode_winding_ceiling,
    squeeze_angle,
    winding_rotation,
)

SEED = 7
NUM_TESTS = 15
TOL = 1.0e-8
DENSE = np.linspace(0.0, 6.0, 241)


def fit(problem, num_auxiliary_modes, seed=SEED, num_tests=NUM_TESTS, **overrides):
    """Run the oracle on the fully connected graph; return ``(success, info)``."""
    kwargs = problem.optimizer_kwargs(kwargs_optimization={"num_tests": num_tests})
    kwargs.update(overrides)
    optimizer = CovarianceArchitectureOptimizer(
        problem.target, num_auxiliary_modes=num_auxiliary_modes, seed=seed,
        make_initial_test=False, **kwargs)
    success, infos = optimizer.test_graph(optimizer.space.fully_connected())
    info = next((i for i in infos if i["success"]),
                min(infos, key=lambda i: i["loss_reached"]))
    return optimizer, success, info


def is_monotone(angle, tolerance=3.0e-3):
    steps = np.diff(np.asarray(angle, dtype=float))
    return not (np.any(steps > tolerance) and np.any(steps < -tolerance))


# --------------------------------------------------------------------------
# B.8-imp -- impure rotation: the purity spectrum forces added noise
# --------------------------------------------------------------------------

def test_b8imp_target_is_impure_and_most_mixed_near_the_crossover():
    """``mu(Omega) = sqrt(det V) > 1`` everywhere in band, peaking near Omega_c.

    That is the whole content of the target: B.8 sits on ``det V = 1``, this one
    is deliberately off it, so the axis fit and the depth/purity fit are coupled.
    """
    crossover = 1.0
    problem = impure_rotation(crossover=crossover, loss=0.2)
    grid = problem.target.omegas
    values = rotation_spectrum(grid, [0.5], decay_ratios=[np.sqrt(2.0) * crossover],
                               losses=[0.2])
    purity = rotation_purity(values)

    assert np.all(purity > 1.0 + 1.0e-3)                 # nowhere pure
    peak = grid[int(np.argmax(purity))]
    assert abs(peak - crossover) <= 0.5                  # most mixed at the crossover
    assert purity[-1] < purity.max()                     # pure again far off resonance

    # the pins really are that spectrum
    for pin in problem.target.pins:
        index = int(np.argmin(np.abs(grid - pin.omega)))
        assert np.isclose(values[index][pin.row, pin.col],
                          float(np.real(pin.value)), atol=1e-12)


def test_b8imp_a_lossless_single_mode_can_never_leave_the_pure_variety():
    """Structural, no optimiser: a lossless passive single mode is all-pass, so
    it rotates the input ellipse symplectically and ``det V = 1`` for *every*
    detuning and decay ratio.  No amount of tuning fits ``mu > 1``."""
    for detuning in (-3.0, -0.5, 0.0, 0.5, 2.0, 7.0):
        for ratio in (0.2, 1.0, np.sqrt(2.0), 5.0):
            values = rotation_spectrum(DENSE, [detuning], decay_ratios=[ratio])
            assert np.allclose(rotation_purity(values), 1.0, atol=1e-9)


def test_b8imp_fit_needs_the_intrinsic_loss_live():
    """``gamma_i`` in ``x``: the fit recovers the declared loss, and freezing the
    losses (a lossless single-mode build) cannot reach tolerance."""
    problem = impure_rotation(loss=0.2)

    _, success, info = fit(problem, num_auxiliary_modes=0)
    assert success
    assert info["loss_reached"] <= TOL
    assert info["max_real_eigenvalue"] < 0.0
    parameters = info["parameters"]
    assert np.isclose(parameters["intrinsic_losses"]["gamma_0"], 0.2, atol=1e-4)
    assert np.isclose(parameters["detunings"]["Delta_0/kappa_0"], 0.5, atol=1e-4)
    assert np.isclose(parameters["decay_ratios"]["kappa~_0"], np.sqrt(2.0), rtol=1e-4)
    # still a purely passive device: the squeezing is all in the declared input
    assert all(value < 1e-9 for value in parameters["cooperativities"].values())

    _, success_lossless, info_lossless = fit(problem, num_auxiliary_modes=0,
                                             intrinsic_losses=False)
    assert not success_lossless
    assert info_lossless["loss_reached"] > TOL


def test_b8imp_an_auxiliary_mode_supplies_the_noise_instead():
    """The honest version of the "lossless graphs cannot do it" claim.

    What is provable is the *single-mode* statement above.  A second, still
    lossless, mode is not noiseless from the port's point of view -- it carries
    its own vacuum input channel -- so a two-mode lossless graph does fit the
    impure target.  Impurity costs *added noise*, and a mode is one way to buy
    it; that is a mode-count statement, not a "no lossless graph" statement.
    """
    problem = impure_rotation(loss=0.2)
    _, success, info = fit(problem, num_auxiliary_modes=1, intrinsic_losses=False)
    assert success
    assert info["loss_reached"] <= TOL
    assert info["max_real_eigenvalue"] < 0.0


# --------------------------------------------------------------------------
# B.8-nm -- non-monotone rotation: a turning point costs the coupled partner
# --------------------------------------------------------------------------

def test_b8nm_one_mode_can_never_turn_around():
    """Single-pole phase is monotone in ``|Omega|`` -- swept over the whole
    single-mode family, not argued."""
    for detuning in np.linspace(-6.0, 6.0, 25):
        for ratio in np.exp(np.linspace(-1.5, 1.5, 7)):
            angle = squeeze_angle(rotation_spectrum(DENSE, [detuning],
                                                    decay_ratios=[ratio]))
            assert is_monotone(angle), (detuning, ratio)


def test_b8nm_an_uncoupled_partner_is_invisible():
    """The trap a mode-*counting* test would fall into: switching the coupling
    off leaves the dynamical matrix block diagonal, so the second mode does not
    reach the monitored port at all and the angle is exactly the one-mode one."""
    single = rotation_spectrum(DENSE, [0.5], decay_ratios=[1.0])
    pair = rotation_spectrum(DENSE, (0.5, 0.5), coupling=0.0, decay_ratios=(1.0, 0.8))
    assert np.allclose(single, pair, atol=1e-12)
    assert is_monotone(squeeze_angle(pair))


def test_b8nm_target_turns_over_inside_the_band():
    problem = non_monotone_rotation()
    dense = np.linspace(0.0, problem.target.omegas.max(), 241)
    angle = squeeze_angle(rotation_spectrum(dense, (0.5, 0.5), coupling=1.0,
                                            decay_ratios=(1.0, 0.8)))
    assert not is_monotone(angle)
    turn = int(np.argmax(angle))
    assert 0 < turn < len(dense) - 1                       # interior extremum
    assert angle[turn] - angle[0] > 0.5                    # a real excursion out
    assert angle[turn] - angle[-1] > 0.5                   # ... and back again

    # the coarse pinned grid resolves the turn, so the fit sees it
    pinned = squeeze_angle(rotation_spectrum(problem.target.omegas, (0.5, 0.5),
                                             coupling=1.0, decay_ratios=(1.0, 0.8)))
    assert not is_monotone(pinned)


def test_b8nm_the_turning_point_costs_the_coupled_partner():
    """The feasibility transition: one mode short the fit never reaches
    tolerance on any restart; adding the coupled mode flips it VALID."""
    problem = non_monotone_rotation()

    _, success_one, info_one = fit(problem, num_auxiliary_modes=0)
    assert not success_one
    assert info_one["loss_reached"] > TOL

    optimizer, success, info = fit(problem, num_auxiliary_modes=1)
    assert success
    assert info["loss_reached"] <= TOL
    assert info["max_real_eigenvalue"] < 0.0

    parameters = info["parameters"]
    # what it found: two detuned modes, strongly hybridised, no squeezing element
    assert np.isclose(parameters["cooperativities"]["C^{BS}_{0,1}"], 4.0, rtol=1e-3)
    assert all(abs(value) > 0.4 for value in parameters["detunings"].values())
    squeezing = [value for key, value in parameters["cooperativities"].items()
                 if key != "C^{BS}_{0,1}"]
    assert all(value < 1e-4 for value in squeezing)

    # and it reproduces the turn, not only the pins
    dense = np.linspace(0.0, problem.target.omegas.max(), 121)
    achieved = np.array([np.real(optimizer.oracle.covariance(info["x"], w))
                         for w in dense])
    assert not is_monotone(squeeze_angle(achieved))


# --------------------------------------------------------------------------
# B.8-wind -- prescribed total rotation: winding as a counted resource
# --------------------------------------------------------------------------

def test_b8wind_single_mode_ceiling_is_about_pi_and_is_measured_not_assumed():
    """Sec. 8.7's caveat, executable: one detuned mode does **not** stop at
    ``pi/2``.  Scanning the single-mode family gives a ceiling of ``~pi``, so the
    winding that forces a second mode is band-dependent and has to be measured.
    """
    ceiling = single_mode_winding_ceiling(omega_max=6.0, num_points=121)
    assert ceiling > 0.5 * np.pi + 0.5                     # not pi/2
    assert ceiling < np.pi + 1.0e-2                        # one pole turns by pi
    assert np.isclose(ceiling, np.pi, atol=0.2)


def test_b8wind_target_winds_past_the_single_mode_ceiling():
    problem = winding_rotation()
    grid = np.linspace(0.0, problem.target.omegas.max(), 241)
    winding = accumulated_rotation(squeeze_angle(rotation_spectrum(
        grid, (1.0, 1.5), coupling=0.6, decay_ratios=(0.9, 0.7))))
    assert winding > np.pi
    assert winding > single_mode_winding_ceiling(omega_max=grid.max(), num_points=121)


def test_b8wind_the_threshold_is_found_by_sweeping_the_prescribed_winding():
    """Sweep the demanded sweep upward and watch the one-mode fit break.

    Inside the single-mode family the winding grows with the detuning and the
    fit keeps succeeding, right up to the measured ceiling; a target that winds
    *past* the ceiling cannot be a single mode at all, because the ceiling is
    the maximum over that whole family.  The transition is located, not
    hard-coded -- which is the point of Sec. 8.7's caveat on the budget.
    """
    ceiling = single_mode_winding_ceiling(omega_max=6.0, num_points=121)
    windings = []
    for detuning, ratio in ((0.5, 1.4), (2.0, 0.7), (5.0, 0.5)):
        problem = winding_rotation(detunings=(detuning,), coupling=0.0,
                                   decay_ratios=(ratio,))
        winding = accumulated_rotation(squeeze_angle(rotation_spectrum(
            np.linspace(0.0, 6.0, 241), (detuning,), decay_ratios=(ratio,))))
        _, success, _ = fit(problem, num_auxiliary_modes=0)
        assert winding < ceiling + 1.0e-9
        assert success                                     # under the ceiling: one mode
        windings.append(winding)

    assert windings == sorted(windings)                    # more detuning, more winding

    for scale in (1.0, 1.5):
        detunings = (1.0 * scale, 1.5 * scale)
        problem = winding_rotation(detunings=detunings, coupling=0.6 * scale)
        winding = accumulated_rotation(squeeze_angle(rotation_spectrum(
            np.linspace(0.0, 6.0, 241), detunings, coupling=0.6 * scale,
            decay_ratios=(0.9, 0.7))))
        _, success, info = fit(problem, num_auxiliary_modes=0)
        assert winding > ceiling > max(windings)
        assert not success and info["loss_reached"] > TOL   # over it: not one mode


def test_b8wind_winding_is_a_sufficient_obstruction_not_a_necessary_one():
    """Guard against reading the ceiling backwards.

    Winding past the ceiling *proves* a second mode is needed.  The converse is
    false and is asserted so nobody builds a "winding budget" out of it: this
    two-mode target winds only ~1.2 rad, far under the ceiling, and still does
    not fit on one mode -- its *shape* is what one pole cannot make.
    """
    scale = 0.4
    detunings = (1.0 * scale, 1.5 * scale)
    problem = winding_rotation(detunings=detunings, coupling=0.6 * scale)
    winding = accumulated_rotation(squeeze_angle(rotation_spectrum(
        np.linspace(0.0, 6.0, 241), detunings, coupling=0.6 * scale,
        decay_ratios=(0.9, 0.7))))
    assert winding < single_mode_winding_ceiling(omega_max=6.0, num_points=121)
    _, success, info = fit(problem, num_auxiliary_modes=0)
    assert not success and info["loss_reached"] > TOL


def test_b8wind_the_extra_winding_costs_a_mode():
    problem = winding_rotation()
    _, success_one, info_one = fit(problem, num_auxiliary_modes=0)
    assert not success_one
    assert info_one["loss_reached"] > TOL

    _, success, info = fit(problem, num_auxiliary_modes=1)
    assert success
    assert info["loss_reached"] <= TOL
    assert info["max_real_eigenvalue"] < 0.0


# --------------------------------------------------------------------------
# cross-cutting (Sec. 8.6 conventions inherited by the whole family)
# --------------------------------------------------------------------------

def test_rotation_family_is_registered_and_spectral():
    """All three ship as gallery entries alongside B.8 and inherit its grid /
    live-``kappa~`` / full-Nambu setup."""
    for key in ("B.8", "B.8-imp", "B.8-nm", "B.8-wind"):
        assert key in ROTATION_FAMILY
        assert key in SPECTRAL_GALLERY

    for key, builder in ROTATION_FAMILY.items():
        problem = builder()
        omegas = problem.target.omegas
        assert omegas.min() >= 0.0 and len(omegas) > 1, key
        assert problem.suggested["free_decay_ratios"] is True, key
        # the cross term sigma_xp is pinned: the family is invisible to any
        # phase-preserving (N x N) reduction
        assert any(pin.row == 0 and pin.col == 1 for pin in problem.target.pins), key
        optimizer = CovarianceArchitectureOptimizer(
            problem.target, num_auxiliary_modes=0, make_initial_test=False,
            **problem.optimizer_kwargs())
        free = set(optimizer.param.free_indices(optimizer.space.fully_connected()))
        assert set(int(i) for i in optimizer.param.log_kappa_idx) <= free, key


def test_rotation_family_cross_term_is_even_in_omega():
    """Inherited from B.8: ``V(-Omega) = V(Omega)^T``, so the grids run over
    ``Omega >= 0`` and ``sigma_xp`` is even."""
    for detunings, coupling in (((0.5,), 0.0), ((0.5, 0.5), 1.0)):
        for omega in (0.37, 1.9):
            plus = rotation_spectrum([omega], detunings, coupling=coupling)[0]
            minus = rotation_spectrum([-omega], detunings, coupling=coupling)[0]
            assert np.allclose(minus, plus.T, atol=1e-12)


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
    print("%s" % ("all rotation-family tests passed" if not failures
                  else "%i failures" % failures))
    sys.exit(1 if failures else 0)
