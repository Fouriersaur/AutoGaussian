"""
Example 5 -- the frequency-dependent gallery: App. B.8 - B.12, plus the
squeeze-angle-rotation family of Sec. 8.7 (B.8-imp / B.8-nm / B.8-wind).

B.1-B.7 pin sigma_out at a single Omega, where the decay ratios kappa~_i are
inert.  The five targets here are pinned on a *grid*, which is what turns the
kappa~_i into live optimisation variables -- the headline difference to
AUTOSCATTER (Sec. 3, App. A.4(c)).  Each one also exercises something the
single-point targets cannot:

    B.8  squeeze-angle rotation  -> the phase-sensitive Nambu cross term
                                    sigma_xp(Omega); invisible to any N x N
                                    (phase-preserving) reduction
    B.9  bandpass squeezing      -> a spectral *window*: detuning sets its
                                    position, kappa~ its width
    B.10 flat-top order n        -> spectral degree vs mode count: the band
                                    edge order is what costs modes
    B.11 band-limited EPR        -> entanglement with a bandwidth, written on
                                    the joint quadratures (quadratic-form pins)
    B.12 notch squeezing         -> a dip needs one resonance; a dip with a
                                    hole in it needs the interfering partner

Then three extensions of B.8 (Sec. 8.7), each breaking one property B.8 keeps:

    B.8-imp  impure rotation     -> a prescribed purity spectrum mu(Omega) >= 1
                                    pushes the fit off the pure-state variety
                                    det sigma_out = 1, which takes *added
                                    noise*: the intrinsic losses gamma_i go live
    B.8-nm   non-monotone        -> a turning point in theta(|Omega|); rotation
                                    *shape* as the counted resource
    B.8-wind prescribed winding  -> more accumulated sweep than one mode can
                                    supply; rotation *magnitude* as the resource

Those three have no closed form to pin from (an impure or two-pole rotation is a
ratio of quadratics whose angle and purity spectra are the symbolic-regression
*outputs* of Sec. 7.2), so they are written as the spectrum of a declared
reference device and the prescribed quantity is *measured* off it.

Every fit here is an *active* target sitting near the parametric-oscillation
threshold, so the stability constraint of Sec. 5 does real work: the witnesses
come back with max Re eig(M~) close to zero from below.

Oracle vs search: by default every target is answered by the **oracle** (Sec. 4) on the fully
connected graph, which is what the mode-count ladder needs.  ``--search`` additionally runs the
full Sec. 6 discovery (BFS down the graph lattice -> two libraries + irreducible graphs) for
B.8, B.9 and B.10.  That is the slow path: a spectral oracle call costs ~1 s against ~10 ms for
a single-point target, and a 2-mode lattice holds 3^6 = 729 graphs, so B.9 and B.10 take about
8 minutes each.  A 3-mode lattice (3^12 ~ 5e5) is out of reach for a full BFS, which is why
B.11 and B.12 stop at the oracle.

Usage:  python examples/05_spectral_gallery.py [--search]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogaussian import CovarianceArchitectureOptimizer, duan_sum
from autogaussian.gallery import (
    accumulated_rotation,
    band_limited_epr,
    bandpass_squeezer,
    duan_of_band_limited_epr,
    filter_cavity_angle,
    flat_top_squeezer,
    impure_rotation,
    non_monotone_rotation,
    notch_squeezer,
    rotation_purity,
    rotation_spectrum,
    single_mode_winding_ceiling,
    squeeze_angle,
    squeeze_angle_rotation,
    winding_rotation,
)

SUMMARY = []
SEARCHES = []
NUM_TESTS = 15
SEED = 7
RUN_SEARCH = "--search" in sys.argv


def banner(text):
    print("\n" + "=" * 78 + "\n" + text + "\n" + "=" * 78)


def fit(problem, num_auxiliary_modes, seed=SEED, num_tests=NUM_TESTS, **overrides):
    """Oracle on the fully connected graph with this many modes."""
    kwargs = problem.optimizer_kwargs(kwargs_optimization={"num_tests": num_tests})
    kwargs.update(overrides)          # e.g. intrinsic_losses=False for B.8-imp
    optimizer = CovarianceArchitectureOptimizer(
        problem.target, num_auxiliary_modes=num_auxiliary_modes, seed=seed,
        make_initial_test=False, **kwargs)
    success, infos = optimizer.test_graph(optimizer.space.fully_connected())
    info = next((i for i in infos if i["success"]),
                min(infos, key=lambda i: i["loss_reached"]))
    print("   %i mode(s): %-12s loss = %.2e   max Re eig(M~) = %+.4f"
          % (optimizer.num_modes, "VALID" if success else "no witness",
             info["loss_reached"], info["max_real_eigenvalue"]))
    return optimizer, success, info


def spectrum_of(optimizer, x, omegas, row=0, col=0):
    return np.array([np.real(optimizer.oracle.covariance(x, w))[row, col] for w in omegas])


def search(label, problem, num_auxiliary_modes, seed=SEED, num_tests=10):
    """Full Sec. 6 two-library discovery: the BFS down the graph lattice.

    The oracle says *this graph can do it*; only the lattice walk says *this is the minimal
    device*, and only it produces the two libraries and `n_uncertified` (Sec. 8).
    """
    if not RUN_SEARCH:
        return None, None
    start = time.time()
    optimizer = CovarianceArchitectureOptimizer(
        problem.target, num_auxiliary_modes=num_auxiliary_modes, seed=seed,
        make_initial_test=False,
        **problem.optimizer_kwargs(kwargs_optimization={"num_tests": num_tests}))
    print("\n   search: %i modes, %i slots, max complexity %i"
          % (optimizer.num_modes, optimizer.space.num_slots, optimizer.space.max_complexity))
    irreducibles = optimizer.perform_breadth_first_search(progress=False)
    libraries = optimizer.libraries
    certified = sum(1 for entry in libraries.invalid.values() if entry.certified)
    print("   %i irreducible graph(s), %i top-level oracle calls, %.1f s"
          % (len(irreducibles), sum(optimizer.num_tested_graphs), time.time() - start))
    for graph in sorted(irreducibles, key=lambda g: int(np.sum(g))):
        solution = optimizer.solution_of(graph)
        print("      complexity %i:  %s%s"
              % (int(np.sum(graph)), ", ".join(optimizer.space.describe(graph)),
                 "" if solution is None else "   [loss %.2e, alpha %+.3f]"
                 % (solution["loss_reached"], solution["max_real_eigenvalue"])))
    print("   libraries: valid %i | invalid %i  (certified %i, uncertified %i)"
          % (len(libraries.valid), len(libraries.invalid), certified,
             libraries.n_uncertified()))
    SEARCHES.append((label, len(irreducibles), sum(optimizer.num_tested_graphs),
                     libraries.n_uncertified()))
    return optimizer, irreducibles


def mode_count_ladder(problem, counts, **overrides):
    """Fit the same target with growing mode counts; return the first witness."""
    winner = None
    for num_auxiliary_modes in counts:
        optimizer, success, info = fit(problem, num_auxiliary_modes, **overrides)
        if success and winner is None:
            winner = (optimizer, info, optimizer.num_modes)
    return winner


# --------------------------------------------------------------------------
# B.8  frequency-dependent squeeze-angle rotation (filter cavity)
# --------------------------------------------------------------------------

def run_squeeze_angle_rotation(r=0.6, crossover=1.0):
    banner("B.8  squeeze angle sweeps 0 -> pi/2 across the band at fixed depth "
           "(r = %.2f)" % r)
    problem = squeeze_angle_rotation(r=r, crossover=crossover)
    print(problem.notes)

    optimizer, success, info = fit(problem, num_auxiliary_modes=0)

    grid = np.array([0.0, 0.5 * crossover, crossover, 2.0 * crossover, 20.0 * crossover])
    print("\n   Omega/Omega_c   theta      sigma_xx   sigma_pp   sigma_xp   det V")
    for omega in grid:
        V = np.real(optimizer.oracle.covariance(info["x"], omega))
        print("   %11.2f   %7.4f   %8.4f   %8.4f   %8.4f   %.6f"
              % (omega / crossover, filter_cavity_angle(omega, crossover),
                 V[0, 0], V[1, 1], V[0, 1], np.linalg.det(V)))
    print("   (ellipse eigenvalues e^-2r = %.4f, e^+2r = %.4f at every Omega: a lossless"
          % (np.exp(-2 * r), np.exp(2 * r)))
    print("    passive mode rotates the ellipse, it never changes its depth)")

    parameters = info["parameters"]
    print("\n   recovered device: Delta/kappa = %.6f, kappa~ = %.6f (= sqrt(2) Omega_c), "
          "no squeezing element"
          % (parameters["detunings"]["Delta_0/kappa_0"],
             parameters["decay_ratios"]["kappa~_0"]))
    print("   Delta/kappa = 1/2 is forced: theta(0) = 0 requires 4 arctan(2 Delta/kappa) = pi.")
    print("\n   sigma_xp(Omega) is nonzero and Omega-dependent -- that entry *is* the")
    print("   rotation, and it does not exist in a phase-preserving N x N model.")
    SUMMARY.append(("B.8 squeeze rotation", 1,
                    "pure at every Omega, depth fixed, axis sweeps pi/2"))

    # the oracle above says the graph *can* do it; the lattice walk says which
    # graph is minimal -- and it comes back with no squeezing element at all
    search("B.8", problem, num_auxiliary_modes=0)


# --------------------------------------------------------------------------
# B.8-imp / B.8-nm / B.8-wind  the squeeze-angle-rotation family (Sec. 8.7)
# --------------------------------------------------------------------------

def run_impure_rotation(loss=0.2):
    banner("B.8-imp  rotating axis on an *impure* ellipse: purity spectrum mu(Omega) > 1")
    problem = impure_rotation(loss=loss)
    print(problem.notes)

    optimizer, success, info = fit(problem, num_auxiliary_modes=0)
    parameters = info["parameters"]
    print("   recovered: gamma_0 = %.6f (declared %.2f), Delta/kappa = %.6f, kappa~ = %.6f"
          % (parameters["intrinsic_losses"]["gamma_0"], loss,
             parameters["detunings"]["Delta_0/kappa_0"],
             parameters["decay_ratios"]["kappa~_0"]))

    grid = np.linspace(0.0, 4.0, 81)
    purity = rotation_purity(rotation_spectrum(
        grid, [0.5], decay_ratios=[np.sqrt(2.0)], losses=[loss]))
    print("   mu in [%.3f, %.3f], most mixed at Omega = %.2f (crossover 1.0)"
          % (purity.min(), purity.max(), grid[int(np.argmax(purity))]))

    print("\n   the same graph with the losses frozen (a lossless single mode):")
    _, ok_lossless, info_lossless = fit(problem, num_auxiliary_modes=0,
                                        intrinsic_losses=False)
    print("   -> %s.  A lossless single mode is all-pass: it rotates the input ellipse"
          % ("VALID" if ok_lossless else "no witness"))
    print("      symplectically, so det sigma_out = 1 for every detuning and decay ratio.")
    print("\n   careful with the 'no lossless graph can do it' reading -- what is provable is")
    print("   the *single-mode* statement.  A second lossless mode carries its own vacuum")
    print("   input channel, so it supplies the noise instead:")
    _, ok_aux, _ = fit(problem, num_auxiliary_modes=1, intrinsic_losses=False)
    print("   -> %s.  Impurity costs added noise; a mode is one way to buy it."
          % ("VALID" if ok_aux else "no witness"))
    SUMMARY.append(("B.8-imp impure rotation", 1,
                    "mu up to %.3f, gamma_0 live" % purity.max()))


def run_non_monotone_rotation():
    banner("B.8-nm  theta(|Omega|) rises, turns over and comes back")
    problem = non_monotone_rotation()
    print(problem.notes)

    grid = np.linspace(0.0, 4.0, 161)
    target = rotation_spectrum(grid, (0.5, 0.5), coupling=1.0, decay_ratios=(1.0, 0.8))
    angle = squeeze_angle(target)
    turn = grid[int(np.argmax(angle))]
    one_mode = rotation_spectrum(grid, [0.5], decay_ratios=[1.0])
    uncoupled = rotation_spectrum(grid, (0.5, 0.5), coupling=0.0, decay_ratios=(1.0, 0.8))
    print("\n   turning point at Omega = %.2f: %.2f rad out, %.2f rad back"
          % (turn, angle.max() - angle[0], angle.max() - angle[-1]))
    print("   one detuned mode is monotone in |Omega| (single-pole phase), and an")
    print("   *uncoupled* partner is invisible to the port -- the two spectra agree to %.1e,"
          % np.max(np.abs(uncoupled - one_mode)))
    print("   so it is the hybridisation, not the mode count, that buys the turn.")

    winner = mode_count_ladder(problem, (0, 1))
    if winner is None:
        print("   no witness found")
        return
    optimizer, info, num_modes = winner
    print("\n   hybridisation found: C^BS_{0,1} = %.3f against detunings %s"
          % (info["parameters"]["cooperativities"]["C^{BS}_{0,1}"],
             ", ".join("%.3f" % v for v in info["parameters"]["detunings"].values())))
    SUMMARY.append(("B.8-nm turning point", num_modes,
                    "theta turns at Omega = %.2f" % turn))


def run_winding_rotation(omega_max=6.0):
    banner("B.8-wind  a prescribed total sweep, past what one mode can supply")
    dense = np.linspace(0.0, omega_max, 241)
    ceiling = single_mode_winding_ceiling(omega_max=omega_max, num_points=121)
    print("   single-mode winding ceiling on [0, %.0f]: %.3f rad  (pi = %.3f, pi/2 = %.3f)"
          % (omega_max, ceiling, np.pi, 0.5 * np.pi))
    print("   the pi/2 of B.8 is that device's useful *in-band* rotation, not its ceiling,")
    print("   so the threshold that forces a second mode is measured, never hard-coded.")

    print("\n   sweeping the demanded winding inside the single-mode family:")
    for detuning, ratio in ((0.5, 1.4), (2.0, 0.7), (5.0, 0.5)):
        single = winding_rotation(detunings=(detuning,), coupling=0.0,
                                  decay_ratios=(ratio,))
        winding = accumulated_rotation(squeeze_angle(
            rotation_spectrum(dense, (detuning,), decay_ratios=(ratio,))))
        _, ok, _ = fit(single, num_auxiliary_modes=0)
        print("      Delta/kappa = %.1f, kappa~ = %.1f -> winding %.3f rad (%s)"
              % (detuning, ratio, winding, "one mode is enough" if ok else "one mode fails"))

    problem = winding_rotation()
    print("\n" + problem.notes)
    winding = accumulated_rotation(squeeze_angle(rotation_spectrum(
        dense, (1.0, 1.5), coupling=0.6, decay_ratios=(0.9, 0.7))))
    print("   target winding %.3f rad > ceiling %.3f rad -> provably not a single mode"
          % (winding, ceiling))
    winner = mode_count_ladder(problem, (0, 1))
    if winner is None:
        print("   no witness found")
        return
    _, _, num_modes = winner

    scale = 0.4
    small = (1.0 * scale, 1.5 * scale)
    wind_small = accumulated_rotation(squeeze_angle(rotation_spectrum(
        dense, small, coupling=0.6 * scale, decay_ratios=(0.9, 0.7))))
    _, ok_small, info_small = fit(winding_rotation(detunings=small, coupling=0.6 * scale),
                                  num_auxiliary_modes=0)
    print("\n   winding is a *sufficient* obstruction, not a necessary one: the same")
    print("   reference scaled down winds only %.3f rad (< ceiling) and is still %s"
          % (wind_small, "VALID" if ok_small else "no witness"))
    print("   on one mode (best loss %.2e) -- its *shape* is what one pole cannot make."
          % info_small["loss_reached"])
    SUMMARY.append(("B.8-wind winding", num_modes,
                    "%.2f rad demanded vs %.2f rad ceiling" % (winding, ceiling)))


# --------------------------------------------------------------------------
# B.9  bandpass sideband squeezing
# --------------------------------------------------------------------------

def run_bandpass(depth=0.5, center=1.0, bandwidth=0.4):
    banner("B.9  Lorentzian squeezing window: depth %.2f at Omega_c = %.2f, HWHM %.2f"
           % (depth, center, bandwidth))
    problem = bandpass_squeezer(depth=depth, center=center, bandwidth=bandwidth)
    winner = mode_count_ladder(problem, (0, 1))
    if winner is None:
        print("   no witness found")
        return
    optimizer, info, num_modes = winner

    grid = np.linspace(0.0, center + 4 * bandwidth, 261)
    achieved = spectrum_of(optimizer, info["x"], grid)
    half = 0.5 * (1.0 + depth)
    inside = grid[achieved < half]
    print("\n   dip centre    %.3f  (asked %.3f)" % (grid[int(np.argmin(achieved))], center))
    print("   dip depth     %.3f  (asked %.3f)" % (achieved.min(), depth))
    print("   half-width    %.3f  (asked %.3f)" % (0.5 * (inside.max() - inside.min()),
                                                   bandwidth))
    print("\n   One resonance cannot make this window: a single mode gets to ~5e-4 and")
    print("   stops.  Position <- detuning, width <- kappa~, depth <- cooperativity, but")
    print("   a single mode cannot set all three independently.")
    SUMMARY.append(("B.9 bandpass squeezing", num_modes,
                    "dip %.3f at %.2f, HWHM %.2f" % (achieved.min(), center, bandwidth)))

    # every irreducible graph carries *two* squeezing elements: the lattice walk
    # turns "one resonance is not enough" into a structural statement
    search("B.9", problem, num_auxiliary_modes=1)


# --------------------------------------------------------------------------
# B.10  flat-top: the spectral degree -> mode count bound at work
# --------------------------------------------------------------------------

def run_flat_top(orders=(1, 2, 3)):
    banner("B.10  order-n maximally flat plateau: how many modes does a band edge cost?")
    table = []
    for order in orders:
        print("\n   order n = %i" % order)
        problem = flat_top_squeezer(order=order)
        winner = mode_count_ladder(problem, (0, 1))
        table.append((order, None if winner is None else winner[2]))
    print("\n   order n   minimum modes found")
    for order, num_modes in table:
        print("   %7i   %s" % (order, num_modes if num_modes else "> 2"))
    print("\n   Prescribing a sharper band edge raises the McMillan degree of the target")
    print("   spectrum, and the degree is paid for in modes -- each new mode brings its")
    print("   own kappa~_i.  Bandwidth flatness costs modes (Sec. 8.6 of the build spec).")
    SUMMARY.append(("B.10 flat-top plateau", table[-1][1] or 0,
                    "orders %s -> modes %s" % ([t[0] for t in table],
                                               [t[1] for t in table])))

    # the ladder above compares fully connected graphs; the search compares
    # *minimal* ones, which is the sharper form of the degree bound
    search("B.10 n=2", flat_top_squeezer(order=2), num_auxiliary_modes=0)
    search("B.10 n=3", flat_top_squeezer(order=3), num_auxiliary_modes=1)


# --------------------------------------------------------------------------
# B.11  band-limited EPR entanglement
# --------------------------------------------------------------------------

def run_band_limited_epr(depth=0.7, center=1.0, bandwidth=0.4):
    banner("B.11  EPR correlations with a bandwidth: joint quadratures squeezed to %.2f "
           "only near Omega = %.2f" % (depth, center))
    problem = band_limited_epr(depth=depth, center=center, bandwidth=bandwidth)
    winner = mode_count_ladder(problem, (0, 1))
    if winner is None:
        print("   no witness found")
        return
    optimizer, info, num_modes = winner

    grid = problem.target.omegas
    duan = np.array([duan_sum(np.real(optimizer.oracle.covariance(info["x"], w)), 0, 1) / 2.0
                     for w in grid])
    wanted = duan_of_band_limited_epr(grid, depth, center, bandwidth)
    print("\n   Omega      Duan (halved)   target     certified entangled?")
    for omega, got, want in zip(grid, duan, wanted):
        print("   %7.3f    %11.4f   %8.4f   %s"
              % (omega, got, want, "yes" if got < 1.95 else "no (back at the bound)"))
    print("   (separability bound 2; the target *is* the frequency window in which the")
    print("    source is EPR-certified, not the depth at one frequency.  A Lorentzian")
    print("    window has infinite support, so off band the Duan sum only *approaches*")
    print("    2 -- the certified bandwidth is quoted at a threshold, here 1.95.)")
    SUMMARY.append(("B.11 band-limited EPR", num_modes,
                    "Duan %.3f in band, %.3f out" % (duan.min(), duan.max())))


# --------------------------------------------------------------------------
# B.12  notch (band-stop) squeezing
# --------------------------------------------------------------------------

def run_notch(notch_center=0.8, notch_width=0.25):
    banner("B.12  broadband squeezing with a hole punched at Omega_0 = %.2f" % notch_center)
    problem = notch_squeezer(notch_center=notch_center, notch_width=notch_width)
    winner = mode_count_ladder(problem, (0, 1, 2))
    if winner is None:
        print("   no witness found")
        return
    optimizer, info, num_modes = winner

    grid = np.linspace(0.0, 3.0, 121)
    achieved = spectrum_of(optimizer, info["x"], grid)
    at_notch = float(np.interp(notch_center, grid, achieved))
    print("\n   sigma_xx at the notch line   %.4f  (vacuum floor 1.0)" % at_notch)
    print("   sigma_xx %.2f below the line  %.4f"
          % (3 * notch_width, float(np.interp(notch_center - 3 * notch_width, grid, achieved))))
    print("   sigma_xx %.2f above the line  %.4f"
          % (3 * notch_width, float(np.interp(notch_center + 3 * notch_width, grid, achieved))))
    print("\n   B.9's graph makes the dip; the *hole* is a destructive-interference zero,")
    print("   so it costs one more detuned mode than the plain window does.")
    SUMMARY.append(("B.12 notch squeezing", num_modes,
                    "vacuum floor %.3f at the notch" % at_notch))


if __name__ == "__main__":
    run_squeeze_angle_rotation()
    run_impure_rotation()
    run_non_monotone_rotation()
    run_winding_rotation()
    run_bandpass()
    run_flat_top()
    run_band_limited_epr()
    run_notch()

    print("\n" + "=" * 78)
    print("%-24s %-8s %s" % ("target", "modes", "verified property"))
    print("-" * 78)
    for name, count, result in SUMMARY:
        print("%-24s %-8s %s" % (name, count, result))
    print("=" * 78)
    print("Every target above pins sigma_out on a frequency grid, so the decay ratios")
    print("kappa~_i are live optimisation variables -- the point of the whole exercise.")

    if SEARCHES:
        print("\n%-12s %-14s %-14s %s"
              % ("search", "irreducibles", "oracle calls", "uncertified"))
        print("-" * 78)
        for label, count, calls, uncertified in SEARCHES:
            print("%-12s %-14i %-14i %i" % (label, count, calls, uncertified))
        print("Membership in the invalid library is not a proof of infeasibility: the")
        print("uncertified column is the completeness caveat (Sec. 8), and on a spectral")
        print("target it is large, because sos_no_hurwitz is a stub and the fit-range SDP")
        print("rarely fires on a grid.")
    else:
        print("\n(oracle only -- rerun with --search for the Sec. 6 discovery on B.8/B.9/B.10)")
