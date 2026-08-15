"""
Example 4 -- the rest of the App. B gallery: B.5, B.6, B.7.

These three are not just extra targets; each exercises a part of the machinery
that nothing else in the examples touches:

    B.5  CV graph-state emitter     -> quadratic-form pins (``pin_form``): a
                                       nullifier variance ``Var(p_i - x_j)`` is
                                       not a single entry of sigma_out
    B.6  backaction-evading readout -> a single pinned cross-term, everything
                                       else free
    B.7  noise diode                -> a *declared non-vacuum* input covariance;
                                       the directionality is visible only
                                       because sigma_in != 1

Each runs the complete Sec. 6 search and then checks, on the cheapest device
found, that the emitted field really has the pinned property.

Usage:  python examples/04_gallery_sweep.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogaussian import CovarianceArchitectureOptimizer, qidx
from autogaussian.gallery import (
    backaction_evading_readout,
    cv_graph_state,
    noise_diode,
)

SUMMARY = []
KWARGS_OPTIMIZATION = {"num_tests": 15}


def banner(text):
    print("\n" + "=" * 74 + "\n" + text + "\n" + "=" * 74)


def search(problem, num_auxiliary_modes=0, seed=11):
    optimizer = CovarianceArchitectureOptimizer(
        problem.target, num_auxiliary_modes=num_auxiliary_modes, seed=seed,
        make_initial_test=False, kwargs_optimization=KWARGS_OPTIMIZATION,
        **problem.optimizer_kwargs())
    irreducibles = optimizer.perform_breadth_first_search(progress=False)
    return optimizer, irreducibles


def show(optimizer, irreducibles, limit=6):
    order = np.argsort([int(np.sum(graph)) for graph in irreducibles])
    print("\n%i irreducible graphs (%i oracle calls):"
          % (len(irreducibles), sum(optimizer.num_tested_graphs)))
    for rank, idx in enumerate(order):
        if rank == limit:
            print("   ... and %i more" % (len(order) - limit))
            break
        graph = irreducibles[idx]
        print("   complexity %i:  %s"
              % (int(np.sum(graph)), ", ".join(optimizer.space.describe(graph))))
    return irreducibles[order[0]]


def cheapest_solution(optimizer, graph):
    solution = optimizer.solution_of(graph)
    if solution is None:                       # re-solve if it was not cached
        success, infos = optimizer.test_graph(graph, num_tests=30)
        solution = infos[-1]
    return solution


# --------------------------------------------------------------------------
# B.5  CV graph-state emitter
# --------------------------------------------------------------------------

def run_cv_graph_state(epsilon=0.1):
    banner("B.5  CV graph state:  both nullifiers of the 2-node graph = %.2f" % epsilon)
    problem = cv_graph_state(edges=((0, 1),), epsilon=epsilon, num_ports=2)
    print(problem.target)
    optimizer, irreducibles = search(problem)
    best = show(optimizer, irreducibles)

    first = np.zeros(4)
    first[qidx(0, 1)], first[qidx(1, 0)] = 1.0, -1.0        # p_0 - x_1
    second = np.zeros(4)
    second[qidx(1, 1)], second[qidx(0, 0)] = 1.0, -1.0      # p_1 - x_0

    print("\n   every irreducible device, both nullifiers:")
    for graph in irreducibles:
        solution = optimizer.solution_of(graph)
        if solution is None:
            continue
        V = np.real(optimizer.oracle.covariance(solution["x"], 0.0))
        print("      %-62s %.4f / %.4f   margin %.3f"
              % (", ".join(optimizer.space.describe(graph)),
                 first @ V @ first, second @ V @ second,
                 solution["max_real_eigenvalue"]))
    print("      (vacuum would give 2.0 for each; stability margin must be < 0)")

    solution = cheapest_solution(optimizer, best)
    V = np.real(optimizer.oracle.covariance(solution["x"], 0.0))
    achieved = float(first @ V @ first)
    print("\n   NOTE: pinning only *one* nullifier admits 12 irreducible devices, several")
    print("   of which are two **decoupled** squeezers -- not a cluster state at all")
    print("   (Sec. 11).  One nullifier per node, as a CV graph state is actually")
    print("   defined, cuts that to %i and every survivor couples the modes." % len(irreducibles))
    SUMMARY.append(("B.5 CV graph state", len(irreducibles),
                    "both nullifiers = %.4f" % achieved))


# --------------------------------------------------------------------------
# B.6  backaction-evading readout
# --------------------------------------------------------------------------

def run_backaction_evading():
    banner("B.6  backaction-evading readout:  sigma_{p_0, x_1} = 0, rest free")
    problem = backaction_evading_readout(signal_port=0, probe_port=1)
    optimizer, irreducibles = search(problem)
    best = show(optimizer, irreducibles)

    solution = cheapest_solution(optimizer, best)
    V = np.real(optimizer.oracle.covariance(solution["x"], 0.0))
    S, _ = optimizer.oracle.scattering(solution["x"], 0.0)
    cross = V[qidx(0, 1), qidx(1, 0)]
    print("\n   cheapest device: %s" % ", ".join(optimizer.space.describe(best)))
    print("   sigma_{p_0,x_1} = %.2e   (pinned to 0)" % cross)
    print("   |S_10|^2 = %.4f      (transmission floor 0.25 keeps the probe coupled)"
          % abs(S[1, 0]) ** 2)
    print("\n   CAVEAT: with vacuum input a purely passive device emits vacuum, so every")
    print("   cross-term is zero and this target is satisfied trivially -- which is why")
    print("   the cheapest solution is a bare beam-splitter.  App. B.6 as written is")
    print("   under-specified (Sec. 11): a real backaction-evading spec also has to pin")
    print("   the signal transduction, so that the cross-term vanishing means something.")
    SUMMARY.append(("B.6 backaction evading", len(irreducibles),
                    "cross-term %.1e, |S_10|^2 = %.2f" % (cross, abs(S[1, 0]) ** 2)))


# --------------------------------------------------------------------------
# B.7  noise diode
# --------------------------------------------------------------------------

def run_noise_diode(n_thermal=1.0):
    banner("B.7  noise diode:  thermal input n=%.1f at port 2, port 1 pinned to vacuum"
           % n_thermal)
    problem = noise_diode(n_thermal=n_thermal)
    optimizer, irreducibles = search(problem)
    best = show(optimizer, irreducibles)

    solution = cheapest_solution(optimizer, best)
    V = np.real(optimizer.oracle.covariance(solution["x"], 0.0))
    S, _ = optimizer.oracle.scattering(solution["x"], 0.0)
    print("\n   cheapest device: %s" % ", ".join(optimizer.space.describe(best)))
    print("   port 1 output:  [%.5f, %.5f]   (vacuum floor 1.0, despite the "
          "thermal drive)" % (V[0, 0], V[1, 1]))
    print("   port 2 output:  [%.4g, %.4g]   (free -- see caveat)" % (V[2, 2], V[3, 3]))
    print("   |S_10|^2 = %.4f   |S_01|^2 = %.4f" % (abs(S[1, 0]) ** 2, abs(S[0, 1]) ** 2))
    print("   stability margin: max Re eig(M~) = %.4f" % solution["max_real_eigenvalue"])
    print("\n   The point of this target: with sigma_in = vacuum, 'port 1 stays at the")
    print("   vacuum floor' is free.  It only bites because port 2 is thermally driven,")
    print("   so the device has to route that noise away from port 1 (Sec. 1.2).")
    print("\n   CAVEAT: port 2's own output is unpinned, and the search drove it to a")
    print("   huge value right against the stability boundary -- free entries are free")
    print("   in both directions (Sec. 11).  Pin them if the emitted state matters.")
    SUMMARY.append(("B.7 noise diode", len(irreducibles),
                    "port 1 = %.4f (vacuum), |S_10|^2 = %.2f"
                    % (V[0, 0], abs(S[1, 0]) ** 2)))


if __name__ == "__main__":
    run_cv_graph_state()
    run_backaction_evading()
    run_noise_diode()

    print("\n" + "=" * 74)
    print("%-26s %-14s %s" % ("target", "irreducibles", "verified property"))
    print("-" * 74)
    for name, count, result in SUMMARY:
        print("%-26s %-14i %s" % (name, count, result))
    print("=" * 74)
