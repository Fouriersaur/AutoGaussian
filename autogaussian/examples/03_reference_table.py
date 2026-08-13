"""
Example 3 -- reproduce the verified reference table of App. F.

    | target                | minimal setup                          | result at Omega = 0        |
    |-----------------------|----------------------------------------|----------------------------|
    | single-mode squeezing | 1 mode, on-site squeezing, 1 port      | 12 dB (variance 0.062)     |
    | EPR entanglement      | 2 modes, 1 squeezing edge, 2 ports     | joint 0.062; local 8.0     |
    | reciprocal squeezing  | 2 modes, BS + squeezing edge, 2 ports  | 5 dB at both ports (0.315) |
    | directional squeezing | 2 signal + 1 damped aux                | 7 dB / vacuum, fwd 1.30    |

Each row states the *setup* explicitly, so this script checks the forward map
and the oracle rather than the search: it pins the target, hands the oracle the
named graph, and prints what came out.

Usage:  python examples/03_reference_table.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogaussian import (
    CovarianceArchitectureOptimizer,
    CovarianceTarget,
    TransmissionConstraint,
    duan_sum,
    variance_to_dB,
)

ROWS = []


def solve(target, graph_elements, num_auxiliary_modes=0, constraints=(), seed=0,
          num_tests=30, **kwargs):
    """Build the named graph, run the oracle on it, return (optimizer, x)."""
    optimizer = CovarianceArchitectureOptimizer(
        target, num_auxiliary_modes=num_auxiliary_modes, optimize_gauge=False,
        constraints=constraints, make_initial_test=False, seed=seed,
        kwargs_optimization={"num_tests": num_tests}, **kwargs)
    if graph_elements is None:
        graph = optimizer.space.fully_connected()
    else:
        graph = optimizer.space.empty()
        for element in graph_elements:
            graph[optimizer.space.slots.index(element[:3])] = element[3]
    success, infos = optimizer.test_graph(graph)
    best = infos[-1]
    return optimizer, best, success


def single_mode_squeezing():
    print("\n-- single-mode squeezing: 1 mode, on-site squeezing, 1 port")
    variance = 0.062
    target = CovarianceTarget(num_ports=1)
    target.pin((1, 1), variance)
    target.pin((0, 0), 1.0 / variance)
    target.pin((0, 1), 0.0)
    optimizer, best, success = solve(target, [("onsite_squeezing", 0, 0, 2)])
    V = np.real(optimizer.oracle.covariance(best["x"], 0.0))
    print("   valid=%s   variance = %.4f  ->  %.2f dB   (target %.2f dB)"
          % (success, V[1, 1], variance_to_dB(V[1, 1]), variance_to_dB(variance)))
    ROWS.append(("single-mode squeezing", success, "%.3f (%.1f dB)"
                 % (V[1, 1], variance_to_dB(V[1, 1]))))


def epr_entanglement():
    print("\n-- EPR entanglement: 2 modes, 1 squeezing edge, 2 ports")
    local = 8.0
    correlation = float(np.sqrt(local ** 2 - 1.0))     # pure two-mode squeezed vacuum
    target = CovarianceTarget(num_ports=2)
    target.pin_matrix([[local, 0.0, correlation, 0.0],
                       [0.0, local, 0.0, -correlation],
                       [correlation, 0.0, local, 0.0],
                       [0.0, -correlation, 0.0, local]])
    optimizer, best, success = solve(target, [("two_mode_squeezing", 0, 1, 2)])
    V = np.real(optimizer.oracle.covariance(best["x"], 0.0))
    joint = 0.5 * (V[0, 0] + V[2, 2]) - V[0, 2]        # Var((x_1 - x_2)/sqrt(2))
    print("   valid=%s   local = %.3f   joint quadrature = %.4f   Duan sum = %.4f (< 4)"
          % (success, V[0, 0], joint, duan_sum(V, 0, 1)))
    ROWS.append(("EPR entanglement", success, "joint %.4f, local %.2f" % (joint, V[0, 0])))


def reciprocal_squeezing():
    print("\n-- reciprocal squeezing: 2 modes, BS + squeezing edge, 2 ports")
    variance = 0.315
    target = CovarianceTarget(num_ports=2)
    target.pin((1, 1), variance)                       # p_1
    target.pin((3, 3), variance)                       # p_2
    optimizer, best, success = solve(
        target, [("beamsplitter", 0, 1, 2), ("two_mode_squeezing", 0, 1, 2)])
    V = np.real(optimizer.oracle.covariance(best["x"], 0.0))
    print("   valid=%s   port 1 = %.4f (%.2f dB)   port 2 = %.4f (%.2f dB)"
          % (success, V[1, 1], variance_to_dB(V[1, 1]), V[3, 3], variance_to_dB(V[3, 3])))
    ROWS.append(("reciprocal squeezing", success, "%.3f / %.3f (%.1f dB both)"
                 % (V[1, 1], V[3, 3], variance_to_dB(V[1, 1]))))


def directional_squeezing():
    print("\n-- directional squeezing: 2 signal modes + 1 damped auxiliary")
    variance = 0.2                                     # ~7 dB
    target = CovarianceTarget(num_ports=2)
    target.pin_matrix([[1.0, 0.0, None, None],
                       [0.0, 1.0, None, None],
                       [None, None, variance, 0.0],
                       [None, None, 0.0, 1.0 / variance]])
    # this row is the one App. F leaves open ("on-site sqz + BS + phases"), so
    # hand the oracle the fully connected 3-mode graph and read off afterwards
    # which elements the solution actually used
    optimizer, best, success = solve(
        target, None, num_auxiliary_modes=1,
        constraints=(TransmissionConstraint(1, 0, 1.3),), seed=3, num_tests=40)
    V = np.real(optimizer.oracle.covariance(best["x"], 0.0))
    minimal = optimizer.greedy_minimal_subgraph(optimizer.space.fully_connected(),
                                                num_tests=10)
    print("   a minimal setup: %s" % ", ".join(optimizer.space.describe(minimal)))
    best = optimizer.solution_of(minimal) or best
    V = np.real(optimizer.oracle.covariance(best["x"], 0.0))
    S, _ = optimizer.oracle.scattering(best["x"], 0.0)
    print("   valid=%s   port 1 = [%.4f, %.4f] (vacuum)   port 2 = %.4f (%.2f dB)"
          % (success, V[0, 0], V[1, 1], V[2, 2], variance_to_dB(V[2, 2])))
    print("   transport: forward %.3f   backward %.3f"
          % (abs(S[1, 0]) ** 2, abs(S[0, 1]) ** 2))
    ROWS.append(("directional squeezing", success,
                 "%.1f dB / vacuum, fwd %.2f, bwd %.2f"
                 % (variance_to_dB(V[2, 2]), abs(S[1, 0]) ** 2, abs(S[0, 1]) ** 2)))


if __name__ == "__main__":
    single_mode_squeezing()
    epr_entanglement()
    reciprocal_squeezing()
    directional_squeezing()

    print("\n" + "=" * 72)
    print("%-24s %-8s %s" % ("target", "valid", "result at Omega = 0"))
    print("-" * 72)
    for name, success, result in ROWS:
        print("%-24s %-8s %s" % (name, success, result))
    print("=" * 72)
    sys.exit(0 if all(row[1] for row in ROWS) else 1)
