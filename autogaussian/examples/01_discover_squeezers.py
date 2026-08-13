"""
Example 1 -- discovery + construction rules for squeezing sources.

Runs the full Sec. 9 loop on two gallery targets and then fits the closed-form
recipes of Sec. 7.2:

    B.1  single-mode squeezer   (validation)
    B.2  EPR / two-mode-squeezed source

Usage:  python examples/01_discover_squeezers.py
"""

import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogaussian import CovarianceArchitectureOptimizer, CovarianceTarget, discover
from autogaussian.gallery import epr_source, single_mode_squeezer
from autogaussian.postprocess import symbolic_regression


def banner(text):
    print("\n" + "=" * 72 + "\n" + text + "\n" + "=" * 72)


def discover_single_mode_squeezer():
    banner("B.1  single-mode squeezer:  sigma_target = [[0.5, 0], [0, 2.0]]")
    problem = single_mode_squeezer(variance=0.5)
    print(problem.target)
    result = discover(problem.target, num_auxiliary_modes=0, verbose=True,
                      seed=0, **problem.optimizer_kwargs())
    return result


def construction_rules_for_squeezing_depth():
    banner("Sec. 7.2  construction rule  C_00(v)  for the single-mode squeezer")
    v = sp.Symbol("v", positive=True)
    target = CovarianceTarget(num_ports=1, name="squeezer with free depth")
    target.pin((0, 0), v)          # squeezed quadrature -- the swept parameter
    target.pin((1, 1), 1 / v)      # pure state
    target.pin((0, 1), 0.0)

    optimizer = CovarianceArchitectureOptimizer(
        target, num_auxiliary_modes=0, optimize_gauge=False, make_initial_test=False,
        seed=0)
    graph = optimizer.space.empty()
    graph[optimizer.space.slots.index(("onsite_squeezing", 0, 0))] = 2

    result = symbolic_regression(optimizer, graph, v, np.linspace(0.05, 0.95, 19))
    print("swept %i target values" % len(result["sweep"]))
    for name, (model, expression, rmse) in result["rules"].items():
        print("   %-14s = %s        [%s, rmse %.1e]" % (name, expression, model, rmse))
    print("\n   analytic reference:  C_00(v) = ((1 - sqrt(v)) / (1 + sqrt(v)))^2")
    return result


def discover_epr_source():
    banner("B.2  EPR source:  local 1.25, correlations +-0.75  (Duan sum 1.0 < 2)")
    problem = epr_source(local=1.25, correlation=0.75)
    print(problem.target)
    result = discover(problem.target, num_auxiliary_modes=0, verbose=True,
                      seed=2, **problem.optimizer_kwargs())

    banner("emitted covariance of the cheapest solution")
    optimizer = result["optimizer"]
    best = result["irreducibles"][0]
    solution = optimizer.solution_of(best)
    V = np.real(optimizer.oracle.covariance(solution["x"], 0.0))
    print(", ".join(optimizer.space.describe(best)))
    print(np.round(V, 4))
    return result


if __name__ == "__main__":
    discover_single_mode_squeezer()
    construction_rules_for_squeezing_depth()
    discover_epr_source()
