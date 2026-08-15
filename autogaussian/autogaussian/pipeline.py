"""
TOP-LEVEL PSEUDOCODE (Sec. 9 of the algorithm flow)::

    function MAIN(sigma_target, sigma_in, port_structure, constraints, gauge_policy, budget):
        spec = assemble(...)
        valid, invalid = DISCOVER(spec)                # Sec. 6 -> ORACLE -> FORWARD MAP
        irreducibles = minimal_elements(valid)
        for g in irreducibles:
            g.complexity = analyze_complexity(g)       # Sec. 7.1
            g.rules      = symbolic_regression(g, spec)# Sec. 7.2
        if constraints.has_mediation_restriction:
            irreducibles += asymptotic_solutions(spec) # Sec. 7.3
        return sort(irreducibles), invalid, n_uncertified

The third return value is not decoration.  ``n_uncertified`` counts the graphs
that were rejected without an infeasibility certificate, and it is the exact
strength of the claim the run is allowed to make (Sec. 8): the list is complete
and certified iff ``n_uncertified == 0``, and otherwise complete up to at most
that many false negatives -- which are listed by name.
"""

import numpy as np

from autogaussian.optimizer import (
    CovarianceArchitectureOptimizer,
    find_minimum_number_auxiliary_modes,
)
from autogaussian.postprocess import complexity_table, rank_architectures, symbolic_regression

__all__ = ["discover"]


def discover(
    target,
    sigma_in_signal=None,
    constraints=(),
    start_auxiliary_modes=0,
    max_auxiliary_modes=2,
    num_auxiliary_modes=None,
    regression_symbol=None,
    regression_values=None,
    verbose=True,
    verify=False,
    engine="two_library",
    search_kwargs=None,
    **kwargs
):
    """Run the whole discovery loop for one ``sigma_out(Omega)`` target.

    Parameters
    ----------
    target : CovarianceTarget
    sigma_in_signal : array or callable, optional
        Declared input covariance (vacuum by default).
    constraints : sequence
        Extra equality constraints ``f_j = 0``.
    start_auxiliary_modes, max_auxiliary_modes : int
        Budget for step (a) of DISCOVER.
    num_auxiliary_modes : int, optional
        Skip the search for the minimum and use this number directly.
    regression_symbol, regression_values :
        If given, fit construction rules by sweeping this free target symbol
        over these values for every irreducible graph (Sec. 7.2).
    engine : {'two_library', 'legacy'}
        Lattice walk to use; see
        :meth:`CovarianceArchitectureOptimizer.perform_breadth_first_search`.
    search_kwargs : dict, optional
        Forwarded to :func:`autogaussian.search.discover`
        (``use_certificates``, ``escalate_num_tests``, ``max_oracle_calls``...).
    **kwargs
        Forwarded to :class:`CovarianceArchitectureOptimizer` (graph-space
        restrictions, ``intrinsic_losses``, ``asymptotic_bus_modes``, ...).

    Returns
    -------
    dict with keys ``optimizer``, ``irreducibles`` (ranked), ``complexity``,
    ``cost``, ``rules``, ``solutions``, ``libraries``, ``invalid``,
    ``n_uncertified`` and ``completeness``.
    """
    empty = {"optimizer": None, "irreducibles": np.array([]), "complexity": {},
             "cost": np.array([]), "rules": {}, "solutions": [], "libraries": None,
             "invalid": {}, "n_uncertified": 0,
             "completeness": "No graph was found within the mode budget; nothing is claimed."}
    if num_auxiliary_modes is None:
        optimizer = find_minimum_number_auxiliary_modes(
            target, start_value=start_auxiliary_modes, max_value=max_auxiliary_modes,
            verbose=verbose, sigma_in_signal=sigma_in_signal, constraints=constraints,
            **kwargs)
        if optimizer is None:
            return empty
    else:
        optimizer = CovarianceArchitectureOptimizer(
            target, num_auxiliary_modes=num_auxiliary_modes,
            sigma_in_signal=sigma_in_signal, constraints=constraints, **kwargs)

    irreducibles = optimizer.perform_breadth_first_search(
        verify=verify, progress=verbose, engine=engine, **(search_kwargs or {}))
    if len(irreducibles) == 0:
        return dict(empty, optimizer=optimizer, libraries=optimizer.libraries,
                    n_uncertified=optimizer.n_uncertified(),
                    completeness=optimizer.completeness_statement())

    order, info, cost = rank_architectures(irreducibles, optimizer.space)
    irreducibles = irreducibles[order]
    cost = cost[order]
    info = {key: np.asarray(value)[order] for key, value in info.items()}

    rules = {}
    if regression_symbol is not None and regression_values is not None:
        for idx, graph in enumerate(irreducibles):
            if verbose:
                print("symbolic regression for graph #%i: %s"
                      % (idx, ", ".join(optimizer.space.describe(graph))))
            rules[idx] = symbolic_regression(optimizer, graph, regression_symbol,
                                             regression_values)

    solutions = [optimizer.solution_of(graph) for graph in irreducibles]

    libraries = optimizer.libraries
    completeness = optimizer.completeness_statement()

    if verbose:
        print(optimizer.report(irreducibles))
        print(completeness)

    return {
        "optimizer": optimizer,
        "irreducibles": irreducibles,
        "complexity": info,
        "cost": cost,
        "rules": rules,
        "solutions": solutions,
        "libraries": libraries,
        "invalid": {} if libraries is None else libraries.invalid,
        "n_uncertified": optimizer.n_uncertified(),
        "completeness": completeness,
    }
