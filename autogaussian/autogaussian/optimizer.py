"""
DISCOVERY DRIVER (Sec. 6 of the algorithm flow).

    (a) grow the number of auxiliary modes until the fully connected graph is VALID
    (b) walk down the lattice by complexity -- :func:`autogaussian.search.discover`
    (c) the minimal elements of the valid library are the irreducible graphs

This module owns the *device-level* bookkeeping (graph space, parametrisation,
oracle, graph reduction, reporting).  The lattice walk itself lives in
:mod:`autogaussian.search`, because its pruning rules are where correctness is
won or lost: only a **certified** invalid may condemn its subgraphs, and the
oracle never certifies.  The legacy AUTOSCATTER walk -- in which any invalid
condemns its subtree, repaired afterwards by :meth:`verify_irreducibility` --
is still reachable via ``engine='legacy'``.
"""

import numpy as np
from tqdm import tqdm, trange

from autogaussian.graph import (
    COUPLING_WITH_PHASE,
    COUPLING_WITHOUT_PHASE,
    GraphSpace,
    NO_COUPLING,
    SLOT_BEAMSPLITTER,
    SLOT_DETUNING,
    any_subgraph_of,
    characterize_architectures,
    is_subgraph_of_any,
)
from autogaussian.oracle import CovarianceOracle
from autogaussian.parametrization import Parametrization
from autogaussian.search import discover

__all__ = ["CovarianceArchitectureOptimizer", "find_minimum_number_auxiliary_modes"]


class CovarianceArchitectureOptimizer:
    """Discovery loop for a ``sigma_out(Omega)`` target.

    Parameters
    ----------
    target : :class:`autogaussian.target.CovarianceTarget`
        Pinned entries of the output covariance (Sec. 1.1).
    num_auxiliary_modes : int
        Number of unmonitored (auxiliary) modes on top of the ``P`` ports.
    sigma_in_signal : array, optional
        Declared input covariance of the signal channels (Sec. 1.2).
    intrinsic_losses : bool or sequence of bool
        Modes carrying a free intrinsic loss ``gamma_i``.
    free_decay_ratios : bool or sequence of bool
        Modes carrying a free decay-rate ratio ``kappa~_i``.  These are inert at
        ``Omega = 0``; switch them off for a single-point target to shrink the
        search space.
    optimize_gauge : bool
        Optimise the per-port quadrature phases, or fix them at 0 when the
        target names a quadrature (Sec. 1.5).
    constraints : sequence
        Extra equality constraints ``f_j = 0`` (Sec. 1.4).
    graph_space : GraphSpace, optional
        Pass one explicitly to encode coupling restrictions; otherwise built
        from the ``allow_*`` / ``forbidden_*`` keyword arguments.
    kwargs_optimization : dict
        Forwarded to :meth:`CovarianceOracle.repeated_optimize`
        (``num_tests``, ``method``, ``max_violation_success``, ...).
    make_initial_test : bool
        Verify that the fully connected graph is valid before searching.
    """

    def __init__(
        self,
        target,
        num_auxiliary_modes=0,
        sigma_in_signal=None,
        intrinsic_losses=False,
        free_decay_ratios=None,
        optimize_gauge=True,
        constraints=(),
        graph_space=None,
        stability_margin=1.0e-3,
        stability_weight=1.0,
        asymptotic_bus_modes=(),
        bus_cooperativity=1.0e6,
        log_decay_ratio_bound=np.log(100.0),
        kwargs_optimization=None,
        make_initial_test=True,
        seed=None,
        **graph_space_kwargs
    ):
        self.target = target
        self.num_ports = target.num_ports
        self.num_auxiliary_modes = int(num_auxiliary_modes)
        self.num_modes = self.num_ports + self.num_auxiliary_modes

        if free_decay_ratios is None:
            # decay ratios only shape the spectrum -- pointless for an
            # Omega = 0 only target (App. A.4(c))
            free_decay_ratios = bool(np.any(np.abs(target.omegas) > 0)) or any(
                pin.order > 0 for pin in target.pins)

        if graph_space is None:
            graph_space = GraphSpace(self.num_modes, **graph_space_kwargs)
        elif graph_space_kwargs:
            raise ValueError("pass either graph_space or the graph-space keyword arguments")
        self.space = graph_space
        if self.space.num_modes != self.num_modes:
            raise ValueError("graph space has %i modes, expected %i"
                             % (self.space.num_modes, self.num_modes))

        self.param = Parametrization(
            self.space, self.num_ports,
            intrinsic_losses=intrinsic_losses,
            free_decay_ratios=free_decay_ratios,
            optimize_gauge=optimize_gauge,
            target_symbols=target.free_symbols,
            asymptotic_bus_modes=asymptotic_bus_modes,
            bus_cooperativity=bus_cooperativity,
            log_decay_ratio_bound=log_decay_ratio_bound,
        )
        self.oracle = CovarianceOracle(
            self.param, target,
            sigma_in_signal=sigma_in_signal,
            constraints=constraints,
            stability_margin=stability_margin,
            stability_weight=stability_weight,
        )

        self.kwargs_optimization = {
            "num_tests": 10,
            "method": "BFGS",
            "max_violation_success": 1.0e-10,
            "interrupt_if_successful": True,
        }
        if kwargs_optimization:
            self.kwargs_optimization.update(kwargs_optimization)

        self.rng = np.random.default_rng(seed)

        self.libraries = None        # set by the two-library engine (Sec. 6)
        self.valid_combinations = []
        self.invalid_combinations = []
        self.tested_complexities = []
        self.num_tested_graphs = []
        self.num_tested_invalid_graphs = []
        self.solutions = {}          # bytes(graph) -> info dict of a valid run

        if make_initial_test:
            success, infos = self.test_graph(self.space.fully_connected())
            if not success:
                raise Exception(
                    "fully connected graph is invalid, interrupting "
                    "(best loss %.3g, stable=%s)"
                    % (min(info["loss_reached"] for info in infos),
                       any(info["stable"] for info in infos)))
            print("fully connected graph is a valid graph")

    # ------------------------------------------------------------------ #
    # ORACLE (Sec. 4)
    # ------------------------------------------------------------------ #

    def test_graph(self, graph, **kwargs):
        """Run the oracle on one graph.  Returns ``(success, infos)``."""
        options = dict(self.kwargs_optimization)
        options.update(kwargs)
        free_idxs = self.param.free_indices(graph)
        success, infos = self.oracle.repeated_optimize(free_idxs, rng=self.rng, **options)
        if success:
            self.solutions[bytes(np.asarray(graph, dtype="int8"))] = infos[-1]
        return success, infos

    def solution_of(self, graph):
        """The stored successful run for a graph (if it was ever found valid)."""
        return self.solutions.get(bytes(np.asarray(graph, dtype="int8")))

    # ------------------------------------------------------------------ #
    # graph reduction: read off which elements the solution actually used
    # ------------------------------------------------------------------ #

    def reduce_graph(self, x, tolerance=None):
        """Map a converged parameter vector back to the *smallest* graph that
        describes it (couplings that came out zero are dropped, couplings that
        came out real lose their phase).  Mirrors AUTOSCATTER's
        ``check_all_constraints``."""
        if tolerance is None:
            tolerance = self.kwargs_optimization["max_violation_success"]
        x = np.asarray(x)
        g, nu = self.param.blocks(x)
        g = np.asarray(g)
        nu = np.asarray(nu)
        reduced = np.zeros(self.space.num_slots, dtype="int8")
        for slot_idx, (kind, i, j) in enumerate(self.space.slots):
            if kind == SLOT_DETUNING:
                element = -np.real(g[i, i])
                value = NO_COUPLING if element ** 2 / 2 < tolerance else COUPLING_WITHOUT_PHASE
            else:
                matrix = g if kind == SLOT_BEAMSPLITTER else nu
                element = matrix[i, j]
                if np.abs(element) ** 2 / 2 < tolerance:
                    value = NO_COUPLING
                elif np.imag(element) ** 2 / 2 < tolerance:
                    value = COUPLING_WITHOUT_PHASE
                else:
                    value = COUPLING_WITH_PHASE
            # never claim an element the graph space forbids
            allowed = self.space.possible_values[slot_idx]
            while value not in allowed and value > 0:
                value -= 1
            reduced[slot_idx] = value
        return reduced

    # ------------------------------------------------------------------ #
    # breadth-first search (Sec. 6b)
    # ------------------------------------------------------------------ #

    def identify_potential_combinations(self, complexity_level, skip_check_for_valid_subgraphs=False):
        """Graphs at this complexity that are neither subgraphs of a known
        invalid graph nor extensions of a known valid graph."""
        invalid = np.asarray(self.invalid_combinations)
        valid = np.asarray(self.valid_combinations)
        potential = []
        for graph in self.space.iter_graphs_at_complexity(complexity_level):
            if is_subgraph_of_any(graph, invalid):
                continue
            if not skip_check_for_valid_subgraphs and any_subgraph_of(graph, valid):
                continue
            potential.append(graph)
        return potential

    def find_valid_combinations(self, complexity_level, combinations_to_test=None,
                                perform_graph_reduction=True, progress=True):
        if combinations_to_test is None:
            combinations_to_test = self.identify_potential_combinations(complexity_level)

        newly_added = []
        count_tested = 0
        count_invalid = 0

        iterator = trange(len(combinations_to_test)) if progress else range(len(combinations_to_test))
        for idx in iterator:
            graph = combinations_to_test[idx]
            if any_subgraph_of(graph, np.asarray(newly_added)):
                continue
            success, infos = self.test_graph(graph)
            count_tested += 1
            if success:
                if perform_graph_reduction:
                    graph_to_add = self.reduce_graph(infos[-1]["x"])
                    self.solutions[bytes(np.asarray(graph_to_add, dtype="int8"))] = infos[-1]
                else:
                    graph_to_add = np.asarray(graph, dtype="int8")
                self.valid_combinations.append(graph_to_add)
                newly_added.append(graph_to_add)
            else:
                self.invalid_combinations.append(np.asarray(graph, dtype="int8"))
                count_invalid += 1

        self.tested_complexities.append(complexity_level)
        self.num_tested_graphs.append(count_tested)
        self.num_tested_invalid_graphs.append(count_invalid)

    def cleanup_valid_combinations(self):
        """Keep only the minimal elements of the valid library."""
        if not self.valid_combinations:
            return
        unique = np.unique(np.asarray(self.valid_combinations, dtype="int8"), axis=0)
        cleaned = []
        for idx, graph in enumerate(unique):
            others = np.delete(unique, idx, axis=0)
            if not any_subgraph_of(graph, others):
                cleaned.append(graph)
        self.valid_combinations = cleaned

    def one_step_reductions(self, graph):
        """All graphs obtained by weakening exactly one slot by one step."""
        graph = np.asarray(graph, dtype="int8")
        for slot_idx, value in enumerate(graph):
            if value <= 0:
                continue
            allowed = [v for v in self.space.possible_values[slot_idx] if v < value]
            if not allowed:
                continue
            reduced = graph.copy()
            reduced[slot_idx] = max(allowed)
            yield reduced

    def greedy_minimal_subgraph(self, graph, num_tests=None, verbose=False):
        """Cheap descent: keep weakening one slot at a time as long as the graph
        stays valid.  Returns *a* minimal graph (not the complete list -- use
        :meth:`minimal_valid_subgraphs` for that), which is enough to see which
        elements a solution genuinely needs.
        """
        options = {} if num_tests is None else {"num_tests": num_tests}
        current = np.asarray(graph, dtype="int8")
        improved = True
        while improved:
            improved = False
            for candidate in self.one_step_reductions(current):
                success, _ = self.test_graph(candidate, **options)
                if success:
                    current = candidate
                    improved = True
                    if verbose:
                        print("   -> " + ", ".join(self.space.describe(current)))
                    break
        return current

    def minimal_valid_subgraphs(self, graph, num_tests=None, memo=None, **kwargs):
        """Complete descent through the sub-lattice below a valid graph.

        The layer-by-layer search of Sec. 6 relies on the oracle never
        returning a false negative: one graph that the continuous optimiser
        failed to solve marks *all* of its subgraphs invalid and can hide a
        genuinely minimal solution.  This pass re-derives irreducibility
        directly, with more restarts, and only inside the (small) sub-lattice
        that actually decides the answer.
        """
        memo = {} if memo is None else memo
        options = dict(kwargs)
        if num_tests is not None:
            options["num_tests"] = num_tests

        def valid(candidate):
            key = bytes(np.asarray(candidate, dtype="int8"))
            if key not in memo:
                success, _ = self.test_graph(candidate, **options)
                memo[key] = success
            return memo[key]

        minimal = []
        frontier = [np.asarray(graph, dtype="int8")]
        visited = set()
        while frontier:
            current = frontier.pop()
            key = bytes(current)
            if key in visited:
                continue
            visited.add(key)
            has_valid_child = False
            for candidate in self.one_step_reductions(current):
                if valid(candidate):
                    has_valid_child = True
                    frontier.append(candidate)
            if not has_valid_child:
                minimal.append(current)
        return minimal, memo

    def verify_irreducibility(self, num_tests=None, progress=True, **kwargs):
        """Run :meth:`minimal_valid_subgraphs` below every valid graph found so
        far and replace the library with the confirmed minimal elements."""
        if num_tests is None:
            num_tests = 3 * self.kwargs_optimization["num_tests"]
        memo = {}
        confirmed = []
        graphs = list(self.valid_combinations)
        iterator = tqdm(graphs) if progress else graphs
        for graph in iterator:
            minimal, memo = self.minimal_valid_subgraphs(
                graph, num_tests=num_tests, memo=memo, **kwargs)
            confirmed.extend(minimal)
        if confirmed:
            self.valid_combinations = confirmed
            self.cleanup_valid_combinations()
        return np.array(self.valid_combinations, dtype="int8")

    def perform_breadth_first_search(self, min_complexity=0, progress=True, verify=None,
                                     verify_num_tests=None, verify_kwargs=None,
                                     engine="two_library", **search_kwargs):
        """Full Sec. 6 loop; returns the irreducible graphs.

        ``engine='two_library'`` (default) runs :func:`autogaussian.search.discover`
        and leaves the two libraries on ``self.libraries``; a graph is only
        removed from consideration by a certificate, so no post-hoc repair pass
        is needed and ``verify`` defaults to ``False``.

        ``engine='legacy'`` is the AUTOSCATTER walk: any invalid condemns its
        subgraphs, and :meth:`verify_irreducibility` afterwards re-derives
        irreducibility inside the sub-lattice to repair the false negatives
        that rule creates.  Faster, and complete only if the oracle never
        misses.
        """
        if engine == "two_library":
            libraries = discover(self, min_complexity=min_complexity, progress=progress,
                                 **search_kwargs)
            self.libraries = libraries
            self.valid_combinations = [np.asarray(g, dtype="int8")
                                       for g in libraries.minimal_valid()]
            self.invalid_combinations = [np.asarray(e.graph, dtype="int8")
                                         for e in libraries.invalid.values()]
            if verify:
                print("verifying irreducibility (repairs false negatives)")
                self.verify_irreducibility(num_tests=verify_num_tests, progress=progress,
                                           **(verify_kwargs or {}))
            print("optimisation finished, list of irreducible graphs has %i elements "
                  "(%i uncertified rejections)"
                  % (len(self.valid_combinations), libraries.n_uncertified()))
            return np.array(self.valid_combinations, dtype="int8")

        if engine != "legacy":
            raise ValueError("unknown engine %r" % engine)

        verify = True if verify is None else verify
        print("start breadth-first search over %i slots (max complexity %i)"
              % (self.space.num_slots, self.space.max_complexity))
        for level in range(self.space.max_complexity, min_complexity - 1, -1):
            potential = self.identify_potential_combinations(level)
            if not potential:
                continue
            print("test %i graphs with %i degrees of freedom:" % (len(potential), level))
            self.find_valid_combinations(level, combinations_to_test=potential, progress=progress)
            self.cleanup_valid_combinations()
        if verify:
            print("verifying irreducibility (repairs false negatives)")
            self.verify_irreducibility(num_tests=verify_num_tests, progress=progress,
                                       **(verify_kwargs or {}))
        print("optimisation finished, list of irreducible graphs has %i elements"
              % len(self.valid_combinations))
        return np.array(self.valid_combinations, dtype="int8")

    # ------------------------------------------------------------------ #
    # Sec. 8 -- completeness reporting
    # ------------------------------------------------------------------ #

    def n_uncertified(self):
        """Graphs rejected without an infeasibility certificate (Sec. 8)."""
        return 0 if self.libraries is None else self.libraries.n_uncertified()

    def completeness_statement(self):
        if self.libraries is None:
            return ("No two-library run recorded: completeness cannot be stated "
                    "(the legacy engine prunes on uncertified invalids).")
        return self.libraries.completeness_statement(self.space)

    # ------------------------------------------------------------------ #
    # reporting (Sec. 7.1 / Sec. 8)
    # ------------------------------------------------------------------ #

    def characterize(self, graphs=None):
        graphs = self.valid_combinations if graphs is None else graphs
        return characterize_architectures(graphs, self.space)

    def report(self, graphs=None):
        """Sorted human-readable summary of the discovered devices."""
        graphs = self.valid_combinations if graphs is None else graphs
        info = self.characterize(graphs)
        order = np.lexsort((info["min_number_of_pumps"], info["num_couplings"],
                            info["complexity"]))
        lines = []
        for rank, idx in enumerate(order):
            graph = graphs[idx]
            solution = self.solution_of(graph)
            lines.append("#%i  complexity=%i  couplings=%i  pumps=%s"
                         % (rank, info["complexity"][idx], info["num_couplings"][idx],
                            info["min_number_of_pumps"][idx]))
            for element in self.space.describe(graph):
                lines.append("      " + element)
            if solution is not None:
                lines.append("      loss=%.3g  max Re eig(M~)=%.3g"
                             % (solution["loss_reached"], solution["max_real_eigenvalue"]))
        return "\n".join(lines)


def find_minimum_number_auxiliary_modes(
    target, start_value=0, max_value=3, verbose=True, **kwargs
):
    """Sec. 6(a): grow the number of auxiliary modes until the fully connected
    graph realises the target.  Returns a ready-to-search optimizer (or None).
    """
    for num_auxiliary_modes in range(start_value, max_value + 1):
        if verbose:
            print("testing %i auxiliary modes" % num_auxiliary_modes)
        optimizer = CovarianceArchitectureOptimizer(
            target, num_auxiliary_modes=num_auxiliary_modes,
            make_initial_test=False, **kwargs)
        success, infos = optimizer.test_graph(optimizer.space.fully_connected())
        if success:
            if verbose:
                print("success, minimum number of auxiliary modes is %i" % num_auxiliary_modes)
            return optimizer
        if verbose:
            print("not successful (best loss %.3g)"
                  % min(info["loss_reached"] for info in infos))
    if verbose:
        print("minimum number of auxiliary modes not found within interval [%i,%i]"
              % (start_value, max_value))
    return None
