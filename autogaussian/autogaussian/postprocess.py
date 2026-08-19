"""
POST-PROCESSING (Sec. 7 of the algorithm flow).

7.1  complexity analysis -- couplings + minimum number of parametric pumps,
     ranked for a target platform.
7.2  symbolic regression -- sweep the continuous target parameter, refit the
     device, and fit closed forms ``C_ij(target)``, ``Phi(target)``,
     ``kappa~_i(target)``: the discovered construction rules.
7.3  asymptotic (infinite-cooperativity) handling -- see
     ``Parametrization(asymptotic_bus_modes=..)``; the bus couplings are scaled
     by ``sqrt(C_bus)`` so only the finite ratios matter, with error ~1/C_bus.
"""

import warnings

import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

from autogaussian.graph import characterize_architectures

__all__ = [
    "complexity_table",
    "rank_architectures",
    "sweep_target_parameter",
    "fit_closed_form",
    "symbolic_regression",
    "solution_table",
    "parameter_summary",
]


# --------------------------------------------------------------------------
# 7.1 complexity analysis
# --------------------------------------------------------------------------

def complexity_table(graphs, space):
    """Per-graph counts of detunings / couplings / squeezers / pumps."""
    return characterize_architectures(list(graphs), space)


def rank_architectures(graphs, space, weights=None):
    """Rank irreducible graphs for hardware selection.

    Default cost = ``pumps + couplings + 0.5 * complexity`` -- pumps are the
    expensive resource on most platforms.  Returns ``(order, info, cost)``.
    """
    info = complexity_table(graphs, space)
    weights = weights or {"min_number_of_pumps": 1.0, "num_couplings": 1.0,
                          "complexity": 0.5}
    cost = np.zeros(len(list(graphs)), dtype=float)
    for key, weight in weights.items():
        cost += weight * np.asarray(info[key], dtype=float)
    order = np.argsort(cost)
    return order, info, cost


# --------------------------------------------------------------------------
# 7.2 symbolic regression
# --------------------------------------------------------------------------

def sweep_target_parameter(optimizer, graph, symbol, values, warm_start=True,
                           num_tests=None, **kwargs):
    """Refit one graph while sweeping a free symbol of the target.

    Returns ``(values_kept, dataset)`` where ``dataset`` maps a parameter name
    to an array over the sweep.  Points where the optimisation failed are
    dropped (and reported in ``dataset['_loss']``).
    """
    param = optimizer.param
    if symbol not in param.target_symbols:
        raise ValueError("%s is not a free symbol of the target" % symbol)
    symbol_idx = int(param.target_symbol_idx[param.target_symbols.index(symbol)])

    free_idxs = param.free_indices(graph)
    options = dict(optimizer.kwargs_optimization)
    if num_tests is not None:
        options["num_tests"] = num_tests
    options.update(kwargs)
    num_tests_total = options.pop("num_tests", 10)
    options.pop("interrupt_if_successful", None)

    kept, records = [], []
    previous = None
    for value in np.asarray(values, dtype=float):
        best = None
        for attempt in range(int(num_tests_total)):
            x0 = None
            if warm_start and previous is not None and attempt == 0:
                x0 = np.asarray(previous)[np.asarray(
                    [i for i in free_idxs if i != symbol_idx], dtype=int)]
            success, info = optimizer.oracle.optimize(
                free_idxs, x0=x0, rng=optimizer.rng,
                fixed_values={symbol_idx: value}, **options)
            if best is None or info["loss_reached"] < best["loss_reached"]:
                best = info
            if success:
                break
        if best is None or not best["loss_below_tolerance"]:
            continue
        previous = best["x"]
        kept.append(float(value))
        records.append(best)

    dataset = {"_loss": np.array([record["loss_reached"] for record in records])}
    if records:
        report = records[0]["parameters"]
        for section in ("cooperativities", "phases", "detunings",
                        "intrinsic_losses", "decay_ratios"):
            for key in report[section]:
                dataset[key] = np.array(
                    [record["parameters"][section].get(key, np.nan) for record in records])
    return np.array(kept), dataset


def _candidate_models():
    t = sp.Symbol("t", positive=True)
    a, b, c = sp.symbols("a b c", real=True)
    return [
        ("constant", lambda t, a: a * np.ones_like(t), sp.Float(1) * a, (a,)),
        ("linear", lambda t, a, b: a * t + b, a * t + b, (a, b)),
        ("power", lambda t, a, b: a * np.power(np.abs(t), b), a * t ** b, (a, b)),
        ("inverse", lambda t, a, b: a / t + b, a / t + b, (a, b)),
        ("sqrt", lambda t, a, b: a * np.sqrt(np.abs(t)) + b, a * sp.sqrt(t) + b, (a, b)),
        ("rational", lambda t, a, b, c: (a * t + b) / (t + c), (a * t + b) / (t + c),
         (a, b, c)),
        ("quadratic", lambda t, a, b, c: a * t ** 2 + b * t + c,
         a * t ** 2 + b * t + c, (a, b, c)),
        # squeezing recipes are typically rational in sqrt(variance):
        # e.g. C(v) = ((1 - sqrt(v)) / (1 + sqrt(v)))^2  for a single-mode squeezer
        ("sqrt_rational_squared",
         lambda t, a, b, c: a * ((np.sqrt(np.abs(t)) + b) / (np.sqrt(np.abs(t)) + c)) ** 2,
         a * ((sp.sqrt(t) + b) / (sp.sqrt(t) + c)) ** 2, (a, b, c)),
    ]


def fit_closed_form(t, y, models=None, relative_tolerance=1.0e-3):
    """Fit a small library of closed forms to ``y(t)``; return the simplest
    model whose RMS residual is below ``relative_tolerance`` (relative to the
    spread of ``y``), else the best one.

    Returns ``(name, sympy_expression, rmse)``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(t) & np.isfinite(y)
    t, y = t[finite], y[finite]
    if len(t) < 2:
        return None, None, np.inf
    scale = max(np.ptp(y), 1.0e-12)

    results = []
    rng = np.random.default_rng(0)
    for name, func, expr, symbols in (models or _candidate_models()):
        # multi-start: curve_fit defaults every parameter to 1, which is a
        # degenerate point for several of these models (e.g. the sqrt-rational
        # collapses to a constant at b = 1) and it cannot escape.  Try a few
        # starting points and keep the best fit.
        num_parameters = len(symbols)
        starts = [np.ones(num_parameters), -np.ones(num_parameters)]
        starts.append(np.array([(-1.0) ** k for k in range(num_parameters)]))
        starts.extend(rng.normal(size=(5, num_parameters)))
        popt, residual = None, np.inf
        for start in starts:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    candidate, _ = curve_fit(func, t, y, p0=start, maxfev=20000)
                value = np.sqrt(np.mean((func(t, *candidate) - y) ** 2))
            except Exception:
                continue
            if np.isfinite(value) and value < residual:
                popt, residual = candidate, value
        if popt is None:
            continue
        substituted = expr.subs(
            {symbol: sp.Float(round(float(value), 6)) for symbol, value in zip(symbols, popt)})
        results.append((residual, name, sp.simplify(substituted)))

    if not results:
        return None, None, np.inf
    results.sort(key=lambda item: item[0])
    for residual, name, expr in results:
        if residual < relative_tolerance * scale:
            return name, expr, residual
    residual, name, expr = results[0]
    return name, expr, residual


def symbolic_regression(optimizer, graph, symbol, values, relative_tolerance=1.0e-3,
                        **kwargs):
    """Sec. 7.2 end to end: sweep -> dataset -> closed forms.

    Returns ``{'sweep': t, 'dataset': {...}, 'rules': {name: (model, expr, rmse)}}``.
    """
    t, dataset = sweep_target_parameter(optimizer, graph, symbol, values, **kwargs)
    rules = {}
    for key, series in dataset.items():
        if key.startswith("_"):
            continue
        if np.allclose(np.nan_to_num(series), 0.0):
            continue
        rules[key] = fit_closed_form(t, series, relative_tolerance=relative_tolerance)
    return {"sweep": t, "dataset": dataset, "rules": rules}


# --------------------------------------------------------------------------
# 7.1 / Sec. 8 reporting -- the discovered devices *with* their parameters
# --------------------------------------------------------------------------

def _partner_cooperativity(phase_key):
    """The cooperativity a pump phase belongs to (``arg(nu_{i,j})`` ->
    ``C_{i,i}`` / ``C^{SQZ}_{i,j}``, ``arg(g_{i,j})`` -> ``C^{BS}_{i,j}``)."""
    if not phase_key.startswith("arg("):
        return None
    body = phase_key[4:-1]
    kind, _, indices = body.partition("_")
    i, j = indices.strip("{}").split(",")
    if kind == "g":
        return "C^{BS}_{%s,%s}" % (i, j)
    return "C_{%s,%s}" % (i, j) if i == j else "C^{SQZ}_{%s,%s}" % (i, j)


def _live_entries(report, section, threshold):
    """Entries of one section that describe an element the solution actually
    uses.  A phase whose coupling came out zero is graph-reduction residue and
    is dropped, so the printed parameters match the printed architecture."""
    entries = {key: value for key, value in report[section].items()
               if abs(value) > threshold}
    if section != "phases":
        return entries
    live = {key for key, value in report["cooperativities"].items()
            if abs(value) > threshold}
    return {key: value for key, value in entries.items()
            if _partner_cooperativity(key) in live}


def solution_table(optimizer, graphs=None, limit=None, sort=True,
                   resolve_missing=False, num_tests=30):
    """Human-readable table of valid graphs **and** the parameters the oracle
    converged to for each of them.

    ``optimizer.report()`` lists the architectures; this adds the physical
    solution behind every architecture -- cooperativities, pump phases,
    detunings, decay ratios and intrinsic losses -- i.e. the numbers a
    hardware implementation actually needs.

    Parameters
    ----------
    optimizer : CovarianceArchitectureOptimizer
    graphs : sequence of graph vectors, optional
        Defaults to ``optimizer.valid_combinations`` (the irreducible list).
    limit : int, optional
        Only print this many graphs (the rest are counted in one line).
    sort : bool
        Order by complexity (cheapest device first).
    resolve_missing : bool
        Re-run the oracle for graphs whose solution was not cached (a graph
        proved valid inside a sub-library keeps its solution, but a graph that
        was only ever *inherited* may not have one).
    num_tests : int
        Restarts used by ``resolve_missing``.

    Returns
    -------
    str
    """
    graphs = optimizer.valid_combinations if graphs is None else graphs
    graphs = [np.asarray(graph, dtype="int8") for graph in graphs]
    if not graphs:
        return "no valid graph"

    order = list(range(len(graphs)))
    if sort:
        order.sort(key=lambda idx: (int(np.sum(graphs[idx])), idx))

    info = complexity_table([graphs[idx] for idx in order], optimizer.space)
    lines = ["%i valid graph(s), with the converged parameters:" % len(graphs)]
    shown = order if limit is None else order[:limit]

    for rank, idx in enumerate(shown):
        graph = graphs[idx]
        solution = optimizer.solution_of(graph)
        if solution is None and resolve_missing:
            success, infos = optimizer.test_graph(graph, num_tests=num_tests)
            solution = infos[-1] if success else None
        lines.append("")
        lines.append("#%i  complexity=%i  couplings=%i  pumps=%i"
                     % (rank, info["complexity"][rank], info["num_couplings"][rank],
                        info["min_number_of_pumps"][rank]))
        for element in optimizer.space.describe(graph):
            lines.append("      " + element)
        if solution is None:
            lines.append("      (no cached solution; pass resolve_missing=True)")
            continue
        lines.append("      loss=%.2e   max Re eig(M~)=%+.4f"
                     % (solution["loss_reached"], solution["max_real_eigenvalue"]))
        report = solution["parameters"]
        for section in ("cooperativities", "phases", "detunings",
                        "intrinsic_losses", "decay_ratios", "gauge_phases"):
            entries = _live_entries(report, section, 1.0e-9)
            if not entries:
                continue
            lines.append("      %s:" % section)
            for key, value in entries.items():
                lines.append("         %-20s %+.4f" % (key, value))

    if limit is not None and len(order) > limit:
        lines.append("")
        lines.append("... and %i more graph(s)" % (len(order) - limit))
    return "\n".join(lines)


def parameter_summary(info, space=None, graph=None, indent="   ", threshold=1.0e-9):
    """Compact one-line-per-section view of a single converged solution.

    ``info`` is an oracle result dict (as returned by ``test_graph`` /
    ``solution_of``).  Pass ``space`` and ``graph`` to prefix the architecture
    the numbers belong to.
    """
    lines = []
    if space is not None and graph is not None:
        lines.append("%sgraph: %s" % (indent, ", ".join(space.describe(graph))))
    lines.append("%sloss=%.2e   max Re eig(M~)=%+.4f"
                 % (indent, info["loss_reached"], info["max_real_eigenvalue"]))
    for section in info["parameters"]:
        entries = _live_entries(info["parameters"], section, threshold)
        if not entries:
            continue
        lines.append("%s%-16s %s"
                     % (indent, section + ":",
                        "  ".join("%s=%+.4f" % item for item in entries.items())))
    return "\n".join(lines)
