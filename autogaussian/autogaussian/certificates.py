"""
INFEASIBILITY CERTIFICATES (Sec. 6 ESCALATE of the algorithm flow).

The oracle is an existential search: it can exhibit a witness, never the
absence of one.  Everything in this module goes the other way -- each function
either returns a *proof* that no witness exists, or admits that it does not
know.  Only a proof may set ``certified=True``, and only ``certified=True``
prunes a subtree (Sec. 6).

Three rungs, in increasing order of ambition and decreasing order of maturity:

``certify_fit_infeasible``
    The target is outside the reachable set.  Two sound sub-tests: the pinned
    entries do not describe a bona-fide covariance at all
    (``sigma + i Omega >= 0`` infeasible -- an SDP whose Farkas dual *is* the
    certificate), or the graph is passive and therefore maps vacuum to vacuum,
    so any non-vacuum pin is unreachable *for this graph*.

``certify_no_hurwitz``
    No stable parameter point exists.  Certified here only by the PBH
    dark-mode test: a subspace that the dissipators do not reach is
    non-stabilizable for the whole parameter family, no matter how the
    couplings are tuned.

``sos_no_hurwitz``
    The real thing -- Lyapunov LMI plus the fit as polynomial equalities, run
    through a moment-SOS hierarchy until a level is infeasible.  **STUB.**  The
    v1 stand-in samples the feasible set and reports evidence; evidence is not
    a proof, so it returns ``False`` and the graph lands in
    ``INVALID_DEFAULT`` (non-condemning).  That is by design, not an oversight:
    the alternative is to prune subtrees on a hunch.
"""

import warnings

import numpy as np

from autogaussian.graph import (
    NO_COUPLING,
    SLOT_ONSITE_SQUEEZING,
    SLOT_TWO_MODE_SQUEEZING,
)
from autogaussian.types import REASON_FIT_RANGE, REASON_PASSIVE, REASON_PBH

__all__ = [
    "certify_fit_infeasible",
    "certify_no_hurwitz",
    "certify_target_unphysical",
    "certify_passive_range",
    "pbh_dark_mode",
    "pbh_non_stabilizable",
    "sos_no_hurwitz",
    "lyapunov_feasible",
    "symplectic_form",
    "has_cvxpy",
]


def has_cvxpy():
    try:
        import cvxpy  # noqa: F401
    except Exception:
        return False
    return True


def symplectic_form(num_modes):
    """``Omega`` in the interleaved ``(x_1,p_1,x_2,p_2,...)`` basis.

    With the vacuum floor normalised to variance 1, a matrix ``V`` is a
    bona-fide covariance matrix iff ``V + i Omega >= 0``.
    """
    Omega = np.zeros((2 * num_modes, 2 * num_modes))
    for j in range(num_modes):
        Omega[2 * j, 2 * j + 1] = 1.0
        Omega[2 * j + 1, 2 * j] = -1.0
    return Omega


# ---------------------------------------------------------------------------
# rung 1a -- the target is not a covariance matrix at all
# ---------------------------------------------------------------------------

def certify_target_unphysical(target, num_ports, solver=None, verbose=False):
    """SDP: is there **any** bona-fide covariance matching the pinned entries?

    Variables: one Hermitian ``2P x 2P`` matrix per pinned frequency.
    Constraints: the pins, and ``V + i Omega >= 0``.  Infeasible means no
    quantum state whatsoever has the pinned second moments, so *every* graph
    fails -- a certificate that does not depend on the graph.

    Returns ``(fired, detail)``.  Derivative pins and pins whose value is a free
    sympy symbol are ignored (they weaken the test, never invalidate it).
    """
    if not has_cvxpy():
        return False, {"reason": "cvxpy_missing", "conclusive": False}
    import cvxpy as cp
    import sympy as sp

    Omega = symplectic_form(num_ports)
    dimension = 2 * num_ports
    frequencies = sorted(set(float(pin.omega) for pin in target.pins if pin.order == 0))
    details = {"checked_frequencies": [], "solver_status": {}}

    for omega in frequencies:
        pins = [pin for pin in target.pins if pin.order == 0 and float(pin.omega) == omega]
        constraints = []
        V = cp.Variable((dimension, dimension), hermitian=True)
        constraints.append(V + 1j * Omega >> 0)
        used = 0
        for pin in pins:
            value = pin.value
            if isinstance(value, sp.Expr) and value.free_symbols:
                continue                            # free symbol: unconstrained
            value = complex(sp.N(value)) if isinstance(value, sp.Expr) else complex(value)
            if pin.form is not None:
                vector = np.asarray(pin.form, dtype=float)
                constraints.append(cp.real(cp.quad_form(vector, V)) == value.real)
            elif pin.part == "real":
                constraints.append(cp.real(V[pin.row, pin.col]) == value.real)
            elif pin.part == "imag":
                constraints.append(cp.imag(V[pin.row, pin.col]) == value.imag)
            else:
                constraints.append(V[pin.row, pin.col] == value)
            used += 1
        if used == 0:
            continue

        problem = cp.Problem(cp.Minimize(0), constraints)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                problem.solve(solver=solver)
        except Exception as error:                  # solver blew up: inconclusive
            details["solver_status"][omega] = "error: %s" % error
            continue
        details["checked_frequencies"].append(omega)
        details["solver_status"][omega] = problem.status
        if problem.status in ("infeasible", "infeasible_inaccurate"):
            # the Farkas dual of an infeasible feasibility SDP is the certificate
            detail = {"reason": REASON_FIT_RANGE, "conclusive": True, "omega": omega,
                      "status": problem.status, "test": "uncertainty_relation"}
            detail.update(details)
            if verbose:
                print("target violates sigma + i Omega >= 0 at Omega = %g" % omega)
            return True, detail
    details["reason"] = "no_certificate"
    details["conclusive"] = False
    return False, details


# ---------------------------------------------------------------------------
# rung 1b -- a passive graph cannot leave the vacuum
# ---------------------------------------------------------------------------

def _is_passive(graph, space):
    """No anomalous element anywhere: no on-site and no two-mode squeezing."""
    for value, (kind, i, j) in zip(np.asarray(graph), space.slots):
        if value == NO_COUPLING:
            continue
        if kind in (SLOT_ONSITE_SQUEEZING, SLOT_TWO_MODE_SQUEEZING):
            return False
    return True


def certify_passive_range(optimizer, graph, tolerance=1.0e-9):
    """Graph-dependent range certificate for a passive device.

    A network built only from beam-splitters and detunings implements a passive
    (number-conserving) Gaussian channel.  Fed with vacuum on every input --
    signal ports, auxiliaries and intrinsic-loss channels alike -- its output is
    vacuum at *every* frequency, for *every* value of the couplings: the
    anomalous block of ``H_BdG`` is empty, so no parameter choice can populate
    the anomalous block of ``sigma_out``.  Any pin that vacuum does not already
    satisfy is therefore unreachable for this graph, whatever the optimiser
    does.

    Only fires when the declared input covariance is vacuum; a squeezed input
    makes the reachable set bigger and the argument no longer applies.
    """
    import sympy as sp

    space = optimizer.space
    if not _is_passive(graph, space):
        return False, {"reason": "not_passive", "conclusive": False}

    sigma_in = np.asarray(optimizer.oracle.sigma_in_signal)
    if not np.allclose(sigma_in, np.eye(sigma_in.shape[0]), atol=1e-12):
        return False, {"reason": "non_vacuum_input", "conclusive": False}

    num_ports = optimizer.num_ports
    vacuum = np.eye(2 * num_ports)
    for pin in optimizer.target.pins:
        value = pin.value
        if isinstance(value, sp.Expr) and value.free_symbols:
            continue
        value = complex(sp.N(value)) if isinstance(value, sp.Expr) else complex(value)
        if pin.order > 0:
            reached = 0.0                            # vacuum is frequency independent
        elif pin.form is not None:
            vector = np.asarray(pin.form, dtype=float)
            reached = float(vector @ vacuum @ vector)
        else:
            reached = complex(vacuum[pin.row, pin.col])
        if abs(complex(reached) - value) > tolerance:
            return True, {"reason": REASON_PASSIVE, "conclusive": True,
                          "test": "passive_maps_vacuum_to_vacuum",
                          "pin": str(pin), "reachable_value": complex(reached),
                          "pinned_value": value}
    return False, {"reason": "vacuum_satisfies_all_pins", "conclusive": False}


def certify_fit_infeasible(optimizer, graph, solver=None, verbose=False, cache=None):
    """Fit-range certificate (Sec. 6): is the target outside this graph's range?

    Runs the cheap graph-dependent passive test first, then the graph-
    independent physicality SDP.  The latter is a statement about the *target*,
    so its result is memoised in ``cache`` and the SDP is solved once per run.
    Returns ``(fired, detail)``.
    """
    fired, detail = certify_passive_range(optimizer, graph)
    if fired:
        return True, detail
    passive_detail = detail

    if cache is not None and "target_unphysical" in cache:
        fired, detail = cache["target_unphysical"]
        detail = dict(detail)
    else:
        fired, detail = certify_target_unphysical(
            optimizer.target, optimizer.num_ports, solver=solver, verbose=verbose)
        if cache is not None:
            cache["target_unphysical"] = (fired, detail)
    detail["passive_test"] = passive_detail
    return fired, detail


# ---------------------------------------------------------------------------
# rung 2 -- no stable point exists
# ---------------------------------------------------------------------------

def pbh_non_stabilizable(A, C, margin=0.0, tolerance=None):
    """Popov-Belevitch-Hautus test for stabilizability of ``(A, C)``.

    ``A`` generates the linear dynamics and the rows of ``C`` are the
    dissipative channels (jump operators) that damp it.  If for some eigenvalue
    ``lambda`` with ``Re lambda >= -margin``

        rank [ A - lambda I ; C ]  <  n ,

    then the corresponding mode is invisible to every dissipator -- a *dark
    mode*.  No choice of the remaining parameters can damp it, so the family
    contains no Hurwitz point: this is a genuine universal statement, which is
    why it may certify.

    Returns ``(non_stabilizable, detail)``.
    """
    A = np.asarray(A)
    C = np.atleast_2d(np.asarray(C))
    n = A.shape[0]
    values = np.linalg.eigvals(A)
    if tolerance is None:
        tolerance = max(n, C.shape[0]) * np.finfo(float).eps * max(
            1.0, float(np.linalg.norm(A, 2)))

    worst = None
    for lam in values:
        if np.real(lam) < -margin:
            continue
        stacked = np.vstack([A - lam * np.eye(n), C])
        singular = np.linalg.svd(stacked, compute_uv=False)
        smallest = float(singular[-1]) if singular.size >= n else 0.0
        rank = int(np.sum(singular > tolerance))
        if worst is None or smallest < worst[1]:
            worst = (lam, smallest, rank)
        if rank < n:
            return True, {"eigenvalue": complex(lam), "rank": rank, "dimension": n,
                          "smallest_singular_value": smallest, "tolerance": tolerance}
    detail = {"rank_deficient": False, "tolerance": tolerance}
    if worst is not None:
        detail.update({"eigenvalue": complex(worst[0]),
                       "smallest_singular_value": worst[1], "rank": worst[2],
                       "dimension": n})
    return False, detail


def pbh_dark_mode(optimizer, graph, num_samples=5, rng=None):
    """PBH dark-mode test for one graph of this device family.

    The dissipators are the port couplings ``sqrt(kappa~_i)`` and the intrinsic
    losses ``sqrt(gamma_i)``, one channel per mode and Nambu component.  Note
    what this means for the present model: **every** mode of a coupled-mode
    device is, by construction, coupled to an input line (``kappa_i > 0``), so
    ``C`` has full rank and the test cannot fire.  It is implemented in full
    anyway, because it is the one Hurwitz certificate that is a real theorem
    rather than a stub, and it does fire for families that admit undamped
    internal modes (see :func:`pbh_non_stabilizable`, which takes ``(A, C)``
    directly).

    Returns ``(fired, detail)``; ``detail['conclusive']`` says whether the test
    applied at all.
    """
    rng = np.random.default_rng() if rng is None else rng
    param = optimizer.param
    free_idxs = param.free_indices(graph)
    oracle = optimizer.oracle
    if num_samples <= 0:
        return False, {"reason": "not_sampled", "conclusive": False}

    detail = {}
    for _ in range(int(num_samples)):
        x = np.zeros(param.num_variables)
        if len(free_idxs):
            x[free_idxs] = oracle.initial_guess(free_idxs, rng=rng)
        M = oracle.matrix(x)
        gamma = np.asarray(param.gamma(x))
        kappa = np.asarray(param.kappa_tilde(x))
        # channel matrix: one row per damped mode and Nambu component
        rates = np.concatenate([kappa * (1.0 + gamma), kappa * (1.0 + gamma)])
        C = np.diag(np.sqrt(np.abs(rates)))
        fired, detail = pbh_non_stabilizable(M, C)
        if not fired:
            detail["reason"] = "controllable_by_dissipation"
            detail["conclusive"] = False
            return False, detail
    detail["reason"] = REASON_PBH
    detail["conclusive"] = True
    return True, detail


def lyapunov_feasible(M, solver=None):
    """Is ``M`` Hurwitz, decided by the Lyapunov LMI rather than by ``eig``?

    Feasibility of ``M^H P + P M < 0, P > 0``.  A *check* on one matrix, not a
    certificate over a parameter family -- it is what the SOS hierarchy in
    :func:`sos_no_hurwitz` would have to quantify over.
    """
    if not has_cvxpy():
        return None
    import cvxpy as cp

    M = np.asarray(M)
    n = M.shape[0]
    P = cp.Variable((n, n), hermitian=True)
    constraints = [P >> np.eye(n),
                   np.conj(M).T @ P + P @ M << -1e-9 * np.eye(n)]
    problem = cp.Problem(cp.Minimize(0), constraints)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem.solve(solver=solver)
    except Exception:
        return None
    if problem.status in ("optimal", "optimal_inaccurate"):
        return True
    if problem.status in ("infeasible", "infeasible_inaccurate"):
        return False
    return None


def sos_no_hurwitz(optimizer, graph, num_samples=0, method="sampling", rng=None,
                   max_violation_success=1.0e-10):
    """**STUB.**  Certify that no point of the fit manifold is Hurwitz.

    The real construction encodes stability as the Lyapunov LMI
    ``M^T P + P M < 0, P > 0`` and the fit as polynomial equalities ``r(x) = 0``,
    then runs a Lasserre/moment-SOS hierarchy; infeasibility at some level is
    the certificate.  ``method='sos'`` raises :class:`NotImplementedError` with
    that interface fixed.

    ``method='sampling'`` is the v1 stand-in: draw feasible points, look at the
    abscissa of each, and report what was seen.  All-unstable is evidence, not
    proof, so this **always returns** ``fired=False`` -- the graph becomes
    ``INVALID_DEFAULT`` and keeps its subtree alive.
    """
    if method == "sos":
        raise NotImplementedError(
            "moment-SOS certificate for Hurwitz-infeasibility is not implemented; "
            "encode M^T P + P M < 0, P > 0 together with r(x) = 0 and run a "
            "Lasserre hierarchy, then return the level at which it is infeasible")

    detail = {"reason": "interior_sampling_evidence", "conclusive": False,
              "num_feasible": 0, "num_stable": 0, "abscissas": []}
    if num_samples <= 0:
        return False, detail

    rng = np.random.default_rng() if rng is None else rng
    free_idxs = optimizer.param.free_indices(graph)
    _, infos = optimizer.oracle.repeated_optimize(
        free_idxs, num_tests=int(num_samples), interrupt_if_successful=False,
        check_stability=False, stability_search=False, rng=rng,
        max_violation_success=max_violation_success)
    feasible = [info for info in infos if info["loss_below_tolerance"]]
    detail["num_feasible"] = len(feasible)
    detail["abscissas"] = [info["max_real_eigenvalue"] for info in feasible]
    detail["num_stable"] = sum(1 for info in feasible if info["max_real_eigenvalue"] < 0)
    detail["evidence"] = bool(feasible) and detail["num_stable"] == 0
    return False, detail


def certify_no_hurwitz(optimizer, graph, num_samples=0, rng=None, verbose=False):
    """Stability-infeasibility certificate (Sec. 6): PBH first, then the stub.

    Returns ``(fired, detail)``.  ``fired`` can only come from PBH in v1.
    """
    fired, detail = pbh_dark_mode(optimizer, graph, rng=rng)
    if fired:
        if verbose:
            print("PBH dark mode: the graph is structurally non-stabilizable")
        return True, detail
    pbh_detail = detail

    fired, detail = sos_no_hurwitz(optimizer, graph, num_samples=num_samples, rng=rng)
    detail["pbh"] = pbh_detail
    return fired, detail
