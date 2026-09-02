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
from autogaussian.types import (
    REASON_BATH,
    REASON_FIT_RANGE,
    REASON_PASSIVE,
    REASON_PBH,
)

__all__ = [
    "certify_fit_infeasible",
    "certify_no_hurwitz",
    "certify_target_unphysical",
    "certify_passive_range",
    "certify_bath_unhosted",
    "certify_hot_purity_obstruction",
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


def _is_vacuum(matrix):
    matrix = np.asarray(matrix)
    return bool(np.allclose(matrix, np.eye(matrix.shape[0]), atol=1e-12))


def _port_of(index):
    """Which port a quadrature index belongs to."""
    return int(index) // 2


def _pin_value(pin):
    """The numeric value of a pin, or ``None`` if it still carries free symbols."""
    import sympy as sp

    value = pin.value
    if isinstance(value, sp.Expr):
        if value.free_symbols:
            return None
        return complex(sp.N(value))
    return complex(value)


def _passive_isotropic_violation(optimizer, tolerance):
    """Passive graph + baths without anomalous correlations: what is unreachable?

    A passive network has an empty anomalous block of ``H_BdG``, so ``S`` and
    ``N`` are block diagonal in Nambu space.  If no *input* carries anomalous
    correlations either (thermal baths have ``m = 0``), the anomalous block of
    ``sigma_out`` is zero at every frequency and for every parameter choice.
    That forces each monitored port block to be **isotropic**,

        V_jj(Omega) = c_j(Omega) * 1_2 ,   c_j real,

    with ``c_j`` free.  Anything the target asks for beyond that -- a nonzero
    ``x``/``p`` correlation inside a port, two different values for ``sigma_xx``
    and ``sigma_pp`` of the same port, or a squeezed eigenvalue pair -- is
    unreachable whatever the optimiser does.

    This is the thermal generalisation of the exact-vacuum argument: hot loss
    channels enlarge the reachable set (``c_j`` grows) but cannot tilt it.
    """
    # (a) squeezing asked for through a gauge-free spectrum constraint
    for constraint in getattr(optimizer.oracle, "constraints", []):
        eigenvalues = getattr(constraint, "eigenvalues", None)
        if eigenvalues is None:
            continue
        lo, hi = float(eigenvalues[0]), float(eigenvalues[1])
        if abs(hi - lo) > tolerance:
            return {"test": "passive_output_is_isotropic",
                    "constraint": str(constraint),
                    "reachable": "eigenvalues equal (isotropic block)",
                    "requested": (lo, hi)}

    # (b) squeezing asked for through the pins themselves
    diagonal = {}
    for pin in optimizer.target.pins:
        if pin.form is not None or pin.order > 0:
            continue
        value = _pin_value(pin)
        if value is None:
            continue
        port_row, port_col = _port_of(pin.row), _port_of(pin.col)
        if port_row != port_col:
            continue                       # cross-port entries stay free
        if pin.row != pin.col:
            if abs(value) > tolerance:     # V_xp inside a port is identically 0
                return {"test": "passive_has_no_intra_port_xp_correlation",
                        "pin": str(pin), "reachable_value": 0.0,
                        "pinned_value": value}
            continue
        key = (port_row, float(pin.omega))
        previous = diagonal.get(key)
        if previous is not None and abs(previous[0] - value) > tolerance:
            return {"test": "passive_port_block_is_isotropic",
                    "pin": str(pin), "other_pin": previous[1],
                    "pinned_values": (previous[0], value)}
        diagonal[key] = (value, str(pin))
    return None


def certify_passive_range(optimizer, graph, tolerance=1.0e-9):
    """Graph-dependent range certificate for a passive device.

    A network built only from beam-splitters and detunings implements a passive
    (number-conserving) Gaussian channel.  Two rungs, depending on what the
    channels are fed with:

    * **All inputs vacuum** -- signal ports, auxiliaries *and* intrinsic-loss
      channels.  The output is vacuum at every frequency for every value of the
      couplings, so any pin vacuum does not already satisfy is unreachable.
    * **Signal ports vacuum, some loss channel thermal.**  The output is no
      longer vacuum -- a hot bath heats the ports -- but it still carries no
      anomalous correlations, so every port block stays isotropic
      ``c_j * 1_2``.  Squeezing (a split eigenvalue pair, or an intra-port
      ``x``/``p`` correlation) remains unreachable; see
      :func:`_passive_isotropic_violation`.

    A *squeezed* input makes the reachable set genuinely bigger and neither
    argument applies.
    """
    space = optimizer.space
    if not _is_passive(graph, space):
        return False, {"reason": "not_passive", "conclusive": False}

    sigma_in = np.asarray(optimizer.oracle.sigma_in_signal)
    if not _is_vacuum(sigma_in):
        return False, {"reason": "non_vacuum_input", "conclusive": False}

    sigma_noise = np.asarray(getattr(optimizer.oracle, "sigma_in_noise",
                                     np.eye(sigma_in.shape[0])))
    num_modes = sigma_noise.shape[0] // 2
    anomalous = sigma_noise[:num_modes, num_modes:]
    if not np.allclose(anomalous, 0.0, atol=1e-12):
        # a squeezed loss channel can populate the anomalous output block
        return False, {"reason": "squeezed_noise_channel", "conclusive": False}

    if not _is_vacuum(sigma_noise):
        # hot (but not squeezed) loss channels: the weaker isotropy argument
        detail = _passive_isotropic_violation(optimizer, tolerance)
        if detail is not None:
            return True, dict(detail, reason=REASON_PASSIVE, conclusive=True)
        return False, {"reason": "isotropic_output_satisfies_all_pins",
                       "conclusive": False}

    num_ports = optimizer.num_ports
    vacuum = np.eye(2 * num_ports)
    for pin in optimizer.target.pins:
        value = _pin_value(pin)
        if value is None:
            continue
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
    # the pins are all satisfied by vacuum, but a constraint may still ask for
    # squeezing (the gauge-free spectrum pins of the hot-channel target)
    detail = _passive_isotropic_violation(optimizer, tolerance)
    if detail is not None:
        return True, dict(detail, reason=REASON_PASSIVE, conclusive=True)
    return False, {"reason": "vacuum_satisfies_all_pins", "conclusive": False}


# ---------------------------------------------------------------------------
# rung 1c -- the declared bath has no channel to live on
# ---------------------------------------------------------------------------

def certify_bath_unhosted(optimizer, graph=None, tolerance=1.0e-12):
    """Fires when a hot bath is declared on a channel that does not exist.

    An intrinsic-loss channel enters the forward map through
    ``N = chi sqrt(gamma)``.  If mode ``k*`` has no *live* ``gamma_k`` -- the
    parametrisation freezes it at zero for the whole family, because
    ``intrinsic_losses`` is off for that mode -- then column ``k*`` of ``N`` is
    identically zero and the declared bath never reaches the device at all.
    No graph in the family hosts it, so the posed problem is not the problem
    being solved.

    This is a *configuration* statement, not a graph one: it fires for every
    graph or for none.  It is cheap and conclusive, and it is the reason the
    hot-channel target makes the hot mode's intrinsic loss a live variable.
    """
    oracle = optimizer.oracle
    occupations = np.asarray(getattr(oracle, "channel_occupations",
                                     np.zeros(optimizer.num_modes)))
    hot = np.where(occupations > tolerance)[0]
    if hot.size == 0:
        return False, {"reason": "no_bath_declared", "conclusive": False}

    live = np.asarray(optimizer.param.intrinsic_losses, dtype=bool)
    unhosted = [int(k) for k in hot if not live[k]]
    if unhosted:
        return True, {"reason": REASON_BATH, "conclusive": True,
                      "test": "declared_bath_has_no_live_loss_channel",
                      "channels": unhosted,
                      "occupations": [float(occupations[k]) for k in unhosted]}
    return False, {"reason": "bath_is_hosted", "conclusive": False}


# ---------------------------------------------------------------------------
# rung 1d -- a hot channel that reaches the monitored port forbids purity
# ---------------------------------------------------------------------------

def certify_hot_purity_obstruction(optimizer, graph=None, tolerance=1.0e-12):
    """Purity at the monitored port is unreachable if the hot bath can reach it.

    Split the output covariance by input block.  With the loss channels at
    ``sigma_noise = 1 + 2 n e_k`` the monitored port block is

        sigma_out,jj = A + B ,
        A = ( S sigma_signal S^dag + N N^dag )_jj ,
        B = 2 n [ N e_k N^dag ]_jj .

    ``A`` is exactly the block the *same* device would emit with a cold bath, so
    it is a physical single-mode covariance and ``det A >= 1`` by the
    uncertainty principle.  ``B`` is positive semidefinite.  For 2x2 matrices

        det(A + B) = det A + det B + tr(A adj B) >= det A >= 1 ,

    with equality **iff** ``B = 0``, since ``A`` is positive definite and
    ``adj B = 0`` only for ``B = 0``.  Hence

        det sigma_out,jj = 1   <=>   N_{j,k}(Omega) = 0 ,

    i.e. *the purity floor under a hot bath is exactly the statement that the
    monitored port evades the hot channel*.  That is a theorem about this
    forward map, not a conjecture about the mechanism.

    It becomes an **obstruction** whenever the target itself forces
    ``N_{j,k} != 0``.  ``N = chi sqrt(gamma)`` and, for ``j != k``,
    ``S_{jk} = chi_{jk}``; so a pinned nonzero transmission ``|S_{jk}|^2 = t > 0``
    together with a floored loss ``gamma_k >= gamma_min > 0`` gives
    ``|N_{jk}|^2 = t gamma_k > 0`` at every parameter point.  No graph, at any
    mode count, can then hold ``det = 1``: the search is answering an
    unanswerable question and should be told so before it starts.

    Fires only when all four ingredients are declared: a hot channel, a purity
    floor on port ``j``, a positive lower bound on ``gamma_k``, and a positive
    lower bound on ``|S_{jk}|^2`` at the same frequency.
    """
    oracle = optimizer.oracle
    occupations = np.asarray(getattr(oracle, "channel_occupations",
                                     np.zeros(optimizer.num_modes)))
    hot = [int(k) for k in np.where(occupations > tolerance)[0]]
    if not hot:
        return False, {"reason": "no_bath_declared", "conclusive": False}

    constraints = list(getattr(oracle, "constraints", []))
    purity = [c for c in constraints
              if type(c).__name__ == "PurityFloor" and abs(c.value - 1.0) <= tolerance]
    if not purity:
        return False, {"reason": "no_purity_floor", "conclusive": False}

    loss_floor = {}
    for c in constraints:
        if type(c).__name__ == "MinimumIntrinsicLoss" and c.minimum > tolerance:
            loss_floor[int(c.mode)] = max(loss_floor.get(int(c.mode), 0.0),
                                          float(c.minimum))

    transmission = []
    for c in constraints:
        name = type(c).__name__
        if name == "TransmissionConstraint" and c.value > tolerance:
            transmission.append((int(c.port_out), int(c.port_in), float(c.value),
                                 float(c.omega), str(c)))
        elif name == "MinimumTransmission" and c.minimum > tolerance:
            transmission.append((int(c.port_out), int(c.port_in), float(c.minimum),
                                 float(c.omega), str(c)))

    for floor in purity:
        j = int(floor.port)
        for k in hot:
            if k == j or k not in loss_floor:
                continue
            for out, inp, value, omega, text in transmission:
                if out != j or inp != k or abs(omega - float(floor.omega)) > 1e-12:
                    continue
                return True, {
                    "reason": REASON_FIT_RANGE, "conclusive": True,
                    "test": "hot_channel_reaches_monitored_port",
                    "monitored_port": j, "hot_channel": k,
                    "occupation": float(occupations[k]),
                    "gamma_min": loss_floor[k],
                    "transmission": value,
                    "omega": float(floor.omega),
                    "bound": "|N_{j,k}|^2 >= %.6g > 0  =>  det sigma_out,jj > 1"
                             % (value * loss_floor[k]),
                    "forcing_constraint": text,
                }
    return False, {"reason": "hot_channel_not_forced_onto_port", "conclusive": False}


def certify_fit_infeasible(optimizer, graph, solver=None, verbose=False, cache=None):
    """Fit-range certificate (Sec. 6): is the target outside this graph's range?

    Runs the cheap graph-dependent passive test first, then the graph-
    independent physicality SDP.  The latter is a statement about the *target*,
    so its result is memoised in ``cache`` and the SDP is solved once per run.
    Returns ``(fired, detail)``.
    """
    fired, detail = certify_bath_unhosted(optimizer, graph)
    if fired:
        return True, detail
    bath_detail = detail

    fired, detail = certify_hot_purity_obstruction(optimizer, graph)
    if fired:
        return True, detail
    hot_detail = detail

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
    detail["bath_test"] = bath_detail
    detail["hot_purity_test"] = hot_detail
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
