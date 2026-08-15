"""
STABILITY PRIMITIVES (Sec. 5 of the algorithm flow).

The quantity that decides whether a device exists at all is the **spectral
abscissa** of the dynamical matrix,

    alpha(M) = max_k Re lambda_k(M) ,        stable  <=>  alpha(M) < 0 .

Three facts drive the design of this module.

1.  ``alpha`` is *not* a smooth function of the parameters.  It is smooth only
    while the rightmost eigenvalue is simple and unique; it develops Lipschitz
    corners where two distinct eigenvalues tie for rightmost, and it is not
    even locally Lipschitz at an **exceptional point**, where two eigenvalues
    *and their eigenvectors* coalesce.

2.  Where ``alpha`` *is* smooth, its gradient is analytic and cheap.  For the
    rightmost eigenvalue ``lambda`` with right/left eigenvectors ``v, u``
    (``M v = lambda v``, ``u^H M = lambda u^H``),

        d Re lambda / dx_k = Re( u^H (dM/dx_k) v ) / (u^H v) .

    This is what :func:`abscissa_gradient` computes.  **Never autodiff through
    an eigendecomposition** to get it: the reverse-mode rule silently produces
    garbage exactly at the ties and exceptional points that matter here, and
    it hides the diagnostic that tells you you are near one.

3.  That diagnostic is the eigenvalue condition number
    ``cond(lambda) = 1 / |u^H v|`` (with ``||u|| = ||v|| = 1``).  It diverges
    like ``t^{-1/2}`` on approach to an exceptional point at distance ``t``.
    A large value is *not* a bug: the gradient genuinely blows up there.  The
    response is :func:`gradient_sampling` -- evaluate the gradient at a few
    random points of an epsilon-ball and take the minimum-norm element of their
    convex hull, which is the Clarke-subdifferential steepest-descent direction
    and stays bounded across the non-smoothness.
"""

import numpy as np
import scipy.optimize as sciopt

__all__ = [
    "spectral_abscissa",
    "rightmost_eigenpair",
    "eigenvalue_condition_number",
    "abscissa_gradient",
    "abscissa_and_gradient",
    "is_hurwitz",
    "rightmost_multiplicity",
    "gradient_sampling",
    "min_norm_convex_hull",
    "DEFAULT_CONDITION_LIMIT",
    "DEFAULT_TIE_TOLERANCE",
]

# above this eigenvalue condition number the analytic gradient is not trusted
DEFAULT_CONDITION_LIMIT = 1.0e6
# two rightmost eigenvalues closer than this in Re are treated as a tie
DEFAULT_TIE_TOLERANCE = 1.0e-8


def spectral_abscissa(M):
    """``alpha(M) = max_k Re eig(M)``.  Negative means the device is stable.

    A non-finite ``M`` -- which an optimiser line search does produce, by
    stepping a coupling to where ``exp`` overflows -- has abscissa ``+inf``:
    unstable, and the caller backs off.  Never a crash and never a silent NaN.
    """
    M = np.asarray(M)
    if not np.all(np.isfinite(M)):
        return np.inf
    try:
        values = np.linalg.eigvals(M)
    except np.linalg.LinAlgError:
        return np.inf
    real = np.real(values)
    if not np.all(np.isfinite(real)):
        return np.inf
    return float(np.max(real))


def is_hurwitz(M, delta=0.0):
    """``alpha(M) < -delta`` -- the Sec. 5 stability gate with margin.

    ``delta`` should not be pushed below ``cond(lambda) * eps * ||M||``; below
    that floor the sign of ``alpha`` is numerical noise.
    """
    return bool(spectral_abscissa(M) < -float(delta))


def _eigenpair(values, right, left_values, left_vectors, idx):
    lam = values[idx]
    v = right[:, idx]
    # left eigenvectors come from eig(M^H): if M^H w = mu w then w^H M = mu* w^H,
    # so the left eigenvector of lambda is the w whose conj(mu) matches lambda
    match = int(np.argmin(np.abs(np.conj(left_values) - lam)))
    u = left_vectors[:, match]
    v = v / np.linalg.norm(v)
    u = u / np.linalg.norm(u)
    overlap = np.vdot(u, v)                       # u^H v
    condition = np.inf if overlap == 0 else 1.0 / abs(overlap)
    return {
        "value": lam,
        "right": v,
        "left": u,
        "overlap": overlap,
        "condition": float(condition),
        "index": int(idx),
        "eigenvalues": values,
    }


def rightmost_eigenpair(M):
    """Rightmost eigenvalue of ``M`` with its right and left eigenvectors.

    Returns a dict with ``value``, ``right`` (``v``), ``left`` (``u``),
    ``overlap`` (``u^H v``, both vectors normalised), ``condition``
    (``1/|u^H v|``) and ``index``.
    """
    M = np.asarray(M)
    values, right = np.linalg.eig(M)
    left_values, left_vectors = np.linalg.eig(np.conj(M).T)
    idx = int(np.argmax(np.real(values)))
    return _eigenpair(values, right, left_values, left_vectors, idx)


def rightmost_eigenpairs(M, tie_tolerance=DEFAULT_TIE_TOLERANCE):
    """*Every* eigenpair whose real part ties for rightmost.

    Ties are not exotic here: the BdG structure of ``M`` forces the spectrum to
    be closed under complex conjugation, so a rightmost eigenvalue with
    ``Im lambda != 0`` is *always* accompanied by its conjugate at the same
    real part.  That tie is spurious -- both members give the *same* gradient of
    ``Re lambda`` -- which is exactly why the smoothness test in
    :func:`abscissa_and_gradient` compares the gradients of the tied
    eigenvalues instead of merely counting them.
    """
    M = np.asarray(M)
    if not np.all(np.isfinite(M)):
        return []
    try:
        values, right = np.linalg.eig(M)
        left_values, left_vectors = np.linalg.eig(np.conj(M).T)
    except np.linalg.LinAlgError:
        return []
    real = np.real(values)
    if not np.all(np.isfinite(real)):
        return []
    top = np.max(real)
    idxs = np.flatnonzero(real >= top - float(tie_tolerance))
    return [_eigenpair(values, right, left_values, left_vectors, idx) for idx in idxs]


def eigenvalue_condition_number(M):
    """``1/|u^H v|`` for the rightmost eigenvalue -- the exceptional-point
    detector.  Diverges as ``t^{-1/2}`` at distance ``t`` from an EP."""
    return rightmost_eigenpair(M)["condition"]


def rightmost_multiplicity(M, tolerance=DEFAULT_TIE_TOLERANCE):
    """How many eigenvalues share the rightmost real part (a Lipschitz corner
    of ``alpha`` when this is > 1 with *distinct* eigenvalues)."""
    values = np.linalg.eigvals(np.asarray(M))
    top = np.max(np.real(values))
    return int(np.sum(np.real(values) > top - tolerance))


def abscissa_gradient(M, dM, pair=None):
    """Analytic gradient of ``alpha`` (Sec. 5, biorthogonal form).

    Parameters
    ----------
    M : (n, n) array
    dM : (K, n, n) array
        ``dM[k] = dM/dx_k``.
    pair : dict, optional
        A previously computed :func:`rightmost_eigenpair` (avoids a second
        eigendecomposition).

    Returns
    -------
    (K,) real array   ``d Re lambda_max / dx_k = Re(u^H dM_k v) / (u^H v)``.

    The result is meaningless when ``pair["condition"]`` is huge or the
    rightmost eigenvalue is not unique -- use :func:`gradient_sampling` there.
    """
    pair = rightmost_eigenpair(M) if pair is None else pair
    u, v, overlap = pair["left"], pair["right"], pair["overlap"]
    dM = np.asarray(dM)
    if dM.ndim == 2:
        dM = dM[None]
    numerators = np.einsum("i,kij,j->k", np.conj(u), dM, v)
    if overlap == 0:
        # exactly on an exceptional point: the derivative of Re lambda is
        # genuinely unbounded there, so say so rather than emitting a NaN
        return np.where(np.real(numerators) < 0, -np.inf, np.inf)
    return np.real(numerators / overlap)


def abscissa_and_gradient(M, dM, condition_limit=DEFAULT_CONDITION_LIMIT,
                          tie_tolerance=DEFAULT_TIE_TOLERANCE,
                          gradient_tolerance=1.0e-6):
    """``(alpha, grad_alpha, diagnostics)`` from one pair of eigendecompositions.

    All eigenvalues tied for rightmost are differentiated.  If their gradients
    agree, the tie is spurious (the conjugate-pair tie forced by the BdG
    structure) and ``alpha`` is smooth here: the common gradient is returned
    and ``trustworthy`` is ``True``.  If they disagree, this is a genuine
    Lipschitz corner and the minimum-norm element of their convex hull -- the
    Clarke steepest-descent direction -- is returned with ``trustworthy``
    ``False``, so the caller may widen the search with
    :func:`gradient_sampling`.  An ill-conditioned pair (near an exceptional
    point) is always untrustworthy: there the gradient genuinely diverges.
    """
    pairs = rightmost_eigenpairs(M, tie_tolerance=tie_tolerance)
    if not pairs:
        # M is not finite (an optimiser stepped a coupling into overflow):
        # report an unbounded abscissa and no descent information at all
        dimension = np.asarray(dM).shape[0] if np.asarray(dM).ndim == 3 else 1
        return np.inf, np.zeros(dimension), {
            "condition": np.inf, "multiplicity": 0, "gradient_spread": np.inf,
            "consistent": False, "trustworthy": False, "nonfinite": True,
            "eigenvalue": np.nan, "tied_gradients": np.zeros((0, dimension)),
        }
    alpha = float(np.max([np.real(p["value"]) for p in pairs]))
    gradients = np.array([abscissa_gradient(M, dM, pair=p) for p in pairs])
    conditions = [p["condition"] for p in pairs]

    spread = 0.0
    if len(gradients) > 1:
        spread = float(np.max(np.linalg.norm(gradients - gradients[0], axis=1)))
    scale = 1.0 + float(np.max(np.linalg.norm(gradients, axis=1)))
    consistent = spread <= gradient_tolerance * scale

    if consistent:
        grad = gradients.mean(axis=0)
    else:
        grad = min_norm_convex_hull(gradients)

    trustworthy = bool(consistent
                       and np.max(conditions) < condition_limit
                       and np.all(np.isfinite(grad)))
    diagnostics = {
        "condition": float(np.max(conditions)),
        "multiplicity": len(pairs),
        "gradient_spread": spread,
        "consistent": bool(consistent),
        "trustworthy": trustworthy,
        "eigenvalue": pairs[int(np.argmax([np.real(p["value"]) for p in pairs]))]["value"],
        "tied_gradients": gradients,
    }
    return alpha, grad, diagnostics


# ---------------------------------------------------------------------------
# gradient sampling (Sec. 5: ties and exceptional points)
# ---------------------------------------------------------------------------

def min_norm_convex_hull(gradients):
    """Minimum-norm element of the convex hull of a set of gradients.

    Solves ``min_w || G^T w ||^2`` over the simplex ``w >= 0, sum w = 1``.
    This is the Clarke steepest-descent direction of a non-smooth function
    whose generalised gradient is (approximated by) the hull of ``gradients``;
    it is what keeps a step bounded where a single gradient diverges.
    """
    G = np.atleast_2d(np.asarray(gradients, dtype=float))
    num, dim = G.shape
    if num == 1:
        return G[0].copy()

    gram = G @ G.T
    scale = np.max(np.abs(gram))
    if scale > 0:
        gram = gram / scale

    def objective(w):
        return float(w @ gram @ w), 2.0 * (gram @ w)

    w0 = np.full(num, 1.0 / num)
    result = sciopt.minimize(
        objective, w0, jac=True, method="SLSQP",
        bounds=[(0.0, 1.0)] * num,
        constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
                      "jac": lambda w: np.ones_like(w)},),
        options={"maxiter": 200, "ftol": 1e-12},
    )
    w = np.clip(result.x, 0.0, None)
    total = w.sum()
    w = w / total if total > 0 else w0
    return G.T @ w


def gradient_sampling(matrix_func, jacobian_func, x, epsilon=1.0e-4,
                      num_samples=None, rng=None, include_centre=True):
    """Bounded descent direction for ``alpha`` across a non-smooth point.

    Evaluates the analytic gradient at ``num_samples`` random points of the
    ``epsilon``-ball around ``x`` and returns the minimum-norm element of their
    convex hull (Burke-Lewis-Overton gradient sampling).

    Parameters
    ----------
    matrix_func : callable ``x -> M(x)``
    jacobian_func : callable ``x -> dM/dx`` with shape ``(K, n, n)``
    epsilon : float
        Ball radius.  Must be wide enough to *straddle* the non-smoothness:
        near an exceptional point pick it larger than the distance to the EP.
    num_samples : int
        Defaults to ``2 * K + 2``, the usual "more than the dimension" rule.
    include_centre : bool
        Also sample the gradient at ``x`` itself.

    Returns
    -------
    (grad, info)  where ``info`` holds the per-sample condition numbers and the
    number of samples that were discarded as non-finite.
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(x, dtype=float)
    dimension = x.size
    if num_samples is None:
        num_samples = 2 * dimension + 2

    points = [x] if include_centre else []
    for _ in range(int(num_samples)):
        direction = rng.normal(size=dimension)
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        radius = epsilon * rng.random() ** (1.0 / max(dimension, 1))
        points.append(x + radius * direction / norm)

    gradients = []
    conditions = []
    discarded = 0
    for point in points:
        M = np.asarray(matrix_func(point))
        pair = rightmost_eigenpair(M)
        conditions.append(pair["condition"])
        grad = abscissa_gradient(M, np.asarray(jacobian_func(point)), pair=pair)
        if not np.all(np.isfinite(grad)):
            discarded += 1
            continue
        gradients.append(grad)

    if not gradients:
        return np.zeros(dimension), {"conditions": conditions, "discarded": discarded,
                                     "num_used": 0}
    grad = min_norm_convex_hull(gradients)
    return grad, {"conditions": conditions, "discarded": discarded,
                  "num_used": len(gradients), "max_condition": float(np.max(conditions))}
