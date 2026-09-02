"""
VALIDITY ORACLE (Sec. 4 + Sec. 5 of the algorithm flow).

    function ORACLE(graph, spec):
        x0 = MINIMIZE(fit_loss, n_rep restarts)      # reach the target manifold
        if fit_loss(x0) > tol:    return INVALID_PROV
        if alpha(M(x0)) < -delta: return VALID, x0   # common case: one eig call
        x* = constrained_stability_search(seed=x0)   # min alpha s.t. fit <= tol
        if alpha(M(x*)) < 0:      return VALID, x*
        return INVALID_PROV                          # provisional, never certified

Two properties of that pseudocode are load bearing.

**The fit is a hard constraint, the stability is the objective.**  The raw
abscissa ``alpha`` must never be added to the fit loss: minimising
``fit + rho*alpha`` buys stability by moving off the target (it under-squeezes),
because ``alpha`` keeps decreasing after the device is already stable.  Only the
*violation* may enter, through the hinge ``max(0, alpha + delta)^2``, whose
zero-set is exactly the stable region -- so the weight ``lam`` shapes the path
but not the solution.

**The oracle never certifies INVALID.**  Failing to find a witness is an
existential failure of an optimiser, not a proof that no witness exists.  The
certified verdicts are produced only by :mod:`autogaussian.certificates`, via
:func:`autogaussian.search.escalate`.
"""

import numpy as np
import jax
import jax.numpy as jnp
import scipy.optimize as sciopt
import sympy as sp

from autogaussian.constraints import ForwardContext
from autogaussian.forward import dynamical_matrix, response_matrices
from autogaussian.nambu import (
    nambu_to_quadrature,
    quadrature_matrix,
    stack_input_covariance,
    vacuum_covariance,
)
from autogaussian.parametrization import (
    VAR_ABS,
    VAR_LOG_KAPPA,
    VAR_LOSS,
    VAR_PHASE,
    VAR_USER,
)
from autogaussian.stability import (
    DEFAULT_CONDITION_LIMIT,
    abscissa_and_gradient,
    gradient_sampling,
    spectral_abscissa,
)
from autogaussian.target import PART_FULL, PART_IMAG, PART_REAL
from autogaussian.types import REASON_FIT, REASON_HURWITZ, Verdict

jax.config.update("jax_enable_x64", True)

__all__ = ["CovarianceOracle", "VALID", "INVALID"]

VALID = True
INVALID = False

INIT_RANGES_DEFAULT = {
    VAR_ABS: (-1.0, 1.0),
    VAR_PHASE: (-np.pi, np.pi),
    VAR_LOSS: (1.0e-3, 1.0),
    VAR_LOG_KAPPA: (-0.7, 0.7),
    VAR_USER: (-np.pi, np.pi),
}

# Search box for the gradient-free optimiser.  This must be *wider* than the
# initial-guess ranges: those only say where to start a local descent from,
# whereas a particle swarm can never leave its box.  Reusing the init ranges
# here silently excludes solutions -- e.g. flat-band devices need
# kappa~ ~ 10, i.e. log kappa~ ~ 2.3, far outside the init range of +-0.7.
PSO_RANGES_DEFAULT = {
    VAR_ABS: (-3.0, 3.0),
    VAR_PHASE: (-np.pi, np.pi),
    VAR_LOSS: (0.0, 2.0),
    VAR_LOG_KAPPA: (-5.0, 5.0),
    VAR_USER: (-2 * np.pi, 2 * np.pi),
}


class CovarianceOracle:
    """Loss, continuous optimiser and stability search for one problem spec.

    Parameters
    ----------
    parametrization : :class:`autogaussian.parametrization.Parametrization`
    target : :class:`autogaussian.target.CovarianceTarget`
    sigma_in_signal : (2N, 2N) array, optional
        Declared input covariance of the signal channels (vacuum if omitted).
        *Declared, not optimised* (Sec. 1.2).
    sigma_in_noise : (2N, 2N) array or callable, optional
        Declared input covariance of the *intrinsic-loss* channels (vacuum if
        omitted).  A thermal block here is a hot bath on the corresponding
        channel; it reaches ``sigma_out`` only through ``N sigma_in_noise
        N^dag`` (App. B.3(h)).  Also declared, not optimised.
    constraints : sequence of :class:`autogaussian.constraints.BaseConstraint`
        Extra equality constraints ``f_j = 0``; part of the **fit**, i.e. of the
        hard constraint, not of the objective.
    stability_margin : float
        ``delta`` of Sec. 5: the margin the stability *search* aims for, and the
        floor below which the sign of ``alpha`` is numerical noise.  Do not push
        it below ``cond(lambda) * eps_machine * ||M||``.  Acceptance of a
        witness is ``alpha < 0``; ``delta`` decides when the search stops and
        when a fit that already landed stable is good enough to skip it.
    hinge_weight : float
        ``lam`` of the hinge used by the constrained stability search.  It
        shapes the path only -- the zero-set of the hinge is the stable region
        for any positive ``lam``.
    init_ranges, pso_ranges : dict
        Per variable-type initial-guess / swarm-box ranges.
    """

    def __init__(
        self,
        parametrization,
        target,
        sigma_in_signal=None,
        sigma_in_noise=None,
        constraints=(),
        stability_margin=1.0e-3,
        hinge_weight=1.0,
        init_ranges=None,
        pso_ranges=None,
        condition_limit=DEFAULT_CONDITION_LIMIT,
        stability_weight=None,
    ):
        self.param = parametrization
        self.target = target
        self.constraints = list(constraints)
        self.stability_margin = float(stability_margin)
        # ``stability_weight`` is the pre-two-library name of the hinge weight
        self.hinge_weight = float(hinge_weight if stability_weight is None
                                  else stability_weight)
        self.condition_limit = float(condition_limit)
        self.init_ranges = dict(INIT_RANGES_DEFAULT)
        if init_ranges:
            self.init_ranges.update(init_ranges)
        self.pso_ranges = dict(PSO_RANGES_DEFAULT)
        if pso_ranges:
            self.pso_ranges.update(pso_ranges)

        self.num_modes = self.param.num_modes
        self.num_ports = self.param.num_ports
        if sigma_in_signal is None:
            sigma_in_signal = vacuum_covariance(self.num_modes)
        elif callable(sigma_in_signal):
            sigma_in_signal = sigma_in_signal(self.num_modes)
        self.sigma_in_signal = jnp.asarray(sigma_in_signal, dtype=jnp.complex128)
        if sigma_in_noise is None:
            sigma_in_noise = vacuum_covariance(self.num_modes)
        elif callable(sigma_in_noise):
            sigma_in_noise = sigma_in_noise(self.num_modes)
        self.sigma_in_noise = jnp.asarray(sigma_in_noise, dtype=jnp.complex128)
        self.sigma_in_total = stack_input_covariance(
            self.sigma_in_signal, self.num_modes, sigma_noise=self.sigma_in_noise)

        self._compile()

    # ------------------------------------------------------------------ #
    # forward map wrappers
    # ------------------------------------------------------------------ #

    @property
    def channel_occupations(self):
        """Thermal occupation ``n_k`` read back off the declared noise block.

        ``n_k = (sigma_in_noise[k, k] - 1)/2``; all zeros for the cold (vacuum)
        default.  Certificates use it to notice that a bath was declared.
        """
        diagonal = np.real(np.diag(np.asarray(self.sigma_in_noise)))[: self.num_modes]
        return 0.5 * (diagonal - 1.0)

    @property
    def has_hot_channel(self):
        return bool(np.any(self.channel_occupations > 0.0))


    def _thetas_full(self, x):
        thetas = self.param.thetas(x)
        pad = self.num_modes - self.num_ports
        if pad > 0:
            thetas = jnp.concatenate([thetas, jnp.zeros(pad)])
        return thetas

    def _covariance(self, x, Omega):
        """Full ``2N x 2N`` quadrature covariance at one frequency."""
        H, gamma, kappa_tilde, _ = self.param.unpack(x)
        S, N = response_matrices(H, gamma, kappa_tilde, Omega)
        S_cal = jnp.concatenate([S, N], axis=1)
        sigma_out = S_cal @ self.sigma_in_total @ jnp.conj(S_cal).T
        W = quadrature_matrix(self._thetas_full(x), self.num_modes)
        return nambu_to_quadrature(sigma_out, W)

    def _responses(self, x, Omega):
        H, gamma, kappa_tilde, _ = self.param.unpack(x)
        S, N = response_matrices(H, gamma, kappa_tilde, Omega)
        S_cal = jnp.concatenate([S, N], axis=1)
        sigma_out = S_cal @ self.sigma_in_total @ jnp.conj(S_cal).T
        W = quadrature_matrix(self._thetas_full(x), self.num_modes)
        return S, N, nambu_to_quadrature(sigma_out, W)

    def _covariance_derivative(self, x, Omega, order):
        func = lambda w: self._covariance(x, w)
        for _ in range(order):
            func = jax.jacfwd(func)
        return func(Omega)

    def _dynamical_matrix(self, x):
        H, gamma, kappa_tilde, _ = self.param.unpack(x)
        return dynamical_matrix(H, gamma, kappa_tilde)

    # ------------------------------------------------------------------ #
    # loss assembly -- the FIT ONLY (Sec. 9, invariant 1)
    # ------------------------------------------------------------------ #

    def _compile(self):
        target = self.target
        pins = target.pins
        if len(pins) == 0:
            raise ValueError("the target has no pinned entries")

        self.omegas = np.unique(np.array([pin.omega for pin in pins], dtype=float))
        omega_of = {omega: idx for idx, omega in enumerate(self.omegas)}

        # quadratic-form pins (nullifiers, Duan sums, ...)
        form_pins = [pin for pin in pins if pin.form is not None]
        self._form_group = None
        if form_pins:
            self._form_group = {
                "k": jnp.array([omega_of[pin.omega] for pin in form_pins], dtype=int),
                "v": jnp.asarray(np.array([pin.form for pin in form_pins], dtype=float)),
                "w": jnp.array([pin.weight for pin in form_pins], dtype=float),
                "pins": form_pins,
            }

        # order-0 pins, grouped by which part of the entry is compared
        self._groups = {}
        for part in (PART_FULL, PART_REAL, PART_IMAG):
            sel = [pin for pin in pins
                   if pin.form is None and pin.order == 0 and pin.part == part]
            if sel:
                self._groups[part] = {
                    "k": jnp.array([omega_of[pin.omega] for pin in sel], dtype=int),
                    "row": jnp.array([pin.row for pin in sel], dtype=int),
                    "col": jnp.array([pin.col for pin in sel], dtype=int),
                    "w": jnp.array([pin.weight for pin in sel], dtype=float),
                    "pins": sel,
                }

        # derivative pins, grouped by (order, omega)
        self._derivative_groups = []
        orders = sorted(set(pin.order for pin in pins if pin.order > 0))
        for order in orders:
            for omega in sorted(set(pin.omega for pin in pins if pin.order == order)):
                sel = [pin for pin in pins if pin.order == order and pin.omega == omega]
                self._derivative_groups.append({
                    "order": order,
                    "omega": float(omega),
                    "row": jnp.array([pin.row for pin in sel], dtype=int),
                    "col": jnp.array([pin.col for pin in sel], dtype=int),
                    "w": jnp.array([pin.weight for pin in sel], dtype=float),
                    "part": [pin.part for pin in sel],
                    "pins": sel,
                })

        # target values (may depend on free symbols of the target)
        symbols = self.param.target_symbols
        self._value_funcs = {}
        for key, group in list(self._groups.items()):
            group["values"] = self._make_value_function([pin.value for pin in group["pins"]],
                                                        symbols)
        for group in self._derivative_groups:
            group["values"] = self._make_value_function([pin.value for pin in group["pins"]],
                                                        symbols)
        if self._form_group is not None:
            self._form_group["values"] = self._make_value_function(
                [pin.value for pin in self._form_group["pins"]], symbols)

        omegas = jnp.asarray(self.omegas)
        self._omegas_jnp = omegas

        def residual(x):
            """Stacked real residual vector ``r(x)`` of the fit (App. C).

            Complex residuals contribute their real and imaginary parts as
            separate rows, each already multiplied by ``sqrt(weight)``, so that
            ``fit_loss = 0.5 * ||r||^2`` exactly.
            """
            responses = jax.vmap(lambda w: self._responses(x, w))(omegas)
            S_all, N_all, V_all = responses
            parts = []

            for part, group in self._groups.items():
                actual = V_all[group["k"], group["row"], group["col"]]
                wanted = group["values"](x)
                root = jnp.sqrt(group["w"])
                if part == PART_REAL:
                    parts.append(root * (jnp.real(actual) - jnp.real(wanted)))
                elif part == PART_IMAG:
                    parts.append(root * (jnp.imag(actual) - jnp.imag(wanted)))
                else:
                    delta = actual - wanted
                    parts.append(root * jnp.real(delta))
                    parts.append(root * jnp.imag(delta))

            if self._form_group is not None:
                group = self._form_group
                vectors = group["v"]                        # (M, 2P)
                num_ports = self.num_ports
                blocks = V_all[group["k"], : 2 * num_ports, : 2 * num_ports]
                actual = jnp.real(jnp.einsum("ma,mab,mb->m", vectors, blocks, vectors))
                parts.append(jnp.sqrt(group["w"]) * (actual - jnp.real(group["values"](x))))

            for group in self._derivative_groups:
                dV = self._covariance_derivative(x, group["omega"], group["order"])
                actual = dV[group["row"], group["col"]]
                wanted = group["values"](x)
                root = jnp.sqrt(group["w"])
                if all(part == PART_REAL for part in group["part"]):
                    parts.append(root * (jnp.real(actual) - jnp.real(wanted)))
                elif all(part == PART_IMAG for part in group["part"]):
                    parts.append(root * (jnp.imag(actual) - jnp.imag(wanted)))
                else:
                    delta = actual - wanted
                    parts.append(root * jnp.real(delta))
                    parts.append(root * jnp.imag(delta))

            if self.constraints:
                H, gamma, kappa_tilde, thetas = self.param.unpack(x)
                ctx = ForwardContext(
                    x=x, H=H, gamma=gamma, kappa_tilde=kappa_tilde, thetas=thetas,
                    omegas=self.omegas, S=S_all, N=N_all, V=V_all,
                    num_modes=self.num_modes, num_ports=self.num_ports,
                )
                values = jnp.concatenate([jnp.atleast_1d(c(ctx)) for c in self.constraints])
                parts.append(jnp.real(values))
                parts.append(jnp.imag(values))

            return jnp.concatenate([jnp.atleast_1d(p) for p in parts])

        def loss(x):
            r = residual(x)
            return 0.5 * jnp.sum(r ** 2)

        self._residual = residual
        self._loss = loss
        self.residual_func = jax.jit(residual)
        self.residual_jacobian = jax.jit(jax.jacfwd(residual))
        self.loss_func = jax.jit(loss)
        self.loss_and_grad = jax.jit(jax.value_and_grad(loss))

        # dM/dx for the analytic abscissa gradient (Sec. 5).  Differentiating
        # the *matrix* is safe; differentiating its eigenvalues is not, which
        # is why the eigen-part is done by hand in autogaussian.stability.
        self._matrix_func = jax.jit(self._dynamical_matrix)
        self._matrix_jacobian = jax.jit(jax.jacfwd(self._dynamical_matrix))

    def _make_value_function(self, values, symbols):
        """Turn the pinned values (numbers and/or sympy expressions) into a
        jax function of the full variable vector."""
        has_symbol = any(isinstance(v, sp.Expr) and v.free_symbols for v in values)
        if not has_symbol:
            constant = jnp.asarray([complex(sp.N(v)) if isinstance(v, sp.Expr) else complex(v)
                                    for v in values])
            return lambda x: constant
        exprs = [sp.sympify(v) for v in values]
        lambdified = sp.lambdify(symbols, exprs, modules="jax")
        idxs = jnp.asarray(self.param.target_symbol_idx, dtype=int)

        def func(x):
            args = [x[idxs[k]] for k in range(len(symbols))]
            return jnp.asarray(lambdified(*args), dtype=jnp.complex128)

        return func

    # ------------------------------------------------------------------ #
    # evaluation helpers
    # ------------------------------------------------------------------ #

    def covariance(self, x, omega=0.0, ports_only=True):
        """``sigma_out(omega)`` in the quadrature basis for a parameter vector."""
        V = np.asarray(self._covariance(jnp.asarray(x), float(omega)))
        if ports_only:
            V = V[: 2 * self.num_ports, : 2 * self.num_ports]
        return V

    def spectrum(self, x, omegas, ports_only=True):
        return np.array([self.covariance(x, omega, ports_only) for omega in np.asarray(omegas)])

    def scattering(self, x, omega=0.0):
        H, gamma, kappa_tilde, _ = self.param.unpack(jnp.asarray(x))
        S, N = response_matrices(H, gamma, kappa_tilde, float(omega))
        return np.asarray(S), np.asarray(N)

    def fit_loss(self, x):
        return float(self.loss_func(jnp.asarray(x, dtype=float)))

    def residual(self, x):
        return np.asarray(self.residual_func(jnp.asarray(x, dtype=float)))

    def matrix(self, x):
        """The dimensionless dynamical matrix ``M~(x)`` as a numpy array."""
        return np.asarray(self._matrix_func(jnp.asarray(x, dtype=float)))

    def matrix_jacobian(self, x, free_idxs=None):
        """``dM~/dx_k`` with shape ``(K, 2N, 2N)`` (``K`` = free variables)."""
        jac = np.asarray(self._matrix_jacobian(jnp.asarray(x, dtype=float)))
        jac = np.moveaxis(jac, -1, 0)                      # (num_vars, 2N, 2N)
        if free_idxs is not None:
            jac = jac[np.asarray(free_idxs, dtype=int)]
        return jac

    def abscissa(self, x):
        """``alpha(M~(x)) = max Re eig`` -- negative means stable."""
        return spectral_abscissa(self.matrix(x))

    # kept under the old name: a lot of reporting code reads this key
    def max_real_eigenvalue(self, x):
        return self.abscissa(x)

    def is_stable(self, x, margin=None):
        margin = self.stability_margin if margin is None else float(margin)
        return bool(self.abscissa(x) < -margin)

    def abscissa_gradient(self, x, free_idxs, rng=None, epsilon=1.0e-4,
                          allow_sampling=True):
        """``(alpha, d alpha/dx_free, diagnostics)`` at ``x``.

        Uses the analytic biorthogonal formula, and falls back to gradient
        sampling when the rightmost eigenvalue is tied or ill-conditioned
        (Sec. 5 hazards (a) and (b)).
        """
        free_idxs = np.asarray(free_idxs, dtype=int)
        x = np.asarray(x, dtype=float)
        M = self.matrix(x)
        dM = self.matrix_jacobian(x, free_idxs)
        alpha, grad, diagnostics = abscissa_and_gradient(
            M, dM, condition_limit=self.condition_limit)
        if diagnostics["trustworthy"] or not allow_sampling:
            diagnostics["sampled"] = False
            return alpha, grad, diagnostics

        base = x.copy()

        def matrix_of(x_free):
            point = base.copy()
            point[free_idxs] = x_free
            return self.matrix(point)

        def jacobian_of(x_free):
            point = base.copy()
            point[free_idxs] = x_free
            return self.matrix_jacobian(point, free_idxs)

        grad, info = gradient_sampling(matrix_of, jacobian_of, x[free_idxs],
                                       epsilon=epsilon, rng=rng)
        diagnostics.update(info)
        diagnostics["sampled"] = True
        return alpha, grad, diagnostics

    # ------------------------------------------------------------------ #
    # continuous optimisation
    # ------------------------------------------------------------------ #

    def initial_guess(self, free_idxs, rng=None, init_ranges=None):
        rng = np.random.default_rng() if rng is None else rng
        ranges = dict(self.init_ranges)
        if init_ranges:
            ranges.update(init_ranges)
        guess = np.zeros(self.param.num_variables)
        for idx in free_idxs:
            low, high = ranges[self.param.types[idx]]
            guess[idx] = rng.uniform(low, high)
        return guess[np.asarray(free_idxs, dtype=int)]

    def _constrained_functions(self, free_idxs, fixed_values=None):
        free_idxs = np.asarray(free_idxs, dtype=int)
        num_variables = self.param.num_variables
        base = np.zeros(num_variables)
        if fixed_values:
            for idx, value in fixed_values.items():
                base[int(idx)] = float(value)

        def embed(x_free):
            x = base.copy()
            x[free_idxs] = np.asarray(x_free, dtype=float)
            return jnp.asarray(x)

        def value_and_grad(x_free):
            value, grad = self.loss_and_grad(embed(x_free))
            return float(value), np.asarray(grad)[free_idxs]

        return embed, value_and_grad

    def optimize(
        self,
        free_idxs,
        method="BFGS",
        max_violation_success=1.0e-10,
        rng=None,
        x0=None,
        solver_options=None,
        check_stability=True,
        stability_search=True,
        init_ranges=None,
        fixed_values=None,
    ):
        """One continuous run: fit first, then (if needed) the Sec. 5 search.

        Returns ``(success, info)``.  ``fixed_values`` maps variable indices to
        values held fixed (used to sweep a free target symbol during symbolic
        regression, Sec. 7.2).

        With ``check_stability`` the run only succeeds if the witness is
        Hurwitz; with ``stability_search`` a fit that landed unstable is handed
        to :meth:`constrained_stability_search` instead of being thrown away --
        that is the step which makes stability a *search constraint* rather
        than a post-hoc filter.
        """
        free_idxs = np.asarray(free_idxs, dtype=int)
        if fixed_values:
            free_idxs = np.array([i for i in free_idxs if i not in fixed_values], dtype=int)
        embed, value_and_grad = self._constrained_functions(free_idxs, fixed_values)
        if x0 is None:
            x0 = self.initial_guess(free_idxs, rng=rng, init_ranges=init_ranges)

        # gtol has to be tiny: the loss is quadratic in the residuals, so the
        # gradient is already small long before the target is actually met
        options = {"maxiter": 2000, "gtol": 1.0e-14}
        if method.upper() in ("L-BFGS-B", "TNC"):
            options = {"maxiter": 2000, "maxfun": 100000, "ftol": 0.0, "gtol": 0.0}
        if solver_options:
            options.update(solver_options)

        hybrid = method.lower() in ("pso+bfgs", "hybrid")
        if hybrid and len(free_idxs) > 0:
            # App. C: broadband / eigenvalue-sensitive losses are non-smooth,
            # so explore with the swarm first, then polish with the gradient
            # method -- the swarm locates the basin, BFGS reaches the 1e-10
            # tolerance the swarm alone cannot.
            swarm_options = dict(solver_options or {})
            swarm_x, _ = particle_swarm(
                lambda xf: value_and_grad(xf)[0], len(free_idxs),
                bounds=self._pso_bounds(free_idxs), rng=rng,
                **{k: v for k, v in swarm_options.items()
                   if k in ("num_particles", "num_iterations", "inertia",
                            "cognitive", "social")})
            x0 = swarm_x
            method = "BFGS"
            solver_options = {k: v for k, v in (solver_options or {}).items()
                              if k not in ("num_particles", "num_iterations",
                                           "inertia", "cognitive", "social")}
            options = {"maxiter": 2000, "gtol": 1.0e-14}
            if solver_options:
                options.update(solver_options)

        if len(free_idxs) == 0:
            # fully constrained graph (e.g. the empty graph): nothing to
            # optimise, just evaluate.  scipy cannot handle a 0-d problem.
            x_free = np.zeros(0)
            final = float(value_and_grad(x_free)[0])
            message = "no free variables"
            nit = 0
        elif method.lower() in ("pso", "particle_swarm"):
            x_free, final = particle_swarm(
                lambda xf: value_and_grad(xf)[0], len(free_idxs),
                bounds=self._pso_bounds(free_idxs), rng=rng, **(solver_options or {}))
            message = "particle swarm finished"
            nit = None
        else:
            # stop as soon as the target is met (AUTOSCATTER does the same via
            # a StopIteration callback) -- the search spends most of its time
            # polishing solutions that are already good enough
            state = {"value": np.inf}

            def objective(x_free):
                value, grad = value_and_grad(x_free)
                state["value"] = value
                return value, grad

            def callback(xk, *args):
                if state["value"] < max_violation_success:
                    raise StopIteration

            result = sciopt.minimize(
                objective, np.asarray(x0, dtype=float), jac=True, method=method,
                callback=callback, options=options)
            x_free = result.x
            final = float(result.fun)
            message = str(result.message)
            nit = int(result.get("nit", -1))

        x = np.asarray(embed(x_free))
        loss_ok = final < max_violation_success
        # acceptance is alpha < 0 (Sec. 5.4); the margin delta is what the
        # stability *search* aims for, and the floor below which the sign of
        # alpha is numerical noise -- not an extra requirement on a witness
        alpha = self.abscissa(x)
        stable = alpha < 0.0
        margin_met = alpha < -self.stability_margin
        stability_info = None

        if loss_ok and check_stability and not margin_met and stability_search and len(free_idxs):
            # Sec. 5: the fit landed on the target manifold but in the unstable
            # part of it.  Walk along the manifold towards the stable region.
            x_star, stability_info = self.constrained_stability_search(
                free_idxs, x, max_violation_success=max_violation_success,
                fixed_values=fixed_values, rng=rng)
            if stability_info["success"]:
                x = x_star
                final = stability_info["loss_reached"]
                alpha = stability_info["abscissa"]
                loss_ok = True
                stable = True
                margin_met = stability_info["margin_met"]
                message = message + " + stability search"

        success = bool(loss_ok and (stable or not check_stability))

        info = {
            "success": success,
            "loss_reached": final,
            "loss_below_tolerance": bool(loss_ok),
            "stable": bool(stable),
            "margin_met": bool(margin_met),
            "max_real_eigenvalue": alpha,
            "x": x,
            "x0": np.asarray(embed(x0)),
            "free_idxs": free_idxs,
            "solution_dict": self.param.solution_dict(x),
            "parameters": self.param.physical_report(x),
            "message": message,
            "nit": nit,
            "stability_search": stability_info,
        }
        return success, info

    def _pso_bounds(self, free_idxs):
        bounds = []
        for idx in np.asarray(free_idxs, dtype=int):
            bounds.append(self.pso_ranges[self.param.types[idx]])
        return np.asarray(bounds, dtype=float)

    def repeated_optimize(
        self,
        free_idxs,
        num_tests=10,
        interrupt_if_successful=True,
        rng=None,
        **kwargs
    ):
        """Repeat the continuous optimisation ``num_tests`` times to suppress
        false negatives from local minima (Sec. 4).

        The restarts have a second role (Sec. 5): distinct starting points land
        in *distinct feasible components* of the target manifold -- different
        Bloch-Messiah branches / auxiliary frames -- and stability is not a
        property of the manifold but of the component.
        """
        rng = np.random.default_rng() if rng is None else rng
        infos = []
        for _ in range(int(num_tests)):
            success, info = self.optimize(free_idxs, rng=rng, **kwargs)
            infos.append(info)
            if success and interrupt_if_successful:
                return True, infos
        return any(info["success"] for info in infos), infos

    # ------------------------------------------------------------------ #
    # Sec. 5 -- constrained stability search
    # ------------------------------------------------------------------ #

    def constrained_stability_search(
        self,
        free_idxs,
        seed,
        method="hinge",
        max_violation_success=1.0e-10,
        hinge_weight=None,
        max_iterations=300,
        fixed_values=None,
        rng=None,
        step=0.1,
        allow_sampling=True,
        **method_kwargs
    ):
        """``min alpha(M(x))  s.t.  fit(x) <= tol``, warm-started from ``seed``.

        ``method='hinge'`` minimises ``||r||^2/2 + lam*max(0, alpha+delta)^2``
        with L-BFGS-B and the analytic abscissa gradient; robust, and the only
        thing ``lam`` can change is the path, because the hinge vanishes on the
        whole stable region.

        ``method='reduced'`` is the projected variant: step along
        ``-(I - J^+ J) grad alpha`` (the steepest stability descent that
        preserves the fit to first order), then restore feasibility with one or
        two Gauss-Newton steps ``x <- x - J^+ r``.  It gives an explicit margin
        rather than a trade-off.

        Returns ``(x, info)``.
        """
        free_idxs = np.asarray(free_idxs, dtype=int)
        seed = np.asarray(seed, dtype=float)
        if method == "reduced":
            return self._reduced_gradient_search(
                free_idxs, seed, max_violation_success=max_violation_success,
                max_iterations=max_iterations, step=step, rng=rng,
                allow_sampling=allow_sampling, **method_kwargs)
        return self._hinge_search(
            free_idxs, seed, max_violation_success=max_violation_success,
            hinge_weight=hinge_weight, max_iterations=max_iterations,
            fixed_values=fixed_values, rng=rng, allow_sampling=allow_sampling,
            **method_kwargs)

    def _hinge_search(self, free_idxs, seed, max_violation_success, hinge_weight,
                      max_iterations, fixed_values, rng, allow_sampling,
                      weight_ladder=(1.0, 10.0, 100.0)):
        """Hinge descent with a weight ladder.

        The zero-set of ``||r||^2/2 + lam*max(0, alpha+delta)^2`` does not
        depend on ``lam``, so climbing the ladder cannot change *which* points
        count as solutions -- it only re-balances the two terms along the way,
        which is what a local method needs when the seed sits deep inside the
        unstable region (the fit term is already ~0 there and cannot help).
        """
        base_lam = self.hinge_weight if hinge_weight is None else float(hinge_weight)
        delta = self.stability_margin
        _, fit_value_and_grad = self._constrained_functions(free_idxs, fixed_values)
        base = np.asarray(seed, dtype=float)
        total_sampled = 0
        attempts = []

        for factor in weight_ladder:
            lam = base_lam * float(factor)
            state = {"best": None, "best_alpha": np.inf, "hit": False, "sampled": 0}

            def full(x_free):
                point = base.copy()
                point[free_idxs] = np.asarray(x_free, dtype=float)
                return point

            def objective(x_free):
                fit, fit_grad = fit_value_and_grad(x_free)
                point = full(x_free)
                alpha, alpha_grad, diagnostics = self.abscissa_gradient(
                    point, free_idxs, rng=rng, allow_sampling=allow_sampling)
                if diagnostics.get("sampled"):
                    state["sampled"] += 1
                if not np.isfinite(alpha) or not np.isfinite(fit):
                    # the line search walked into overflow; hand back a wall
                    # rather than an inf, which L-BFGS-B cannot back off from
                    return 1.0e30, np.zeros_like(np.asarray(fit_grad))
                violation = max(0.0, alpha + delta)
                value = fit + lam * violation ** 2
                grad = fit_grad + 2.0 * lam * violation * alpha_grad
                if fit < max_violation_success and alpha < state["best_alpha"]:
                    state["best_alpha"] = alpha
                    state["best"] = point.copy()
                    if alpha < -delta:
                        state["hit"] = True
                return value, grad

            def callback(xk, *args):
                if state["hit"]:
                    raise StopIteration

            try:
                result = sciopt.minimize(
                    objective, base[free_idxs], jac=True, method="L-BFGS-B",
                    callback=callback,
                    options={"maxiter": int(max_iterations), "maxfun": 100000,
                             "ftol": 0.0, "gtol": 0.0})
                endpoint = full(result.x)
            except StopIteration:
                endpoint = None

            total_sampled += state["sampled"]
            candidates = [p for p in (state["best"], endpoint) if p is not None]
            for candidate in candidates:
                loss = self.fit_loss(candidate)
                alpha = self.abscissa(candidate)
                attempts.append((candidate, loss, alpha, lam))
                if loss < max_violation_success and alpha < 0.0:
                    return candidate, {
                        "method": "hinge", "success": True, "loss_reached": loss,
                        "abscissa": alpha, "margin_met": bool(alpha < -delta),
                        "hinge_weight": lam, "num_sampled_gradients": total_sampled,
                    }

        if attempts:
            # report the attempt that came closest to a stable fit
            feasible = [a for a in attempts if a[1] < max_violation_success]
            candidate, loss, alpha, lam = min(feasible or attempts,
                                              key=lambda a: (a[2], a[1]))
        else:
            candidate, lam = base, base_lam
            loss, alpha = self.fit_loss(base), self.abscissa(base)
        return candidate, {
            "method": "hinge",
            "success": False,
            "loss_reached": loss,
            "abscissa": alpha,
            "margin_met": bool(alpha < -delta),
            "hinge_weight": lam,
            "num_sampled_gradients": total_sampled,
        }

    def _reduced_gradient_search(self, free_idxs, seed, max_violation_success,
                                 max_iterations, step, rng, allow_sampling):
        delta = self.stability_margin
        base = np.asarray(seed, dtype=float)
        x = base.copy()
        sampled = 0

        def residual_and_jacobian(point):
            r = np.asarray(self.residual_func(jnp.asarray(point)))
            J = np.asarray(self.residual_jacobian(jnp.asarray(point)))[:, free_idxs]
            return r, J

        current_step = float(step)
        for _ in range(int(max_iterations)):
            alpha, alpha_grad, diagnostics = self.abscissa_gradient(
                x, free_idxs, rng=rng, allow_sampling=allow_sampling)
            if diagnostics.get("sampled"):
                sampled += 1
            if not np.isfinite(alpha):
                break
            if alpha < -delta and self.fit_loss(x) < max_violation_success:
                break
            r, J = residual_and_jacobian(x)
            pinv = np.linalg.pinv(J)
            projector = np.eye(len(free_idxs)) - pinv @ J
            direction = -projector @ alpha_grad
            norm = np.linalg.norm(direction)
            if norm < 1.0e-14:
                break
            candidate = x.copy()
            candidate[free_idxs] = candidate[free_idxs] + current_step * direction / norm
            # restore the fit: one or two Gauss-Newton steps back onto r = 0
            for _ in range(2):
                r_c, J_c = residual_and_jacobian(candidate)
                candidate[free_idxs] = candidate[free_idxs] - np.linalg.pinv(J_c) @ r_c
            if self.fit_loss(candidate) > max_violation_success:
                current_step *= 0.5
                if current_step < 1.0e-10:
                    break
                continue
            if self.abscissa(candidate) >= alpha:
                current_step *= 0.5
                if current_step < 1.0e-10:
                    break
                continue
            x = candidate

        loss = self.fit_loss(x)
        alpha = self.abscissa(x)
        success = bool(loss < max_violation_success and alpha < 0.0)
        return x, {
            "method": "reduced",
            "success": success,
            "loss_reached": loss,
            "abscissa": alpha,
            "margin_met": bool(alpha < -delta),
            "num_sampled_gradients": sampled,
        }

    # ------------------------------------------------------------------ #
    # Sec. 4 -- the verdict itself
    # ------------------------------------------------------------------ #

    def verdict(self, free_idxs, num_tests=10, rng=None, **kwargs):
        """The Sec. 4 pseudocode.  Returns ``(Verdict, info)``.

        Only ``VALID`` or ``INVALID_PROV`` -- the oracle never certifies
        (Sec. 9, invariant 2).
        """
        success, infos = self.repeated_optimize(free_idxs, num_tests=num_tests,
                                                rng=rng, **kwargs)
        if success:
            witness = next(info for info in infos if info["success"])
            return Verdict.VALID, witness
        best = min(infos, key=lambda info: info["loss_reached"])
        best["reason"] = (REASON_HURWITZ if best["loss_below_tolerance"] else REASON_FIT)
        best["all_infos"] = infos
        return Verdict.INVALID_PROV, best


# ---------------------------------------------------------------------------
# gradient-free optimiser (App. C: broadband / eigenvalue-sensitive targets)
# ---------------------------------------------------------------------------

def particle_swarm(
    func, dimension, bounds, num_particles=40, num_iterations=200,
    inertia=0.7, cognitive=1.5, social=1.5, rng=None,
):
    """Minimal particle-swarm optimiser for non-smooth losses.

    Returns ``(best_position, best_value)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    bounds = np.asarray(bounds, dtype=float)
    low, high = bounds[:, 0], bounds[:, 1]
    span = high - low

    position = rng.uniform(low, high, size=(num_particles, dimension))
    velocity = rng.uniform(-span, span, size=(num_particles, dimension)) * 0.1
    value = np.array([func(p) for p in position])

    best_position = position.copy()
    best_value = value.copy()
    global_idx = int(np.argmin(best_value))
    global_position = best_position[global_idx].copy()
    global_value = float(best_value[global_idx])

    for _ in range(int(num_iterations)):
        r1 = rng.random((num_particles, dimension))
        r2 = rng.random((num_particles, dimension))
        velocity = (inertia * velocity
                    + cognitive * r1 * (best_position - position)
                    + social * r2 * (global_position - position))
        position = position + velocity
        value = np.array([func(p) for p in position])
        improved = value < best_value
        best_position[improved] = position[improved]
        best_value[improved] = value[improved]
        idx = int(np.argmin(best_value))
        if best_value[idx] < global_value:
            global_value = float(best_value[idx])
            global_position = best_position[idx].copy()

    return global_position, global_value
