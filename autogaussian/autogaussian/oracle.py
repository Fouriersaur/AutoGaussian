"""
VALIDITY ORACLE (Sec. 4 of the algorithm flow).

    function ORACLE(graph, spec):
        x* = MINIMIZE(LOSS)                  # continuous optimisation
        if LOSS(x*) > tol:   return INVALID
        if not STABLE(M(x*)): return INVALID # Sec. 5, a *separate* gate
        return VALID, x*

The loss (App. C) is

    L(x) = sum_Omega sum_pinned w |sigma_out(Omega)_ij - target_ij|^2
         + sum_m lambda_m |d^m/dOmega^m [pinned entry] - target|^2
         + sum_j |f_j(x)|^2

Derivative pins are evaluated by *forward-mode autodiff* in ``Omega`` rather
than finite differences -- exact, and it removes one source of the kinks that
App. C warns about.

Because ``sigma_out = S_cal sigma_in S_cal^dag`` is a flat evaluation for a
stable graph, the verdict is genuinely two-sided: there is no hidden universal
quantifier and no UNDECIDED outcome.
"""

import numpy as np
import jax
import jax.numpy as jnp
import scipy.optimize as sciopt
import sympy as sp

from autogaussian.constraints import ForwardContext
from autogaussian.forward import max_real_eigenvalue, response_matrices
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
from autogaussian.target import PART_FULL, PART_IMAG, PART_REAL

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
    """Loss, continuous optimiser and stability gate for one problem spec.

    Parameters
    ----------
    parametrization : :class:`autogaussian.parametrization.Parametrization`
    target : :class:`autogaussian.target.CovarianceTarget`
    sigma_in_signal : (2N, 2N) array, optional
        Declared input covariance of the signal channels (vacuum if omitted).
        *Declared, not optimised* (Sec. 1.2).
    constraints : sequence of :class:`autogaussian.constraints.BaseConstraint`
    stability_margin : float
        Required ``-Re eig(M~) > margin``.
    stability_weight : float
        Weight of a soft stability penalty added to the loss.  The hard gate is
        applied regardless; the penalty only steers the optimiser away from the
        unstable region (0 disables it).
    init_ranges : dict
        Per variable-type initial-guess ranges.
    """

    def __init__(
        self,
        parametrization,
        target,
        sigma_in_signal=None,
        constraints=(),
        stability_margin=0.0,
        stability_weight=1.0,
        init_ranges=None,
        pso_ranges=None,
    ):
        self.param = parametrization
        self.target = target
        self.constraints = list(constraints)
        self.stability_margin = float(stability_margin)
        self.stability_weight = float(stability_weight)
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
        self.sigma_in_total = stack_input_covariance(self.sigma_in_signal, self.num_modes)

        self._compile()

    # ------------------------------------------------------------------ #
    # forward map wrappers
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # loss assembly
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

        def loss(x):
            responses = jax.vmap(lambda w: self._responses(x, w))(omegas)
            S_all, N_all, V_all = responses
            total = 0.0

            for part, group in self._groups.items():
                actual = V_all[group["k"], group["row"], group["col"]]
                wanted = group["values"](x)
                if part == PART_REAL:
                    residual = jnp.real(actual) - jnp.real(wanted)
                elif part == PART_IMAG:
                    residual = jnp.imag(actual) - jnp.imag(wanted)
                else:
                    residual = actual - wanted
                total = total + jnp.sum(group["w"] * jnp.abs(residual) ** 2)

            if self._form_group is not None:
                group = self._form_group
                vectors = group["v"]                        # (M, 2P)
                num_ports = self.num_ports
                blocks = V_all[group["k"], : 2 * num_ports, : 2 * num_ports]
                actual = jnp.real(jnp.einsum("ma,mab,mb->m", vectors, blocks, vectors))
                residual = actual - jnp.real(group["values"](x))
                total = total + jnp.sum(group["w"] * jnp.abs(residual) ** 2)

            for group in self._derivative_groups:
                dV = self._covariance_derivative(x, group["omega"], group["order"])
                actual = dV[group["row"], group["col"]]
                wanted = group["values"](x)
                residual = actual - wanted
                if all(part == PART_REAL for part in group["part"]):
                    residual = jnp.real(actual) - jnp.real(wanted)
                elif all(part == PART_IMAG for part in group["part"]):
                    residual = jnp.imag(actual) - jnp.imag(wanted)
                total = total + jnp.sum(group["w"] * jnp.abs(residual) ** 2)

            if self.constraints:
                H, gamma, kappa_tilde, thetas = self.param.unpack(x)
                ctx = ForwardContext(
                    x=x, H=H, gamma=gamma, kappa_tilde=kappa_tilde, thetas=thetas,
                    omegas=self.omegas, S=S_all, N=N_all, V=V_all,
                    num_modes=self.num_modes, num_ports=self.num_ports,
                )
                residuals = jnp.concatenate([jnp.atleast_1d(c(ctx)) for c in self.constraints])
                total = total + jnp.sum(jnp.abs(residuals) ** 2)

            if self.stability_weight > 0.0:
                H, gamma, kappa_tilde, _ = self.param.unpack(x)
                excess = max_real_eigenvalue(H, gamma, kappa_tilde) + self.stability_margin
                total = total + self.stability_weight * jnp.maximum(excess, 0.0) ** 2

            return 0.5 * total

        self._loss = loss
        self.loss_func = jax.jit(loss)
        self.loss_and_grad = jax.jit(jax.value_and_grad(loss))

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

    def max_real_eigenvalue(self, x):
        H, gamma, kappa_tilde, _ = self.param.unpack(jnp.asarray(x))
        return float(np.real(max_real_eigenvalue(H, gamma, kappa_tilde)))

    def is_stable(self, x):
        return self.max_real_eigenvalue(x) < -self.stability_margin

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
        init_ranges=None,
        fixed_values=None,
    ):
        """One continuous run.  Returns ``(success, info)``.

        ``fixed_values`` maps variable indices to values held fixed (used to
        sweep a free target symbol during symbolic regression, Sec. 7.2).
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
        stable = self.is_stable(x)
        success = bool(loss_ok and (stable or not check_stability))

        info = {
            "success": success,
            "loss_reached": final,
            "loss_below_tolerance": bool(loss_ok),
            "stable": bool(stable),
            "max_real_eigenvalue": self.max_real_eigenvalue(x),
            "x": x,
            "x0": np.asarray(embed(x0)),
            "free_idxs": free_idxs,
            "solution_dict": self.param.solution_dict(x),
            "parameters": self.param.physical_report(x),
            "message": message,
            "nit": nit,
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
        false negatives from local minima (Sec. 4)."""
        rng = np.random.default_rng() if rng is None else rng
        infos = []
        for _ in range(int(num_tests)):
            success, info = self.optimize(free_idxs, rng=rng, **kwargs)
            infos.append(info)
            if success and interrupt_if_successful:
                return True, infos
        return any(info["success"] for info in infos), infos


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
