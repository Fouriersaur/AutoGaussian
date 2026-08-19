# AutoGaussian

Automated discovery of quantum-output coupled-mode setups.

Given a target **output covariance spectrum** `σ_target(ω)` — a desired fingerprint of
squeezing, entanglement and directionality in the field *leaving the ports* — this package
discovers the complete set of **minimal irreducible** coupled-mode graphs whose emitted
`σ_out(ω)` matches it, together with the symbolic construction rules behind them.

It is [AUTOSCATTER](https://github.com/jlandgr/autoscatter)'s discrete + continuous discovery
loop run on a **second-moment** (covariance) target instead of a **first-moment**
(scattering-matrix) target. The device model, building blocks, parametrisation, search engine
and stability gate are inherited; the target, the forward map, and a few parameters that
become "live" are what differ.

|                  | AUTOSCATTER (S target)              | AutoGaussian (σ_out target)                     |
|------------------|-------------------------------------|-------------------------------------------------|
| object compared  | `S`                                 | `σ_out = 𝒮 σ_in 𝒮†`, built from `S` **and** `N` |
| moment           | first (mean amplitude)              | second (fluctuations / correlations)             |
| extra input      | —                                   | noise response `N`, input covariance `σ_in`      |
| frequency        | one point (carrier)                 | a spectrum on a grid                             |
| live parameters  | `C_ij, Φ, Δ/κ, γ`                   | those **+ decay ratios `κ̃_i`**                  |
| structure        | reduced `N×N` (phase preserving)    | full `2N×2N` Nambu (phase sensitive)             |
| gauge            | scalar phase per port on `S`        | quadrature rotation on each `2×2` block          |

---

## Install

```bash
cd autogaussian
pip install -e .          # numpy, scipy, jax, sympy, networkx, matplotlib, tqdm
```

or just run from the directory (every script inserts the package root on `sys.path`).

## Quick start

```python
from autogaussian import CovarianceTarget, discover

target = CovarianceTarget(num_ports=1, name="3 dB squeezer")
target.pin((0, 0), 0.5)      # sigma_xx  (vacuum floor = 1)
target.pin((1, 1), 2.0)      # sigma_pp  (pure state)
target.pin((0, 1), 0.0)

result = discover(target, num_auxiliary_modes=0, optimize_gauge=False)
print(result["optimizer"].report())
```

```
#0  complexity=2  couplings=1  pumps=1
      on-site squeezing 0 (complex)
#1  complexity=2  couplings=1  pumps=1
      detuning Delta_0
      on-site squeezing 0 (real)
```

Two irreducible devices: a complex-pumped degenerate squeezer, and a real-pumped one whose
squeezing axis is rotated into place by a detuning.

## Where the algorithm lives

| algorithm-flow section              | module                                |
|-------------------------------------|---------------------------------------|
| §0, §2 Nambu algebra, building blocks | `autogaussian/nambu.py`             |
| §3, App. A forward map `graph → σ_out(Ω)` | `autogaussian/forward.py`       |
| §1 problem specification (pins)     | `autogaussian/target.py`              |
| §1.4 extra constraints `f_j = 0`    | `autogaussian/constraints.py`         |
| §2, App. A.4 learnable parameters   | `autogaussian/parametrization.py`     |
| §4 validity oracle + §5 constrained stability search + App. C loss/optimisers | `autogaussian/oracle.py` |
| §5 abscissa, biorthogonal gradient, gradient sampling | `autogaussian/stability.py` |
| §4/§6 verdicts, two libraries, `certified` flag | `autogaussian/types.py`     |
| §6 discrete search, two-library BFS, ESCALATE | `autogaussian/search.py`, `optimizer.py`, `graph.py` |
| §6 infeasibility certificates (fit-range SDP, PBH, SOS stub) | `autogaussian/certificates.py` |
| §7 complexity analysis + symbolic regression | `autogaussian/postprocess.py` |
| §9 top-level `MAIN`                 | `autogaussian/pipeline.py` (`discover`) |
| App. B worked target gallery        | `autogaussian/gallery.py`             |

## Conventions

* Nambu vector `a = (a_1..a_N, a_1†..a_N†)`; `H_BdG = [[g, ν], [ν*, g*]]` with `g` Hermitian
  (beam-splitters, detunings on the diagonal) and `ν` symmetric (two-mode squeezing,
  on-site/degenerate squeezing on the diagonal).
* Everything dimensionless: `H = κ^{-1/2} H_BdG κ^{-1/2}`, `γ_i = Γ_i/κ_i`, `κ̃_i = κ_i/κ_ref`,
  `Ω = ω/κ_ref`. Cooperativities are `C_ij = 4|H_ij|²`, detunings `Δ_i/κ_i = -H_ii`.
* Forward map:
  `D(Ω) = -i σ_z H - (1+γ)/2 + i Ω κ̃^{-1}`, `S = 1 + D^{-1}`, `N = D^{-1} √γ`,
  `𝒮 = [S | N]`, `σ_out = 𝒮 σ_in 𝒮†`, then `V = ½ W σ_out W†` restricted to the ports.
* Quadratures `x_j = a_j e^{-iθ_j} + h.c.`, `p_j = -i(a_j e^{-iθ_j} - h.c.)`, interleaved as
  `(x_1, p_1, x_2, p_2, …)`, **vacuum floor = 1**.
* Input covariance per channel: `[[2n+1, 2m], [2m*, 2n+1]]` — vacuum is the identity, a pure
  squeezed bath has `|m|² = n(n+1)`. It is *declared*, never optimised.

## Writing a target

```python
target.pin((row, col), value, omega=0.0, weight=1.0)      # single point
target.pin_derivative((row, col), order=2, value=0.0)     # flat-band / broadband
target.pin_spectrum((row, col), omegas, values)           # full spectrum
target.pin_matrix(matrix)                                 # None entries stay free ('*')
target.pin_form(vector, value)                            # Var(v·R): nullifiers, Duan sums
```

Values may be sympy expressions; their free symbols become optimisation variables (e.g. pin
`σ_pp = t` to say "any anti-squeezed value").

The gallery in `autogaussian/gallery.py` ships App. B ready-made: `single_mode_squeezer`,
`epr_source`, `directional_squeezed_source`, `broadband_squeezer`, `cv_graph_state`,
`backaction_evading_readout`, `noise_diode`.

### Frequency-dependent (full-spectrum) targets — App. B.8–B.12

B.1–B.7 pin `σ_out` at a single `Ω`, where the decay ratios `κ̃_i` are inert. `SPECTRAL_GALLERY`
holds the targets pinned on a *grid*, which is exactly what makes `κ̃_i` live optimisation
variables (§3, App. A.4(c)):

| target | function | what it exercises | minimum modes found |
|---|---|---|---|
| B.8 squeeze-angle rotation (filter cavity) | `squeeze_angle_rotation` | phase-sensitive Nambu cross term `σ_xp(Ω)`; declared squeezed input | 1 (passive, exact) |
| B.9 bandpass sideband squeezing | `bandpass_squeezer` | a spectral *window*: `Ω_c` ← detuning, `B` ← `κ̃` | 2 |
| B.10 flat-top order `n` | `flat_top_squeezer` | spectral degree → mode count | 1 (n≤2), 2 (n=3) |
| B.11 band-limited EPR | `band_limited_epr` | entanglement bandwidth, written on joint quadratures | 3 |
| B.12 notch (band-stop) squeezing | `notch_squeezer` | a dip needs one resonance, a dip *with a hole* needs the partner | 3 |
| B.8-imp impure rotation | `impure_rotation` | a prescribed purity spectrum `μ(Ω) ≥ 1` → the intrinsic losses `γ_i` go live | 1 (with `γ₀`) |
| B.8-nm non-monotone rotation | `non_monotone_rotation` | a turning point in `θ(|Ω|)` → rotation *shape* costs a coupled partner | 2 |
| B.8-wind prescribed winding | `winding_rotation` | more accumulated sweep than one mode supplies → rotation *magnitude* costs a mode | 2 |

Two conventions these targets rely on:

* **Grids run over `Ω ≥ 0`.** The symmetrised spectrum obeys `V(−Ω) = V(Ω)ᵀ`, so every diagonal
  entry is automatically even in `Ω`. Pinning `−Ω` is redundant, and pinning a shape that is *not*
  even asks for something no device can emit.
* **Shape targets are matched to ~1e-8, not 1e-10.** B.9–B.12 carry
  `kwargs_optimization={"max_violation_success": 1e-8}` in their `GalleryProblem`; use
  `problem.optimizer_kwargs(kwargs_optimization={"num_tests": 15})`, which merges rather than
  replaces. B.8 is exact (residual ~1e-16) because its target is built from the closed form a
  single detuned lossless mode actually emits.

B.8 doubles as a forward-map golden test: one lossless mode at `Δ/κ = 1/2` reflecting `r`-squeezed
vacuum returns `det σ_out(Ω) = 1` at every `Ω` with `Ω`-independent ellipse eigenvalues
`(e^{−2r}, e^{+2r})`, while the axis sweeps `0 → π/2` — depth untouched, angle rotated, as a
lossless passive element must. `Δ/κ = 1/2` is forced, not chosen: `θ(0) = 0` requires
`4 arctan(2Δ/κ) = π`. The fit recovers `Δ/κ = 0.500000` and `κ̃ = √2 Ω_c` with no squeezing
element used at all.

#### The squeeze-angle-rotation family — §8.7 (`ROTATION_FAMILY`)

B.8 asks for a rotating axis at *fixed depth* on a *pure* state. The three extensions each break
one of those and ask what the break costs. None of them has a closed form to pin from — an
impure or two-pole rotation is a ratio of quadratics whose angle and purity spectra are the
symbolic-regression *outputs* of §7.2, never inputs — so they are written as the spectrum of a
declared **reference device** (`rotation_device` → `rotation_spectrum`), with the prescribed
quantity *measured* off it by `squeeze_angle`, `rotation_purity` and `accumulated_rotation`. The
fit never sees the reference, only the pinned numbers.

Three results here are measured, and two of them differ from how §8.7 words the claim:

* **B.8-imp.** A lossless *single* mode is all-pass — it rotates the input ellipse
  symplectically, so `det σ_out = 1` for every detuning and decay ratio and `μ(Ω) > 1` is out of
  reach. Freeing `γ₀` fixes it in one parameter (the fit recovers the declared `γ = 0.2`). But
  "no lossless graph can do it" is *too strong*: a second, still lossless, mode carries its own
  vacuum input channel and supplies the noise instead, so the two-mode lossless graph does fit.
  Impurity costs *added noise*; an auxiliary mode is one way to buy it.
* **B.8-nm.** One detuned mode is strictly monotone in `|Ω|` (single-pole phase — swept over the
  whole single-mode family, not argued), so a turning point needs two modes. §8.7 attributes it
  to *opposite-sign detunings*; in this coupled-mode parametrisation the sign of the bare
  detunings is not what decides it (measured over a wide random sweep, the normal-mode sign rule
  predicts monotonicity ~70% of the time). What does decide it is **hybridisation**: switch the
  coupling off and the partner is invisible to the port — the spectra agree to 1e-16 — so the
  default reference is a same-sign pair with `C^BS ≫` its detunings.
* **B.8-wind.** The single-mode winding ceiling on `[0, 6κ]` measures **3.013 rad ≈ π**, not
  `π/2`: the `π/2` of B.8 is that device's useful *in-band* rotation, not its ceiling. So
  `single_mode_winding_ceiling` finds the threshold by scanning the family, and nothing is
  hard-coded. Winding past the ceiling *proves* a second mode is needed; the converse is false,
  and is asserted so nobody turns the ceiling into a budget — the same reference scaled down to
  1.2 rad of winding still fails on one mode, because its *shape* is what one pole cannot make.

## The graph lattice

Because the oracle works in the full Nambu structure, a mode **pair carries two independent
edges** (a beam-splitter one and a squeezing one), and each mode carries a detuning and an
on-site squeezing slot. A graph is an integer vector over those slots with AUTOSCATTER's
value semantics — `0` none, `1` real, `2` complex — so complexity is additive and validity
stays monotone under edge addition/removal.

```python
space = GraphSpace(num_modes=3,
                   forbidden_couplings=[(0, 1)],      # must be mediated
                   passive_only_couplings=[(0, 2)],   # no active link here
                   real_only_couplings=[(1, 2)])      # no synthetic flux here
```

Search-space sizes: `N=1` → 6 graphs, `N=2` → 324, `N=3` → 157 464, `N=4` → 6.9e8, where
`N = P + n_aux`. The monotone pruning of §6 tests only a fraction of these — measured on the EPR
target, 324 graphs → 47 oracle calls, with complexity levels 5 and below never touched. Full
enumeration is a seconds-to-a-minute job at `N ≤ 2` and roughly hours at `N = 3` (0.19 s per
oracle call there). Beyond that, either restrict the graph space or skip full enumeration and use
`optimizer.greedy_minimal_subgraph(...)` / `optimizer.minimal_valid_subgraphs(...)` to descend
from the fully connected graph — you keep correctness (every graph reported is oracle-verified)
and lose only the "and there are no others" claim.

Full searches actually run at `N = 2` (lattice 324):

| target | oracle calls | time | irreducible graphs |
|---|---|---|---|
| B.1 single-mode squeezer | 47 | 2 s | 2 |
| B.2 EPR source | 47 | 4 s | 3 |
| B.4 broadband squeezer | 241 | 43 s | 9 (see note) |
| B.5 CV graph state (2-node cluster) | 153 | 20 s | 3 |
| B.6 backaction-evading readout | 207 | 10 s | 3 |
| B.7 noise diode | 29 | 38 s | 3 |

B.1/B.2 come from `examples/01`, B.4 from `examples/02`, B.5–B.7 from `examples/04`.

B.6 pins a single cross-term and is satisfied trivially by a bare beam-splitter — that is the
target being loose, not the search misbehaving. B.5 was the same until its target was tightened:
pinning *one* nullifier admitted 12 irreducible devices, several of them two **decoupled**
squeezers (squeeze `p₀` in one mode, `x₁` in the other, blow the conjugate combination up to 3e6
to stay pure). `cv_graph_state` now pins one nullifier **per node**, `n_i = p_i − Σ_j A_ij x_j`,
which is how a CV graph state is actually defined; that cuts the list to 3 and every survivor
genuinely couples the modes.

**B.4 is not converged at `num_tests=15`.** One of its irreducible graphs is found in only ~50 %
of seeds with plain BFGS restarts, so the count above is a lower bound. This is the false-negative
problem, and it is specific to broadband / derivative targets, exactly as App. C warns. Measured on
that graph:

| optimiser | found in | wall clock |
|---|---|---|
| `BFGS`, 15 restarts | 2/6 seeds | 11.5 s |
| `BFGS`, 40 restarts | 4/6 seeds | 13.9 s |
| `pso+bfgs`, 2 restarts | **6/6 seeds** | 9.1 s |

Use `kwargs_optimization={"method": "pso+bfgs"}` for spectral targets. The swarm locates the basin
and BFGS polishes to the `1e-10` tolerance — the swarm alone never gets there, and BFGS alone often
starts in the wrong basin. It costs roughly 10x per oracle call, so the practical recipe is plain
BFGS for the layer sweep and the hybrid for the verification descent, via
`perform_breadth_first_search(verify_kwargs={"method": "pso+bfgs"})`.

## Two libraries and the completeness caveat (§6, §8)

A search that fails to find a witness has learned something much weaker than a search that
proves none exists, and the difference decides whether a whole subtree may be deleted. The
engine therefore keeps **two libraries** and a `certified` flag:

| verdict           | meaning                                  | may condemn subgraphs? |
|-------------------|------------------------------------------|------------------------|
| `VALID`           | witness stored (`fit ≤ tol`, `α < 0`)    | extensions are valid   |
| `INVALID_PROV`    | the oracle did not find a witness        | **no**                 |
| `INVALID_CERT`    | a certificate proved there is none       | yes                    |
| `INVALID_DEFAULT` | escalation ran, nothing fired            | **no**                 |

```python
result = discover(target, num_auxiliary_modes=0)
result["n_uncertified"]     # graphs rejected without a proof
print(result["completeness"])
# The list of irreducible graphs is complete up to at most 3 false negatives ...
```

`n_uncertified == 0` is the only configuration in which the run may claim a complete, certified
answer; otherwise the uncertified rejections are listed by name so the caveat is auditable. The
escalation ladder (`autogaussian/search.py`) is: cheap structural certificates → fresh seeds from
different feasible components (this rung can promote a graph to `VALID`) → PBH dark mode → give
up loudly. Certificates live in `certificates.py`:

* `certify_passive_range` — a beam-splitter-only graph maps vacuum to vacuum at every frequency,
  so any non-vacuum pin is unreachable *for that graph*. Closed form, graph-dependent, a proof.
* `certify_target_unphysical` — SDP feasibility of `V + iΩ ⪰ 0` under the pins. Infeasible means
  no quantum state has those second moments, so no graph does either. Needs `cvxpy`
  (`pip install autogaussian[certificates]`); without it the rung reports "inconclusive".
* `pbh_dark_mode` — real, and honest about its reach: in *this* device family every mode is
  coupled to an input line, so the test can never fire. It is the one Hurwitz certificate that is
  a theorem rather than a stub, and it does fire for families with undamped internal modes.
* `sos_no_hurwitz` — **stub**. `method="sos"` raises `NotImplementedError` with the interface
  fixed; the v1 stand-in samples the feasible set and reports evidence, which is not a proof, so
  it never certifies. Everything it touches becomes `INVALID_DEFAULT` and keeps its subtree alive.

The price is oracle calls: not condemning uncertified subtrees means testing them. Pass
`search_kwargs={"prune_uncertified": True}` to reinstate AUTOSCATTER's faster, **unsound** rule,
or `engine="legacy"` for the old walk plus its `verify_irreducibility` repair pass.

## Notes on the oracle

* **Stability is a search constraint, not a filter** (§5). The fit is the hard constraint and
  contains *no* stability term; a fit that lands in the unstable part of the target manifold is
  handed to `constrained_stability_search`, which minimises the abscissa subject to `fit ≤ tol`
  (hinge `‖r‖²/2 + λ·max(0, α+δ)²`, or a reduced-gradient variant that steps in the tangent space
  of the fit manifold). Never add the raw `α` to the loss: it keeps paying for margin it already
  has, and the only currency it has is the target — `tests/test_constrained_stability.py`
  documents the resulting drift (0.500 → 0.552 as `ρ` grows), and the hinge's independence of `λ`.
* **The abscissa gradient is analytic, never autodiffed through `eig`** (§5):
  `∂Re λ_max/∂x_k = Re(uᴴ (∂M/∂x_k) v)/(uᴴv)`, with the eigenvalue condition number `1/|uᴴv|`
  monitored. Two hazards are handled: a rightmost tie whose members disagree (Lipschitz corner →
  minimum-norm element of the convex hull) and an exceptional point where `1/|uᴴv|` diverges like
  `t^{-1/2}` (→ gradient sampling over a ball wide enough to straddle it). The conjugate tie that
  BdG structure forces on every oscillating mode is *spurious* and is recognised as such by
  comparing the tied gradients rather than counting them.
* **Derivative pins use forward-mode autodiff in `Ω`**, not finite differences — exact, and it
  removes one of the sources of kinks App. C warns about. A gradient-free particle swarm is
  available via `kwargs_optimization={"method": "pso"}` for the rest.
* **False negatives are the one real failure mode**, and the two-library engine answers them by
  construction: an uncertified invalid never removes its subgraphs from the search, so a bad
  afternoon at the optimiser costs oracle calls rather than solutions. What survives is reported
  as `n_uncertified`. Raise `kwargs_optimization={"num_tests": …}` to shrink it; with
  `engine="legacy"` the old repair pass `verify_irreducibility()` is still available.
* **Under-specified targets are met trivially** (§11): "squeeze port 1, vacuum port 2" is
  satisfied by decoupling the ports. Add a `TransmissionConstraint` so the minimal solution is a
  single connected device.
* **Derivative pins constrain the spectrum only *at* the pinned point.** A flat-band target
  written with derivatives alone is satisfiable by shrinking `κ̃` until the curvature vanishes on
  a linewidth far narrower than the band you meant — flat at a point, not across a band. Pin the
  band as well (`broadband_squeezer(..., bandwidth=W)`, or `pin_spectrum` directly). Two runs of
  the same 3 dB target, `σ_xx` sampled at Ω = 0 … 1:

  ```
  derivatives only        0.500 0.501 0.521 0.605 0.942 1.063 1.053    kappa~ = 0.53
  derivatives + bandwidth 0.500 0.500 0.501 0.493 0.500 0.722 0.847    kappa~ = 0.29, 0.98
  ```
* **Decay ratios are bounded**, `κ̃ ∈ [0.01, 100]` by default, via a smooth `tanh` squash on
  `log κ̃` (`log_decay_ratio_bound`, set `None` to disable). Unbounded ratios overflow `exp` and
  feed the degeneracy above.

## Asymptotic building blocks (§7.3)

When a coupling restriction forces mediation through a bus, pass
`asymptotic_bus_modes=(b,), bus_cooperativity=1e6`. All couplings touching the bus are scaled by
`√Λ` per bus endpoint, so a self term scales with `Λ` and a link with `√Λ`: the ratios
`C_ajb/√C_bb` stay finite, the bus is virtually occupied, and the residual error is `~1/C_bus`.

Worked example in `1_squeezing_sources.ipynb` (B.2′): the EPR target of B.2 with the port–port
edge forbidden, `forbidden_couplings=[(0, 1)]` — the covariance analogue of AUTOSCATTER's
far-detuned bus modes in `6_optomechanical_circulator.ipynb`. Two irreducible mirror-image
architectures come out, each with **two** couplings (squeezing port→bus, beam-splitter bus→port),
and holding their ratios fixed while lowering `Λ` reproduces the `1/C_bus` error law. Without the
block the same architecture is still found, but only by running the cooperativities up to `~1e5`:
`asymptotic_bus_modes` adds no solutions, it makes the limit point representable.

## Devices vs limit points: the cooperativity budget

A graph can pass the oracle without being a device: it may meet the target only along a ray
that runs to infinity, stopping wherever the fit first crosses tolerance. B.10 order 3 is the
worked case — its cheapest architectures diverge as `C_11 ~ Delta_1^2 -> ∞` at fixed
`|ν_11|/Δ_1 -> 1`, an auxiliary mode held at its detuned parametric threshold.

Tightening `max_violation_success` does **not** fix this; it walks further along the ray
(`C_11` = 518 → 1153 → 1373 as the tolerance goes 1e-8 → 1e-9 → 1e-10). Bound the diverging
knob instead:

```python
from autogaussian import CooperativityBudget
optimizer = CovarianceArchitectureOptimizer(
    target, constraints=(CooperativityBudget(100.0),), ...)
```

a one-sided hinge `4|H_ij|² ≤ C_max` — pumps carry finite power. A limit point fails at every
finite budget and its loss degrades as the budget tightens; a genuine device is indifferent:

| architecture | none | C ≤ 100 | C ≤ 25 |
|---|---|---|---|
| cheapest order-3 graph | VALID 1.0e-8 | **INVALID** 2.9e-7 | **INVALID** 2.2e-6 |
| richer order-3 graph | VALID 8.5e-9 | VALID 4.7e-9 | VALID 2.9e-9 |

This doubles as the *detector*: refit under a budget and watch the loss. Two caveats. The
budget closes the `C, Δ → ∞` ray but not every ray — solutions can run away in `κ̃` instead
and sit on `log_decay_ratio_bound` (default `κ̃ ≤ 100`), so a full "buildable only" statement
needs both budgets. And searching under a budget changes the *claim* ("complete within this
budget") and does not shrink the list: constrained graphs need more elements to stay minimal,
so B.10 order 3 returns 25 irreducible graphs under `C ≤ 100, κ̃ ≤ 10` against 20 unbounded.

Contrast with §7.3: there the limit is *necessary* (a hardware restriction forces mediation),
so the ray is reparametrised and kept. Here nothing forces it, so it is excluded. Worked
through in the B.10 section of `5_spectral_gallery.ipynb`.

## Reading off the discovered devices (§8)

`optimizer.report()` lists the architectures that survived the search. To also get the numbers
behind them — what a hardware implementation has to dial in — use

```python
from autogaussian import solution_table, parameter_summary

print(solution_table(optimizer, irreducibles))     # every valid graph + its converged solution
print(parameter_summary(info, optimizer.space, graph))   # one oracle result, compact
```

`solution_table` prints, per graph, the complexity/coupling/pump counts, the element list, the
final loss and stability margin `max Re eig(M̃)`, and then the physical parameters grouped as
cooperativities `C_ij`, pump phases `arg(g_ij)`/`arg(ν_ij)`, detunings `Δ_i/κ_i`, intrinsic
losses `γ_i`, decay ratios `κ̃_i` and gauge phases. Options: `limit` (truncate the list), `sort`
(cheapest first, default), `resolve_missing=True` (re-run the oracle for graphs whose solution
was not cached). For an asymptotic-bus run the bus couplings appear as the finite ratios of §7.3.

Both helpers read the `parameters` field the oracle already stores on every successful run, so
they cost nothing extra. All five notebooks now print them for every device they report:
`solution_table` after each search, `parameter_summary` after each single-graph fit.

## Construction rules (§7.2)

```python
from autogaussian import symbolic_regression
result = symbolic_regression(optimizer, graph, v, np.linspace(0.05, 0.95, 19))
```

sweeps a free target symbol, refits the device at each point, and fits closed forms. For the
single-mode squeezer it recovers the exact analytic recipe:

```
C_00 = 1.000*(sqrt(v) - 1.0)**2/(sqrt(v) + 1.000)**2      [rmse 9e-08]
arg(nu_00) = -1.570796                                     [constant]
```

## Notebooks

Every example also exists as a notebook, runnable in Google Colab (no local install, no CPU
of yours burnt) — the first cell installs the package when it detects Colab:

| notebook | contents | Colab |
|---|---|---|
| `1_squeezing_sources.ipynb` | B.1 single-mode squeezer, `C₀₀(v)` construction rule, B.2 EPR source, B.2′ EPR through an asymptotic bus (§7.3) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fouriersaur/AutoGaussian/blob/main/autogaussian/1_squeezing_sources.ipynb) |
| `2_directional_and_broadband.ipynb` | B.3 directionality (aux-mode budget + greedy minimal subgraph), B.4 flat-band squeezer + full BFS over all 3⁶ graphs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fouriersaur/AutoGaussian/blob/main/autogaussian/2_directional_and_broadband.ipynb) |
| `3_reference_table.ipynb` | the App. F reference table, row by row | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fouriersaur/AutoGaussian/blob/main/autogaussian/3_reference_table.ipynb) |
| `4_gallery_sweep.ipynb` | B.5 nullifiers, B.6 cross-term, B.7 thermal input, with graph plots | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fouriersaur/AutoGaussian/blob/main/autogaussian/4_gallery_sweep.ipynb) |
| `5_spectral_gallery.ipynb` | B.8 squeeze-angle rotation, the rotation family B.8-imp/nm/wind (§8.7), B.9 bandpass, B.10 flat-top order sweep, B.11 band-limited EPR, B.12 notch — spectra plotted against their targets, plus the full §6 discovery for B.8, B.9 and B.10 (those three cells take ~8 min each) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fouriersaur/AutoGaussian/blob/main/autogaussian/5_spectral_gallery.ipynb) |

The badges point at `main` of the public repo; the notebooks also run locally from a clone
(the install cell falls back to putting the package root on `sys.path`).

## Tests and examples

```bash
python tests/test_forward.py               # forward map vs closed forms (9 checks)
python tests/test_oracle.py                # two-sided verdicts, gauge, constraints (8 checks)
python tests/test_search.py                # lattice, monotonicity, full searches (7 checks)
python tests/test_stability.py             # abscissa gradient, ties, exceptional points (7)
python tests/test_constrained_stability.py # hinge vs value penalty, reduced gradient (5)
python tests/test_search_invariants.py     # two-library invariants, pruning frontier (8)
python tests/test_certificates.py          # fit-range SDP, passive range, PBH, stubs (9)
python tests/test_integration.py           # end-to-end libraries + completeness (2)
python tests/test_gallery_spectral.py      # B.8-B.12 full-spectrum targets, live kappa~ (13)
python tests/test_gallery_rotation_family.py  # B.8-imp/nm/wind: purity, turning point, winding (15)

python examples/01_discover_squeezers.py        # B.1 + B.2 discovery + construction rules
python examples/02_directional_and_spectral.py  # B.3 directionality, B.4 flat band
python examples/03_reference_table.py           # reproduces the App. F reference table
python examples/04_gallery_sweep.py             # B.5 nullifiers, B.6 cross-term, B.7 thermal input
python examples/05_spectral_gallery.py          # B.8-B.12 + the B.8-imp/nm/wind rotation family
python examples/05_spectral_gallery.py --search # + full §6 discovery on B.8, B.9, B.10 (~17 min)
```

`05_spectral_gallery.py` runs the whole full-spectrum gallery in ~20 s and prints the mode-count
ladder for each target — the executable form of "spectral structure costs modes": one mode short
of the structural minimum the fit never reaches tolerance on any restart (a *provisional*,
non-condemning INVALID, §6), and adding the mode flips it VALID. `test_gallery_spectral.py`
asserts the same transitions, plus the `V(−Ω) = V(Ω)ᵀ` convention, the B.8 forward-map golden
values, and an explicit check that freezing `κ̃` (AUTOSCATTER's choice) *fails* on a spectral
target. `test_gallery_rotation_family.py` does the same for §8.7: the purity, turning-point and
winding claims above, each asserted as the feasibility transition rather than an exact mode
count, plus the two places where the measured behaviour is weaker than §8.7's wording.

### Oracle vs search on a spectral target

Everything above is the **oracle** (§4) on the fully connected graph — *can this many modes do
it* — which is what the mode-count ladder needs. The **search** (§6) is the breadth-first walk
down the lattice that returns the two libraries and the *irreducible* graphs, and on a spectral
target it is expensive: one oracle call costs ~1 s against ~10 ms for a single-point target, and
a 2-mode lattice holds `3⁶ = 729` graphs.

| target | modes | slots | full BFS | irreducibles | uncertified |
|---|---|---|---|---|---|
| B.8 | 1 | 2 | 0.9 s, 4 calls | **1** — `detuning Δ₀`, complexity 1 | 3 |
| B.9 | 2 | 6 | 485 s, 313 calls | 11, complexity 6–7 | 269 |
| B.10 n=2 | 1 | 2 | 0.8 s, 4 calls | **1** — `detuning Δ₀ + on-site sqz 0 (real)` | 2 |
| B.10 n=3 | 2 | 6 | 327 s, 305 calls | 20, cheapest complexity 4 | 160 |
| B.11, B.12 | 3 | 12 | `3¹² ≈ 5×10⁵`, out of reach | — | — |

Three results worth reading off. B.8's minimal device contains **no squeezing element at all** —
a bare detuned passive mode, with all the squeezing supplied by the declared `σ_in`; the lattice
walk found that unaided. Every one of B.9's 11 irreducible graphs carries **two** squeezing
elements, which turns "one resonance cannot make this window" from an observation about a
stalled fit into a structural statement. And B.10's two searches are the degree bound stated on
*minimal* graphs rather than fully connected ones: order 2 is one detuned mode with a single
real on-site squeezer (complexity 2), order 3 needs two modes and complexity 4 at its cheapest.

Note the last column. On a spectral target almost every rejection lands `certified=False`
(`sos_no_hurwitz` is a stub and the fit-range SDP rarely fires on a grid), so B.9's honest
completeness statement is *complete up to at most 269 false negatives* — a much weaker claim
than the single-point targets of examples 01–04 support. That count is the deliverable, not a
blemish to hide (§8). For B.11/B.12 use `greedy_minimal_subgraph` / `minimal_valid_subgraphs`
below the fully connected graph instead of a full BFS.

`04_gallery_sweep.py` covers the three gallery targets that exercise machinery nothing else
touches: `pin_form` quadratic-form pins (B.5), a lone pinned cross-term (B.6), and a declared
non-vacuum `σ_in` (B.7). It also prints where those App. B targets are under-specified — B.6 as
written is satisfied trivially by a bare beam-splitter, and B.7 leaves port 2 free, which the
search drives to variance ~1e7 against the stability boundary.

Example 2 also reproduces App. F's structural claim directly: with the backward direction
pinned to zero, **0 auxiliary modes is INVALID** (best loss 0.27) and **1 damped auxiliary is
VALID** — two directly coupled modes cannot be made directional by any coupling phase alone.
The solution reaches forward `|S₁₀|² = 1.300`, backward `0.000`, 7.0 dB on port 2, port 1 at
the vacuum floor.

`examples/03_reference_table.py` reproduces the verified reference table:

| target                | minimal setup                              | result at Ω = 0                    |
|-----------------------|--------------------------------------------|------------------------------------|
| single-mode squeezing | 1 mode, on-site squeezing, 1 port          | 12.1 dB (variance 0.062)           |
| EPR entanglement      | 2 modes, 1 squeezing edge, 2 ports         | joint 0.0627, local 8.00           |
| reciprocal squeezing  | 2 modes, BS + squeezing edge, 2 ports      | 5.0 dB at both ports (0.315)       |
| directional squeezing | 2 signal + 1 damped aux, on-site sqz + BS  | 7.0 dB / vacuum, fwd 1.30, bwd 0.72 |

## Caveats

* Active targets live near the parametric-oscillation threshold — target a finite squeezing
  depth and map the stability frontier rather than chasing `variance → 0`.
* Intrinsic loss `γ` mixes vacuum in through `N`; the squeezing floor rises with `γ`.
* Directionality itself is a *finite*-cooperativity effect; a mediation restriction is what pushes
  a target onto the asymptotic manifold — and then only its *minimal* architectures. B.2′ makes
  the distinction concrete: the two-coupling solutions of the mediated EPR source live at
  `C_bus → ∞`, while the same target with the full element budget (port detunings, a degenerate
  pump on the mediator) is met by a resonant auxiliary mode at cooperativities of order one.
