"""
AutoGaussian -- automated discovery of quantum-output coupled-mode setups.

AUTOSCATTER's discrete + continuous discovery loop run on a **second-moment**
(output covariance) target instead of a **first-moment** (scattering-matrix)
target.  Device model, building blocks, parametrisation, search engine and
stability gate are inherited; the target, the forward map, and the parameters
that become "live" are what differ:

===============  ==========================  ================================
                 AUTOSCATTER (S-target)      AutoGaussian (sigma_out-target)
===============  ==========================  ================================
object compared  S                           sigma_out = S_cal sigma_in S_cal^dag
moment           first                       second
extra input      --                          noise response N, input cov sigma_in
frequency        one point (carrier)         a spectrum on a grid
live parameters  C_ij, Phi, Delta/kappa, gam  + decay ratios kappa~_i
structure        reduced N x N               full 2N x 2N Nambu
gauge            scalar phase per port       quadrature rotation per 2x2 block
===============  ==========================  ================================

Typical use::

    from autogaussian import CovarianceTarget, CovarianceArchitectureOptimizer, qidx

    target = CovarianceTarget(num_ports=1)
    target.pin((0, 0), 0.5)          # 3 dB squeezing in x at Omega = 0
    target.pin((1, 1), 2.0)

    optimizer = CovarianceArchitectureOptimizer(target, num_auxiliary_modes=0)
    irreducibles = optimizer.perform_breadth_first_search()
    print(optimizer.report())
"""

import jax as _jax

# double precision everywhere: the oracle compares covariances at the 1e-10
# level, which float32 cannot resolve
_jax.config.update("jax_enable_x64", True)

from autogaussian.constraints import (
    CustomConstraint,
    EqualCooperativities,
    IsolationConstraint,
    MinimumTransmission,
    TransmissionConstraint,
)
from autogaussian.forward import (
    full_response,
    is_stable,
    max_real_eigenvalue,
    output_covariance_quadrature,
    response_matrices,
)
from autogaussian.graph import (
    GraphSpace,
    characterize_architectures,
    min_number_of_pumps,
    plot_graph,
    plot_list_of_graphs,
)
from autogaussian.nambu import (
    build_H_bdg,
    channel_covariance,
    duan_sum,
    squeezed_bath,
    symplectic_eigenvalues,
    thermal_bath,
    vacuum_covariance,
    variance_to_dB,
)
from autogaussian.optimizer import (
    CovarianceArchitectureOptimizer,
    find_minimum_number_auxiliary_modes,
)
from autogaussian.oracle import CovarianceOracle
from autogaussian.parametrization import Parametrization
from autogaussian.pipeline import discover
from autogaussian.postprocess import (
    complexity_table,
    fit_closed_form,
    rank_architectures,
    symbolic_regression,
)
from autogaussian.target import CovarianceTarget, P, Pin, X, qidx

__version__ = "0.1.0"

__all__ = [
    "CovarianceTarget", "Pin", "qidx", "X", "P",
    "discover", "CovarianceArchitectureOptimizer", "find_minimum_number_auxiliary_modes",
    "CovarianceOracle", "Parametrization", "GraphSpace",
    "TransmissionConstraint", "MinimumTransmission", "IsolationConstraint",
    "EqualCooperativities", "CustomConstraint",
    "response_matrices", "full_response", "output_covariance_quadrature",
    "max_real_eigenvalue", "is_stable",
    "build_H_bdg", "channel_covariance", "vacuum_covariance", "thermal_bath",
    "squeezed_bath", "variance_to_dB", "symplectic_eigenvalues", "duan_sum",
    "characterize_architectures", "min_number_of_pumps", "plot_graph",
    "plot_list_of_graphs", "complexity_table", "rank_architectures",
    "symbolic_regression", "fit_closed_form",
]
