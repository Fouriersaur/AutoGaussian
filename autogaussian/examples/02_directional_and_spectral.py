"""
Example 2 -- the two things a covariance target can do that an S target cannot.

    B.3  directional squeezed source  -- asymmetry *between* the port blocks,
         at finite cooperativity, enforced together with a transmission pin so
         the minimal device stays connected (Sec. 11).
    B.4  broadband squeezer           -- point + Omega-derivative pins; the
         decay ratios kappa~_i are what flattens the band (App. A.4(c)).

Usage:  python examples/02_directional_and_spectral.py [--plot]
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogaussian import CovarianceArchitectureOptimizer, variance_to_dB
from autogaussian.gallery import broadband_squeezer, directional_squeezed_source


def banner(text):
    print("\n" + "=" * 72 + "\n" + text + "\n" + "=" * 72)


def directional_source():
    banner("B.3  directional squeezed source: how many auxiliary modes are needed?")
    problem = directional_squeezed_source(variance=0.2, forward_transmission=1.3,
                                          isolate_backward=True)

    # Sec. 6(a): grow the auxiliary count until the fully connected graph works.
    # Two directly coupled modes cannot be made directional by any coupling
    # phase alone (mode-swap symmetry), so 0 auxiliaries must fail here.
    optimizer = None
    for num_auxiliary_modes in (0, 1):
        candidate = CovarianceArchitectureOptimizer(
            problem.target, num_auxiliary_modes=num_auxiliary_modes, seed=3,
            make_initial_test=False, kwargs_optimization={"num_tests": 30},
            **problem.optimizer_kwargs())
        success, infos = candidate.test_graph(candidate.space.fully_connected())
        print("  %i auxiliary modes: %-8s best loss %.2g"
              % (num_auxiliary_modes, "VALID" if success else "INVALID",
                 min(info["loss_reached"] for info in infos)))
        if success:
            optimizer = candidate
            break
    if optimizer is None:
        return None

    x = infos[-1]["x"]
    V = np.real(optimizer.oracle.covariance(x, 0.0))
    S, _ = optimizer.oracle.scattering(x, 0.0)
    print("\nport covariance at Omega = 0:")
    print(np.round(V, 4))
    print("\nport 1 (vacuum target): diag = %.4f, %.4f" % (V[0, 0], V[1, 1]))
    print("port 2 squeezing:       %.2f dB  (variance %.4f)"
          % (variance_to_dB(V[2, 2]), V[2, 2]))
    print("transport: forward |S_10|^2 = %.3f, backward |S_01|^2 = %.3f"
          % (abs(S[1, 0]) ** 2, abs(S[0, 1]) ** 2))
    print("stability margin: max Re eig(M~) = %.3f" % infos[-1]["max_real_eigenvalue"])
    return optimizer, x


def broadband_source(plot=False, bandwidth=0.5):
    banner("B.4  broadband squeezer: sigma_xx = 0.5 flat across |Omega| <= %.2f" % bandwidth)
    # derivative pins alone only constrain the spectrum *at* Omega = 0 -- the
    # optimiser can satisfy them by shrinking kappa~ instead of widening the
    # band, so pin the band itself as well
    problem = broadband_squeezer(variance=0.5, num_derivatives=2, bandwidth=bandwidth)
    print(problem.target)

    optimizer = CovarianceArchitectureOptimizer(
        problem.target, num_auxiliary_modes=1, seed=1, make_initial_test=False,
        kwargs_optimization={"num_tests": 25}, **problem.optimizer_kwargs())
    success, infos = optimizer.test_graph(optimizer.space.fully_connected())
    print("fully connected graph:", "VALID" if success else "INVALID",
          " loss %.2g" % infos[-1]["loss_reached"])
    if not success:
        return None

    x = infos[-1]["x"]
    omegas = np.linspace(-1.0, 1.0, 21)
    spectrum = np.array([np.real(optimizer.oracle.covariance(x, w)[0, 0]) for w in omegas])

    print("\nsigma_xx(Omega):")
    for omega, value in zip(omegas[::2], spectrum[::2]):
        bar = "#" * int(round(40 * value))
        print("  Omega = %+5.2f   %.4f  %s" % (omega, value, bar))
    print("\ndecay ratios found:", optimizer.param.physical_report(x)["decay_ratios"])

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(omegas, spectrum, "o-", label="flat-band solution")
        plt.axhline(1.0, color="k", ls=":", label="vacuum floor")
        plt.xlabel(r"$\Omega = \omega/\kappa_{\rm ref}$")
        plt.ylabel(r"$\sigma_{xx}(\Omega)$")
        plt.legend()
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "broadband_spectrum.png")
        plt.savefig(path, dpi=140, bbox_inches="tight")
        print("wrote %s" % path)
    return optimizer, x


if __name__ == "__main__":
    directional_source()
    broadband_source(plot="--plot" in sys.argv)
