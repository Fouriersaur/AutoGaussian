from setuptools import find_packages, setup

with open("README.md") as handle:
    long_description = handle.read()

setup(
    name="autogaussian",
    version="0.1.0",
    description="Automated discovery of coupled-mode setups from output-covariance targets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "examples"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "jax",
        "jaxlib",
        "sympy",
        "networkx",
        "matplotlib",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
