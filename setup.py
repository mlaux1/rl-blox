from setuptools import setup

if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()

    setup(
        name="rl_blox",
        version="0.3.0",
        maintainer="Melvin Laux",
        maintainer_email="melvin.laux@uni-bremen.de",
        description="Modular RL implementations",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="Not public",
        packages=["rl_blox"],
        install_requires=[
            "numpy",
            "pytest",
            "torch",
            "gymnasium",
            "jax",
            "optax",
            "distrax",
            "chex",
            "tqdm",
        ],
    )
