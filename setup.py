from setuptools import setup

if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()

    setup(
        name="rl_blox",
        version="0.3.1",
        maintainer="Melvin Laux",
        maintainer_email="melvin.laux@uni-bremen.de",
        description="Modular RL implementations",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="Not public",
        packages=["rl_blox"],
        install_requires=[
            "numpy",
            "gymnasium",
            "jax",
            "optax",
            "distrax",
            "chex",
            "tqdm",
            "flax",
        ],
        extras_require={
            "examples": ["gymnasium[classic-control,toy-text,mujoco]"],
            "dev": ["pytest", "pre-commit", "flake8"],
        },
    )
