from setuptools import setup


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()

    setup(name='modular_rl',
          version="0.1.0",
          maintainer='Melvin Laux',
          maintainer_email='melvin.laux@dfki.de',
          description='Modular RL implementations',
          long_description=long_description,
          long_description_content_type="text/markdown",
          license='Not public',
          packages=["modular_rl"],
          install_requires=["numpy",
                            "pytest",
                            "torch",
                            "gymnasium"
                            ])
