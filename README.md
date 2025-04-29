[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

# RL-BLOX
<p float="center">
    <img src="doc/source/_static/rl_blox_logo_v1.png" height="150px" />
</p>

This project contains modular implementations of various model-free and model-based RL algorithms and consists of deep neural network-based as well as tabular representation of Q-Values, policies, etc. which can be used interchangeably.
The goal of this project is for the authors to learn by reimplementing various RL algorithms and to eventually provide an algorithmic toolbox for research purposes.

> [!CAUTION]
> This library is still experimental and under development. Using it will not
> result in a good user experience. It is not well-documented, it is buggy,
> its interface is not clearly defined, its most interesting features are in
> feature branches. We recommend not to use it now. If you are an RL developer
> and want to collaborate, feel free to contact us.

## Design Principles

The implementation of this project follows the following principles:
1. Algorithms are functions!
2. Algorithms are implemented in single files.
3. Policies and values functions are data containers.

### Dependencies

1. Our environment interface is Gymnasium.
2. We use JAX for everything.
3. We use Chex to write reliable code.
4. For optimization algorithms we use Optax.
5. For probability distributions we use Distrax.
6. For all neural networks we use Flax NNX.
7. To save checkpoints we use Orbax.

## Installation

```bash
git clone git@github.com:mlaux1/rl-blox.git
```

After cloning the repository, it is recommended to install the library in editable mode.

```bash
pip install -e .
```

To be able to run the provided examples use `pip install -e '.[examples]'`.
To install development dependencies, please use `pip install -e '.[dev]'`.
You can install all optional dependencies using `pip install -e '.[all]'`.

## Getting Started

RL-BLOX relies on gymnasium's environment interface. This is an example with
the SAC RL algorithm.

```python
import gymnasium as gym

from rl_blox.algorithms.model_free.sac import (
    GaussianMlpPolicyNetwork,
    SoftMlpQNetwork,
    train_sac,
)

env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)
envs = gym.vector.SyncVectorEnv([lambda: env])

policy = GaussianMlpPolicyNetwork.create([256, 256], envs)
q = SoftMlpQNetwork(hidden_nodes=[256, 256])

policy, policy_params, q, q1_params, q2_params = train_sac(
    envs,
    policy,
    q,
    seed=seed,
    total_timesteps=8_000,
    buffer_size=1_000_000,
    gamma=0.99,
    learning_starts=5_000,
)
envs.close()

# Do something with the trained policy...
```

## API Documentation

You can build the sphinx documentation with

```bash
pip install -e '.[doc]'
cd doc
make html
```

The HTML documentation will be available under `doc/build/html/index.html`.

## Contributing

If you wish to report bugs, please use the [issue tracker](https://github.com/mlaux1/rl-blox/issues). If you would like to contribute to RL-BLOX, just open an issue or a
[pull request](https://github.com/mlaux1/rl-blox/pulls). The target branch for
merge requests is the development branch. The development branch will be merged to master for new releases. If you have
questions about the software, you should ask them in the discussion section.

The recommended workflow to add a new feature, add documentation, or fix a bug is the following:
- Push your changes to a branch (e.g. feature/x, doc/y, or fix/z) of your fork of the RL-BLOX repository.
- Open a pull request to the main branch.

It is forbidden to directly push to the main branch.

## Testing

Run the tests with

```bash
pip install -e '.[dev]'
pytest
```

## Releases

### Semantic Versioning

Semantic versioning must be used, that is, the major version number will be incremented when the API changes in a backwards incompatible way, the minor version will be incremented when new functionality is added in a backwards compatible manner, and the patch version is incremented for bugfixes, documentation, etc.


## Funding

This library is currently developed at the [Robotics Group](https://robotik.dfki-bremen.de/en/about-us/university-of-bremen-robotics-group.html) of the
[University of Bremen](http://www.uni-bremen.de/en.html) together with the
[Robotics Innovation Center](http://robotik.dfki-bremen.de/en/startpage.html) of the
[German Research Center for Artificial Intelligence (DFKI)](http://www.dfki.de) in Bremen.

<p float="left">
    <img src="doc/source/_static/Uni_Logo.png" height="100px" />
    <img src="doc/source/_static/DFKI_Logo.png" height="100px" />
</p>
