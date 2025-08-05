[![Tests](https://github.com/mlaux1/rl-blox/actions/workflows/test.yaml/badge.svg)](https://github.com/mlaux1/rl-blox/actions/workflows/test.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![DOI](https://zenodo.org/badge/641058888.svg)](https://doi.org/10.5281/zenodo.15746631)

# RL-BLOX

<table>
  <tr>
    <td>
        This project contains modular implementations of various model-free and model-based RL algorithms and consists of deep neural network-based as well as tabular representation of Q-Values, policies, etc. which can be used interchangeably.
        The goal of this project is for the authors to learn by reimplementing various RL algorithms and to eventually provide an algorithmic toolbox for research purposes.
    </td>
    <td><img src="doc/source/_static/rl_blox_logo_v1.png" width=750px"/></td>
  </tr>
</table>

<img src="doc/source/_static/blox.svg" width="100%"/>

> [!CAUTION]
> This library is still experimental and under development. Using it may lead
> to experiencing bugs or changing interfaces. If you encounter any bugs or
> other issues, please let us know via the
> [issue tracker](https://github.com/mlaux1/rl-blox/issues). If you are an
> RL developer and want to collaborate, feel free to contact us.

## Design Principles

The implementation of this project follows the following principles:

1. Algorithms are functions!
2. Algorithms are implemented in single files.
3. Policies and values functions are data containers.

### Dependencies

1. Our environment interface is [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).
2. We use [JAX](https://github.com/jax-ml/jax) for everything.
3. We use [Chex](https://github.com/google-deepmind/chex) to write reliable code.
4. For optimization algorithms we use [Optax](https://github.com/google-deepmind/optax).
5. For probability distributions we use [TensorFlow Probability](https://www.tensorflow.org/probability).
6. For all neural networks we use [Flax NNX](https://github.com/google/flax).
7. To save checkpoints we use [Orbax](https://github.com/google/orbax).

## Installation

### Install via PyPI

The easiest way to install is via PyPI:
```bash
pip install rl-blox
```

### Install from source

Alternatively, e.g. if you want to develop extensions for the library, you can
also install rl-blox from source:

```bash
git clone git@github.com:mlaux1/rl-blox.git
```

After cloning the repository, it is recommended to install the library in editable mode.

```bash
pip install -e .
```

### Optional dependencies

To be able to run the provided examples use `pip install 'rl-blox[examples]'`.

To install development dependencies, please use `pip install 'rl-blox[dev]'`.

To enable logging with [aim](https://github.com/aimhubio/aim), please use `pip install 'rl_blox[logging]'`

You can install all optional dependencies (except logging) using `pip install 'rl_blox[all]'`.

## Algorithm Implementations

We currently provide implementations of the following algorithms (ordered from
SotA to classic RL algorithms):
MR.Q, TD7, TD3+LAP, PE-TS, SAC, TD3, DDPG, DDQN, DQN, double Q-learning,
CMA-ES, Dyna-Q, actor-critic, REINFORCE, Q-learning, MC.

## Getting Started

RL-BLOX relies on gymnasium's environment interface. This is an example with
the SAC RL algorithm.

```python
import gymnasium as gym

from rl_blox.algorithm.sac import create_sac_state, train_sac
from rl_blox.logging.checkpointer import OrbaxCheckpointer
from rl_blox.logging.logger import AIMLogger, LoggerList

env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
verbose = 1
env = gym.wrappers.RecordEpisodeStatistics(env)

hparams_models = dict(
    policy_hidden_nodes=[128, 128],
    policy_learning_rate=3e-4,
    q_hidden_nodes=[512, 512],
    q_learning_rate=1e-3,
    seed=seed,
)
hparams_algorithm = dict(
    total_timesteps=11_000,
    buffer_size=11_000,
    gamma=0.99,
    learning_starts=5_000,
)

if verbose:
    print(
        "This example uses the AIM logger. You will not see any output on "
        "stdout. Run 'aim up' to analyze the progress."
    )
checkpointer = OrbaxCheckpointer("/tmp/rl-blox/sac_example/", verbose=verbose)
logger = LoggerList([
    AIMLogger(),
    # uncomment to store checkpoints
    # checkpointer,
])
logger.define_experiment(
    env_name=env_name,
    algorithm_name="SAC",
    hparams=hparams_models | hparams_algorithm,
)
logger.define_checkpoint_frequency("policy", 1_000)

sac_state = create_sac_state(env, **hparams_models)
sac_result = train_sac(
    env,
    sac_state.policy,
    sac_state.policy_optimizer,
    sac_state.q,
    sac_state.q_optimizer,
    logger=logger,
    **hparams_algorithm,
)
env.close()
policy, _, q, _, _, _, _ = sac_result

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
