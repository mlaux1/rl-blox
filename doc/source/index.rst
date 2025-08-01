.. RL-BLOX documentation master file, created by
   sphinx-quickstart on Mon Mar 31 16:33:37 2025.

RL-BLOX
=======


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   aim
   memory
   api


Interface
---------

The implementation of this project follows the following principles:

1. Algorithms are functions!
2. Algorithms are implemented in single files.
3. Policies and values functions are data containers.

RL-BLOX provides a two-level API:

1. Algorithm interface: a high-level interface for RL algorithms that provides
   a collection of preimplemented algorithms based on the low level interface.
2. Toolbox interface (aka blox): a low-level interface, which provides a
   collection of building blocks to easily create RL algorithms, e.g.,
   value networks, exploration strategies, loss functions, replay buffers,
   environment models, etc.

Algorithm Interface
^^^^^^^^^^^^^^^^^^^

The algorithm interface defines a function ``train_ALGORITHM(...)`` for each
algorithm.

The arguments of the function are usually defined in the following order:

* ``env`` - The first argument is the Gymnasium environment.
* Followed by data containers like function approximators and optimizers.
* (Hyper-)parameters of the algorithm. These are native types to enable
  hyperparameter tuning and experiment tracking. Native types are
  ``bool``, ``int``, ``float``, ``tuple``, ``list``, ``dict``.
  A callable is not a native type.

  * Parameters that are present in almost all algorithms are
    ``total_timesteps`` and ``seed``.
  * We use the more self-explanatory name ``learning_rate`` for alpha.
* ``logger`` - At the end, we allow to pass objects like loggers, that are not
  essential for the algorithm, but for the tooling around training.

The main training loop of an algorithm runs for ``total_timesteps`` as the
default termination condition as this will ensure that algorithms will have
the specified number of interactions with the environment, which might not be
the case in environments with variable episode lengths otherwise and number of
interactions is a key metric when evaluating sample efficiency.
However, if algorithms state a specific termination condition, then this is the
one we use in the algorithm interface.

All optional inputs have clearly defined default values, ideally to have the
default behavior as close as possible to the original paper, e.g., using Rprop
as an optimizer in DQN.

The train function of an algorithm will return the internal state of the
algorithm as a namedtuple. The internal state might, for instance, consist of
policy, value function, target networks, optimizers, replay buffer.

Note that full experiment evaluation, tracking, and hyperparameter tuning
is not something that this library is meant to provide and remains separate.
However, we provide some basic logging and experiment tracking, e.g., with AIM
and we ensure that RL-BLOX can easily be use in libraries that provide these
features, e.g., optuna or hyperopt.

BLOX
^^^^

The building blocks of the low-level API are documented in the :ref:`api`.

.. figure:: _static/blox.svg
   :alt: RL-BLOX
   :align: center
   :width: 100%

   Illustrations of blox and how they can be used to assemble RL algorithms.

Dependencies
------------

* Our environment interface is Gymnasium.
* We use JAX for everything.
* We use Chex to write reliable code.
* For optimization algorithms we use Optax.
* For probability distributions we use TensorFlow Probability.
* For all neural networks we use Flax NNX.
* To save checkpoints we use Orbax.

Repository Structure
--------------------

.. code-block::

   ├── examples
   ├── rl_blox
   │ ├── algorithm (contains high level algorithm API)
   │ ├── blox (contains low level toolbox API)
   │ ├── logging (contains logging with AIM)
   │ ├── util (contains all misc utilities, currently in helper and tools)
   ├── tests
   └── ...
