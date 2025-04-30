.. _api:

=================
API Documentation
=================

:mod:`rl_blox.algorithms.model_free`
====================================

.. automodule:: rl_blox.algorithms.model_free
    :no-members:
    :no-inherited-members:

Algorithm Interface
-------------------

.. autosummary::
   :toctree: _apidoc/

   ~q_learning.q_learning
   ~reinforce.train_reinforce_epoch
   ~actor_critic.train_ac_epoch
   ~ddpg.train_ddpg
   ~sac.train_sac

Functional Blox
---------------

.. autosummary::
   :toctree: _apidoc/

   ~reinforce.discounted_reward_to_go
   ~reinforce.reinforce_gradient
   ~reinforce.policy_gradient_pseudo_loss
   ~reinforce.mse_value_loss
   ~actor_critic.actor_critic_policy_gradient
   ~ddpg.mse_action_value_loss
   ~ddpg.deterministic_policy_value_loss
   ~ddpg.update_target
   ~ddpg.ddpg_update_actor
   ~ddpg.ddpg_update_critic
   ~sac.sac_actor_loss
   ~sac.sac_exploration_loss
   ~sac.sac_update_actor
   ~sac.sac_update_critic
   ~double_q_learning.double_q_learning

Data Blox
---------

.. autosummary::
   :toctree: _apidoc/

   ~reinforce.MLP
   ~reinforce.GaussianMLP
   ~reinforce.StochasticPolicyBase
   ~reinforce.GaussianPolicy
   ~reinforce.SoftmaxPolicy
   ~ddpg.ReplayBuffer
   ~ddpg.MLP
   ~ddpg.DeterministicPolicy
   ~sac.GaussianMLP
   ~sac.StochasticPolicyBase
   ~sac.GaussianPolicy
   ~sac.EntropyCoefficient
   ~sac.EntropyControl


:mod:`rl_blox.algorithms.model_based`
=====================================

.. automodule:: rl_blox.algorithms.model_based
    :no-members:
    :no-inherited-members:

Algorithm Interface
-------------------

.. autosummary::
   :toctree: _apidoc/

   ~pets.train_pets

Functional Blox
---------------

.. autosummary::
   :toctree: _apidoc/

   ~pets.mpc_action
   ~pets.ts_inf
   ~pets.evaluate_plans
   ~pets.update_dynamics_model

Data Blox
---------

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~pets.PETSMPCConfig
   ~pets.PETSMPCState


:mod:`rl_blox.logging`
======================

.. automodule:: rl_blox.logging
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~logger.LoggerBase
   ~logger.StandardLogger
   ~logger.AIMLogger
   ~logger.LoggerList
