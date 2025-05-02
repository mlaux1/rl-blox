.. _api:

=================
API Documentation
=================

:mod:`rl_blox.algorithm`
========================

.. automodule:: rl_blox.algorithm
    :no-members:
    :no-inherited-members:

Algorithm Interface
-------------------

.. autosummary::
   :toctree: _apidoc/

   q_learning.q_learning
   dqn.train_dqn
   reinforce.train_reinforce
   actor_critic.train_ac
   ddpg.train_ddpg
   sac.train_sac
   pets.train_pets

Functional Blox
---------------

.. autosummary::
   :toctree: _apidoc/

   dqn.critic_loss
   dqn.greedy_policy
   reinforce.discounted_reward_to_go
   reinforce.reinforce_gradient
   reinforce.policy_gradient_pseudo_loss
   reinforce.mse_value_loss
   actor_critic.actor_critic_policy_gradient
   ddpg.mse_action_value_loss
   ddpg.deterministic_policy_value_loss
   ddpg.update_target
   ddpg.ddpg_update_actor
   ddpg.ddpg_update_critic
   sac.sac_actor_loss
   sac.sac_exploration_loss
   sac.sac_update_actor
   sac.sac_update_critic
   double_q_learning.double_q_learning
   pets.mpc_action
   pets.ts_inf
   pets.evaluate_plans
   pets.update_dynamics_model

Data Blox
---------

.. autosummary::
   :toctree: _apidoc/

   dqn.MLP
   reinforce.GaussianMLP
   reinforce.StochasticPolicyBase
   reinforce.GaussianPolicy
   reinforce.SoftmaxPolicy
   ddpg.ReplayBuffer
   ddpg.MLP
   ddpg.DeterministicPolicy
   sac.GaussianMLP
   sac.StochasticPolicyBase
   sac.GaussianPolicy
   sac.EntropyCoefficient
   sac.EntropyControl
   pets.PETSMPCConfig
   pets.PETSMPCState


:mod:`rl_blox.blox`
===================

.. automodule:: rl_blox.blox
    :no-members:
    :no-inherited-members:

Functional Blox
---------------

.. autosummary::
   :toctree: _apidoc/

   cross_entropy_method.cem_sample
   cross_entropy_method.cem_update
   probabilistic_ensemble.gaussian_nll
   probabilistic_ensemble.bootstrap
   probabilistic_ensemble.gaussian_ensemble_loss
   probabilistic_ensemble.train_ensemble
   probabilistic_ensemble.restore_checkpoint

Data Blox
---------

.. autosummary::
   :toctree: _apidoc/

   function_approximator.mlp.MLP
   probabilistic_ensemble.GaussianMLP
   probabilistic_ensemble.GaussianMLPEnsemble
   probabilistic_ensemble.EnsembleTrainState


:mod:`rl_blox.logging`
======================

.. automodule:: rl_blox.logging
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   logger.LoggerBase
   logger.StandardLogger
   logger.AIMLogger
   logger.LoggerList
