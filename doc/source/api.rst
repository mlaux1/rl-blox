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
   dynaq.train_dynaq
   dqn.train_dqn
   reinforce.train_reinforce
   actor_critic.train_ac
   ddpg.train_ddpg
   td3.train_td3
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
   actor_critic.actor_critic_policy_gradient
   ddpg.sample_actions
   ddpg.mse_action_value_loss
   ddpg.deterministic_policy_value_loss
   ddpg.ddpg_update_actor
   ddpg.ddpg_update_critic
   ddpg.q_deterministic_bootstrap_estimate
   td3.td3_update_critic
   td3.double_q_deterministic_bootstrap_estimate
   td3.sample_target_actions
   sac.sac_actor_loss
   sac.sac_exploration_loss
   sac.sac_update_actor
   sac.sac_update_critic
   sac.sac_q_target
   double_q_learning.double_q_learning
   pets.mpc_action
   pets.ts_inf
   pets.evaluate_plans
   pets.update_dynamics_model

Data Blox
---------

.. autosummary::
   :toctree: _apidoc/

   reinforce.StochasticPolicyBase
   reinforce.GaussianPolicy
   reinforce.SoftmaxPolicy
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

   target_net.soft_target_net_update
   cross_entropy_method.cem_sample
   cross_entropy_method.cem_update
   probabilistic_ensemble.gaussian_nll
   probabilistic_ensemble.bootstrap
   probabilistic_ensemble.gaussian_ensemble_loss
   probabilistic_ensemble.train_ensemble
   probabilistic_ensemble.restore_checkpoint
   losses.mse_value_loss
   losses.stochastic_policy_gradient_pseudo_loss

Data Blox
---------

.. autosummary::
   :toctree: _apidoc/

   replay_buffer.ReplayBuffer
   function_approximator.mlp.MLP
   function_approximator.gaussian_mlp.GaussianMLP
   function_approximator.policy_head.DeterministicTanhPolicy
   function_approximator.policy_head.StochasticPolicyBase
   function_approximator.policy_head.GaussianTanhPolicy
   function_approximator.policy_head.GaussianPolicy
   function_approximator.policy_head.SoftmaxPolicy
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
