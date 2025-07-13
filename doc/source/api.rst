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

   q_learning.train_q_learning
   double_q_learning.train_double_q_learning
   dynaq.train_dynaq
   dqn.train_dqn
   ddqn.train_ddqn
   nature_dqn.train_nature_dqn
   reinforce.train_reinforce
   actor_critic.train_ac
   ddpg.train_ddpg
   td3.train_td3
   td3_lap.train_td3_lap
   td7.train_td7
   sac.train_sac
   pets.train_pets
   cmaes.train_cmaes

Functional Blox
---------------

.. autosummary::
   :toctree: _apidoc/

   dqn.greedy_policy
   dqn.train_step_with_loss
   reinforce.discounted_reward_to_go
   reinforce.reinforce_gradient
   actor_critic.actor_critic_policy_gradient
   ddpg.sample_actions
   ddpg.ddpg_update_actor
   td3.sample_target_actions
   td7.td7_update_actor
   td7.deterministic_policy_gradient_loss_sale
   td7.td7_update_critic
   sac.sac_actor_loss
   sac.sac_exploration_loss
   sac.sac_update_actor
   sac.sac_update_critic
   sac.soft_q_target
   pets.mpc_action
   pets.ts_inf
   pets.evaluate_plans
   pets.update_dynamics_model
   cmaes.flat_params
   cmaes.set_params

Data Blox
---------

.. autosummary::
   :toctree: _apidoc/

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

   embedding.sale.state_action_embedding_loss
   embedding.sale.update_sale
   checkpointing.assess_performance_and_checkpoint
   target_net.soft_target_net_update
   target_net.hard_target_net_update
   cross_entropy_method.cem_sample
   cross_entropy_method.cem_update
   probabilistic_ensemble.gaussian_nll
   probabilistic_ensemble.bootstrap
   probabilistic_ensemble.gaussian_ensemble_loss
   probabilistic_ensemble.train_ensemble
   probabilistic_ensemble.restore_checkpoint
   losses.stochastic_policy_gradient_pseudo_loss
   losses.deterministic_policy_gradient_loss
   losses.mse_value_loss
   losses.mse_continuous_action_value_loss
   losses.ddpg_loss
   losses.td3_loss
   losses.td3_lap_loss
   losses.mse_discrete_action_value_loss
   losses.dqn_loss
   losses.nature_dqn_loss
   losses.ddqn_loss
   function_approximator.norm.avg_l1_norm
   replay_buffer.lap_priority

Data Blox
---------

.. autosummary::
   :toctree: _apidoc/

   embedding.sale.SALE
   embedding.sale.ActorSALE
   embedding.sale.CriticSALE
   embedding.sale.DeterministicSALEPolicy
   checkpointing.CheckpointState
   replay_buffer.ReplayBuffer
   replay_buffer.LAP
   function_approximator.mlp.MLP
   function_approximator.gaussian_mlp.GaussianMLP
   function_approximator.policy_head.DeterministicTanhPolicy
   function_approximator.policy_head.StochasticPolicyBase
   function_approximator.policy_head.GaussianTanhPolicy
   function_approximator.policy_head.GaussianPolicy
   function_approximator.policy_head.SoftmaxPolicy
   double_qnet.ContinuousClippedDoubleQNet
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
