.. _api:

=================
API Documentation
=================

:mod:`rl_blox.algorithms.model_free`
====================================

.. automodule:: rl_blox.algorithms.model_free
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/

   ~q_learning.q_learning
   ~reinforce.train_reinforce_epoch
   ~reinforce.discounted_reward_to_go
   ~reinforce.reinforce_gradient
   ~reinforce.policy_gradient_pseudo_loss
   ~reinforce.mse_value_loss
   ~actor_critic.train_ac_epoch
   ~actor_critic.actor_critic_policy_gradient
   ~ddpg.train_ddpg
   ~ddpg.mse_action_value_loss
   ~ddpg.deterministic_policy_value_loss
   ~ddpg.update_target
   ~sac.train_sac
   ~sac.sac_actor_loss
   ~sac.sac_exploration_loss
   ~double_q_learning.double_q_learning


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
