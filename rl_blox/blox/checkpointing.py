import dataclasses


@dataclasses.dataclass
class CheckpointState:
    """State of checkpoint monitoring."""

    episodes_since_udpate: int = 0
    """Number of assessment episodes since last actor update."""
    timesteps_since_upate: int = 0
    """Number of environment steps since last actor update."""
    max_episodes_before_update: int = 1
    """Maximum number of episodes allowed before next actor update."""
    min_return: float = 1e8
    """Minimum return observed for current actor."""
    best_min_return: float = -1e8
    """Best minimum return observed for any previous actor."""


def assess_performance_and_checkpoint(
    checkpoint_state: CheckpointState,
    steps_per_episode: int,
    episode_return: float,
    epoch: int,
    reset_weight: float,
    max_episodes_when_checkpointing: int,
    steps_before_checkpointing: int,
) -> tuple[bool, int]:
    """Monitor policy evaluation.

    The function will determine if a checkpoint should be saved and for how many
    epochs we should train.

    Parameters
    ----------
    checkpoint_state : CheckpointState
        State of checkpoint monitoring.

    steps_per_episode : int
        Number of steps in the previous episode that just finished.

    episode_return : float
        Return of the previous episode that just finished.

    epoch : int
        Training epoch counter.

    reset_weight : float
        When ``steps_before_checkpointing`` training epochs were reached, the
        best minimum return found so far will be multiplied by this factor, so
        that from now on, actors will be evaluated more thoroughly and updated
        only after ``max_episodes_when_checkpointing`` did not fall below the
        best minimum return found so far.

    max_episodes_when_checkpointing : int
        Configuration: maximum number of assessment episodes. In the beginning,
        this value is not used, but an initial value of 1 is used in
        ``checkpoint_state``. After ``steps_before_checkpointing``, the number
        of assessment episodes will be set to this value.

    steps_before_checkpointing : int
        Configuration: number of training epochs before checkpointing with
        ``max_episodes_when_checkpointing`` episodes starts.

    Returns
    -------
    update_checkpoint : False
        Checkpoint should be updated now.

    training_steps : int
        Number of training epochs.

    Notes
    -----

    A checkpoint is a snapshot of the parameters of a model, captured at a
    specific time during training. Using the checkpoint of a policy that
    obtained a high return during training improves the stability of test time
    performance. The standard learning paradigm for off-policy RL algorithms is
    to train after each time step, which means that the policy changes
    throughout each episode, making it hard to evaluate the performance.
    Similar to many on-policy algorithms, TD7 [1]_ keeps the policy fixed for
    several assessment episodes and then batches the training that would have
    occurred. We use these assessment episodes to judge if the current policy
    outperforms the previous best policy in a similar manner to evolutionary
    approaches and save a checkpoint. The checkpoint policy is used at
    evaluation time.

    The ideal performance measure of a policy is the average return in as many
    episodes as possible. However, it is necessary to reduce the number of
    evaluation episodes to improve sample efficiency. We use the minimum
    performance to assess unstable policies with a low number of episodes as
    poorly performing policies do not waste additional assessment episodes and
    training can resume when the performance in any episode falls below the
    checkpoint performance. This idea is first used in TD7 [1]_.

    The method implemented with this function distinguishes two phases of
    checkpointing. In the beginning, we only use one assessement episode per
    actor. After ``steps_before_checkpointing`` training epochs of the actor,
    we will switch to more thorough evaluation with a maximum of
    ``max_episodes_when_checkpointing`` episodes. Before we do so, we reduce
    the best minimum fitness of the previous checkpoint by multiplying it with
    ``reset_weight``, which should be between 0 and 1.

    References
    ----------
    .. [1] Fujimoto, S., Chang, W.D., Smith, E., Gu, S., Precup, D., Meger, D.
       (2023). For SALE: State-Action Representation Learning for Deep
       Reinforcement Learning. In Advances in Neural Information Processing
       Systems 36, pp. 61573-61624. Available from
       https://proceedings.neurips.cc/paper_files/paper/2023/hash/c20ac0df6c213db6d3a930fe9c7296c8-Abstract-Conference.html

    Examples
    --------

    The procedure works as follows:

    .. code-block:: python

        if terminated or truncated:  # episode done
            update_checkpoint, training_steps = maybe_train_and_checkpoint(
                checkpoint_state,
                steps_per_episode,  # number of steps in the last episode
                episode_return,  # accumulated reward of the last episode
                epoch,
                reset_weight,
                max_episodes_when_checkpointing,
                steps_before_checkpointing,
            )
            if update_checkpoint:
                update_actor_checkpoint(...)
            for _ in range(training_steps):
                train(...)
                epoch += 1
    """
    checkpoint_state.episodes_since_udpate += 1
    checkpoint_state.timesteps_since_upate += steps_per_episode
    checkpoint_state.min_return = min(
        checkpoint_state.min_return, episode_return
    )

    update_checkpoint = False
    training_steps = 0

    if checkpoint_state.min_return < checkpoint_state.best_min_return:
        # We know that the current actor is not better than the current
        # checkpoint. End evaluation of current actor early.
        training_steps = checkpoint_state.timesteps_since_upate
    elif (
        checkpoint_state.episodes_since_udpate
        == checkpoint_state.max_episodes_before_update
    ):
        # The current actor seems to be better, and we exhausted our
        # assessment budget. Update actor checkpoint and train actor.
        checkpoint_state.best_min_return = checkpoint_state.min_return
        update_checkpoint = True
        training_steps = checkpoint_state.timesteps_since_upate

    if training_steps > 0:
        # Switch to full checkpointing.
        if (
            epoch
            < steps_before_checkpointing
            <= epoch + checkpoint_state.timesteps_since_upate
        ):
            checkpoint_state.best_min_return *= reset_weight
            checkpoint_state.max_episodes_before_update = (
                max_episodes_when_checkpointing
            )

        # Reset checkpoint monitoring.
        checkpoint_state.episodes_since_udpate = 0
        checkpoint_state.timesteps_since_upate = 0
        checkpoint_state.min_return = 1e8

    return update_checkpoint, training_steps
