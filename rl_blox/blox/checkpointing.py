import dataclasses


@dataclasses.dataclass
class CheckpointState:
    episodes_since_udpate: int = 0
    timesteps_since_upate: int = 0
    max_episodes_before_update: int = 1
    min_return: float = 1e8
    best_min_return: float = -1e8


def maybe_train_and_checkpoint(
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

    The ideal performance measure of a policy is the average return in as many
    episodes as possible. However, it is necessary to reduce the number of
    evaluation episodes to improve sample efficiency. We use the minimum
    performance to assess unstable policies with a low number of episodes as
    poorly performing policies do not waste additional assessment episodes and
    training can resume when the performance in any episode falls below the
    checkpoint performance. This idea is first used in TD7 [1]_.

    Parameters
    ----------
    checkpoint_state : CheckpointState
        State of checkpoint monitoring.

    steps_per_episode : int
        Steps taken in the last episode.

    episode_return : float
        Return of the last episode.

    epoch : int
        Training epoch counter.

    reset_weight : float
        Criteria reset weight.

    max_episodes_when_checkpointing : int
        Maximum number of assessment episodes.

    steps_before_checkpointing : int
        Maximum number of timesteps before checkpointing.

    Returns
    -------
    update_checkpoint : False
        Checkpoint should be updated now

    training_steps : int
        Number of training epochs.

    References
    ----------
    .. [1] Fujimoto, S., Chang, W.D., Smith, E., Gu, S., Precup, D., Meger, D.
       (2023). For SALE: State-Action Representation Learning for Deep
       Reinforcement Learning. In Advances in Neural Information Processing
       Systems 36, pp. 61573-61624. Available from
       https://proceedings.neurips.cc/paper_files/paper/2023/hash/c20ac0df6c213db6d3a930fe9c7296c8-Abstract-Conference.html
    """
    checkpoint_state.episodes_since_udpate += 1
    checkpoint_state.timesteps_since_upate += steps_per_episode
    checkpoint_state.min_return = min(
        checkpoint_state.min_return, episode_return
    )

    update_checkpoint = False
    training_steps = 0

    if checkpoint_state.min_return < checkpoint_state.best_min_return:
        # End evaluation of current policy early
        training_steps = checkpoint_state.timesteps_since_upate
    elif (
        checkpoint_state.episodes_since_udpate
        == checkpoint_state.max_episodes_before_update
    ):
        # Update checkpoint and train
        checkpoint_state.best_min_return = checkpoint_state.min_return
        update_checkpoint = True
        training_steps = checkpoint_state.timesteps_since_upate

    if training_steps > 0:
        for i in range(checkpoint_state.timesteps_since_upate):
            if epoch + i + 1 == steps_before_checkpointing:
                checkpoint_state.best_min_return *= reset_weight
                checkpoint_state.max_episodes_before_update = (
                    max_episodes_when_checkpointing
                )

        checkpoint_state.episodes_since_udpate = 0
        checkpoint_state.timesteps_since_upate = 0
        checkpoint_state.min_return = 1e8

    return update_checkpoint, training_steps
