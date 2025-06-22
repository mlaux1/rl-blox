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
    """Keep track of policy evaluation and determine checkpointing and training.

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
        Number of training steps that should be taken now.
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
