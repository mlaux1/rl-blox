import gymnasium as gym

from rl_blox.algorithm.mrq import create_mrq_state, train_mrq
from rl_blox.logging.logger import AIMLogger

env_name = "Pendulum-v1"
env = gym.make(env_name)

seed = 1
verbose = 2
env = gym.wrappers.RecordEpisodeStatistics(env)

hparams_models = dict(
    seed=seed,
)
hparams_algorithm = dict(
    seed=seed,
    total_timesteps=15_000,
    buffer_size=15_000,
    learning_starts=5_000,
)

if verbose:
    print(
        "This example uses the AIM logger. You will not see any output on "
        "stdout. Run 'aim up' to analyze the progress."
    )
logger = AIMLogger()
logger.define_experiment(
    env_name=env_name,
    algorithm_name="MR.Q",
    hparams=hparams_models | hparams_algorithm,
)

state = create_mrq_state(env, **hparams_models)

result = train_mrq(
    env,
    state.encoder,
    state.encoder_optimizer,
    state.policy,
    state.policy_optimizer,
    state.q,
    state.q_optimizer,
    state.the_bins,
    logger=logger,
    **hparams_algorithm,
)
env.close()

# TODO evaluation
