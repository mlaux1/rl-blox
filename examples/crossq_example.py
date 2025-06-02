import gymnasium as gym

from rl_blox.algorithm.ext.crossq import (
    OrbaxLinenCheckpointer,
    load_checkpoint,
    train_crossq,
)
from rl_blox.logging.logger import AIMLogger

env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
verbose = 1
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)

if verbose:
    print(
        "This example uses the AIM logger. You will not see any output on "
        "stdout. Run 'aim up' to analyze the progress."
    )
hparams = dict(
    total_timesteps=15_000,
    algo="crossq",
    log_interval=1,
    seed=seed,
)
logger = AIMLogger()
logger.define_experiment(
    env_name=env_name,
    algorithm_name="CrossQ",
    hparams=hparams,
)

model, policy, q = train_crossq(env, logger=logger, **hparams)
env.close()

checkpointer = OrbaxLinenCheckpointer(
    checkpoint_dir="/tmp/rl-blox/crossq_example"
)
# uncomment to store checkpoint
# checkpointer.save_model(f"{checkpointer.checkpoint_dir}/policy", policy)
# checkpointer.save_model(f"{checkpointer.checkpoint_dir}/q", q)

env = gym.make(env_name, render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env)
# uncomment to load checkpoint
# model = load_checkpoint(
#     env,
#     policy_path=f"{checkpointer.checkpoint_dir}/policy",
#     q_path=f"{checkpointer.checkpoint_dir}/q",
#     algo="crossq",
# )

# Evaluation
while True:
    done = False
    infos = {}
    obs, _ = env.reset()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        obs = next_obs
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
