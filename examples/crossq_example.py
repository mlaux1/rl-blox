import gymnasium as gym

from rl_blox.algorithm.ext.crossq import train_crossq
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
    seed=seed,
)
logger = AIMLogger()
logger.define_experiment(
    env_name=env_name,
    algorithm_name="CrossQ",
    hparams=hparams,
)

model = train_crossq(env, logger=logger, **hparams)
env.close()

# Evaluation
env = gym.make(env_name, render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env)
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
