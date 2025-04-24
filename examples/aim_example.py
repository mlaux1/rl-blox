from aim import Run

run = Run(
    repo="dummy",
    experiment="test-aim",
)

hparams_dict = {
    "learning_rate": 0.001,
    "batch_size": 32,
}

run["hparams"] = hparams_dict

for i in range(10):
    run.track(i, name="numbers")
    run.log_debug("This is a debug message!")
    run.log_info("This is an info message!")
