import glob
import os
import shutil

from flax import nnx

from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.logging.checkpointer import OrbaxCheckpointer


def test_orbax_checkpointer():
    model = MLP(2, 1, [10], "relu", rngs=nnx.Rngs(42))
    checkpoint_dir = "/tmp/rl-blox-test/"
    model_name = "mlp"

    try:
        checkpointer = OrbaxCheckpointer(checkpoint_dir=checkpoint_dir)
        checkpointer.define_experiment("TestEnv", "TestAlg")
        checkpointer.define_checkpoint_frequency(model_name, 1)

        for _ in range(10):
            checkpointer.record_epoch(model_name, model)

        checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "*/"))
        for cp in checkpoint_paths:
            assert cp in checkpointer.checkpoint_path[model_name]
    finally:
        shutil.rmtree(checkpoint_dir)
