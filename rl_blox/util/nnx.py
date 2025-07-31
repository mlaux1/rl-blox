import flax
from flax import nnx


def Optimizer(model, optimizer):
    """Hack to support nnx.Optimizer in Flax >= 0.11.

    For the new optimizer interface, see
    https://flax.readthedocs.io/en/latest/migrating/nnx_010_to_nnx_011.html#nnx-0-10-to-nnx-0-11
    """
    version_parts = flax.__version__.split(".")
    major, minor = map(int, version_parts[:2])
    if major == 0 and minor < 11:
        return nnx.Optimizer(model, optimizer)
    elif major == 0 and minor >= 11:
        return nnx.ModelAndOptimizer(model, optimizer, wrt=nnx.Param)
    else:
        raise NotImplementedError(
            f"Unsupported Flax version: {flax.__version__}"
        )
