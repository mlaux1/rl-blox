# monkey-patch for flax 0.10.x to work with 0.11.x optimizer interface
import flax

if flax.__version__.startswith("0.10."):
    from flax import nnx

    flax_0_10_style_update = nnx.Optimizer.update

    def flax_0_11_style_update(self, model, grads, **kwargs):
        # ignore model
        return flax_0_10_style_update(self, grads, **kwargs)

    nnx.Optimizer.update = flax_0_11_style_update

__version__ = "0.5.3"
