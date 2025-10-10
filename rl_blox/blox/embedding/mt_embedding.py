import jax.numpy as jnp
from flax import nnx

default_task_embedding_init = nnx.initializers.normal(stddev=1.0)


def embedding_renorm(embedding, max_norm):
    """Renormalizes the embedding to have a maximum norm of max_norm.

    This mimics the behavior of torch.nn.Embedding with max_norm.

    Parameters
    ----------
    embedding : nnx.Embed
        The embedding to renormalize.

    max_norm : float
        The maximum norm to enforce.
    """
    norm = jnp.linalg.norm(embedding.embedding.value, axis=1)[:, jnp.newaxis]
    scale = max_norm / (norm + 1e-7)
    embedding.embedding.value = jnp.where(
        norm > max_norm,
        embedding.embedding.value * scale,
        embedding.embedding.value,
    )
