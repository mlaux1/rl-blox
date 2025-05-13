========================
Managing Memory with JAX
========================

The standard behavior of JAX is to preallocate 75% of the GPU memory.
Read more about it in the
`JAX documentation <https://docs.jax.dev/en/latest/gpu_memory_allocation.html>`_.
This behavior can be deactivated through environment variables:

.. code-block:: bash

    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.XX

We can also do this in Python code before we import JAX:

.. code-block:: python

    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".XX"
