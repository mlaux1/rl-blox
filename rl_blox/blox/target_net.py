from functools import partial

import optax
from flax import nnx


@partial(nnx.jit, static_argnames=["tau"])
def soft_target_net_update(
    net: nnx.Module, target_net: nnx.Module, tau: float
) -> None:
    r"""Inplace (soft) update for target network with Polyak averaging.

    The soft update for the target network is supposed to be applied with about
    the same frequency as the update for the live network. Note that for a
    :math:`\tau` value of 1, this turns into a hard replacement of the target
    nets.

    Update formula:

    .. math::

        \theta' \leftarrow \tau \theta + (1 - \tau) \theta'

    where :math:`\theta` are the weights of the live network and
    :math:`\theta'` are the weights of the target network.

    Parameters
    ----------
    net : nnx.Module
        Live network with weights :math:`\theta`.

    target_net : nnx.Module
        Target network with weights :math:`\theta'`.

    tau : float
        The step size :math:`\tau`, i.e., the coefficient with which the live
        network's parameters will be multiplied. Must be in [0, 1]. Often
        :math:`\tau = 0.005` is used.
    """
    params = nnx.state(net)
    target_params = nnx.state(target_net)
    target_params = optax.incremental_update(params, target_params, tau)
    nnx.update(target_net, target_params)


@nnx.jit
def hard_target_net_update(net: nnx.Module, target_net: nnx.Module) -> None:
    r"""Inplace (hard) update for target network.

    The update completely replaces the weights of the target network with the
    weights of the live network. The hard update for the target network is
    supposed to be applied with a lower frequency than the update for the live
    network.

    Parameters
    ----------
    net : nnx.Module
        Live network with weights :math:`\theta`.

    target_net : nnx.Module
        Target network with weights :math:`\theta'`.
    """
    nnx.update(target_net, nnx.state(net))
