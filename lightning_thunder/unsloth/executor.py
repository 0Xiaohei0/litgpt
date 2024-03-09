# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import thunder
import thunder.torch as ltorch
from thunder.core.proxies import TensorProxy
from thunder.core.transforms import grad, put_grads, get_grad

import torch
import numpy as np
from thunder.extend import OperatorExecutor, register_executor
import lightning_thunder.unsloth.kernels as kernels
from thunder.torch import ne, sum, true_divide


unsloth_ex = OperatorExecutor("unsloth_ex", version="0.1")
register_executor(unsloth_ex)


def unsloth_cross_entropy_like(logits, labels):
    return (
        thunder.TensorProxy(
            shape=(logits.shape[0],),
            # the cross entropy kernel only supports float32
            dtype=thunder.dtypes.float32,
            device=logits.device,
            requires_grad=logits.requires_grad,
        ),
        thunder.TensorProxy(
            shape=(logits.shape[0],), dtype=thunder.dtypes.float32, device=logits.device, requires_grad=False
        ),
    )


u_cross_entropy_loss = unsloth_ex.register_operator(
    "unsloth_cross_entropy_loss",
    like=unsloth_cross_entropy_like,
    fn=kernels.cross_entropy_loss._cross_entropy_forward_impl,
)


def unsloth_cross_entropy_backward_impl(dlosses, logits, logsumexp, labels):
    # clone() because the kernel writes the grads in the logits.
    # If it works, we can remove this it, but it's not a thing we generally anticipate and support right now.
    return kernels.cross_entropy_loss._cross_entropy_backward_impl(dlosses, logits.clone(), logsumexp, labels)


def unsloth_cross_entropy_backward_like(dlosses, logits, logsumexp, labels):
    return thunder.TensorProxy(like=logits)


u_cross_entropy_loss_backward = unsloth_ex.register_operator(
    "unsloth_cross_entropy_loss_backward",
    like=unsloth_cross_entropy_backward_like,
    fn=unsloth_cross_entropy_backward_impl,
)


def unsloth_cross_entropy_grad(
    logits: TensorProxy,
    labels: TensorProxy,
    weight: TensorProxy | None = None,
    size_average: bool | None = None,
    ignore_index: int = -100,
    reduce: bool | None = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
):
    assert unsloth_cross_entropy_checker(**locals())
    loss, logsumexp = u_cross_entropy_loss(logits, labels)
    if reduction == "mean":
        # "mean" reduction is not part of the kernel
        n_items = sum(ne(labels, -100))
        loss = true_divide(sum(loss), n_items)

    g = get_grad(loss)

    if reduction == "mean":
        from thunder.core.transforms import mean_backward

        g = mean_backward(logsumexp.ndim, logsumexp.shape, (0,), g)

    logits_grad = u_cross_entropy_loss_backward(g, logits, logsumexp, labels)
    put_grads((logits,), (logits_grad,))

    return loss


def unsloth_cross_entropy_checker(
    logits: TensorProxy,
    labels: TensorProxy,
    weight: TensorProxy | None = None,
    size_average: bool | None = None,
    ignore_index: int = -100,
    reduce: bool | None = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
):
    return (
        weight is None
        and size_average is None
        and reduce is None
        and reduction in ("none", "mean")
        and ignore_index == -100
        and label_smoothing == 0.0
        and logits.device.type == "cuda"
        and labels.device.type == "cuda"
    )


def cross_entropy_to_unsloth(
    logits: TensorProxy,
    labels: TensorProxy,
    weight: TensorProxy | None = None,
    size_average: bool | None = None,
    ignore_index: int = -100,
    reduce: bool | None = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
):
    assert unsloth_cross_entropy_checker(**locals())
    loss, _ = u_cross_entropy_loss(logits, labels)
    if reduction == "none":
        return loss
    # "mean" reduction is not part of the kernel
    n_items = sum(ne(labels, -100))
    return true_divide(sum(loss), n_items)


# registers as cross entropy implementation, including the execution transform and now a grad transform
unsloth_ex.register_implementation(
    ltorch.cross_entropy,
    checker=unsloth_cross_entropy_checker,
    execution_transform=cross_entropy_to_unsloth,
    grad_transform=unsloth_cross_entropy_grad,
)
