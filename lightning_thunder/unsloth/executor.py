# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import thunder
import thunder.torch as ltorch
from thunder.core.proxies import TensorProxy
from thunder.core.transforms import put_grads, get_grad, mean_backward
from thunder.extend import OperatorExecutor, register_executor
from thunder.torch import ne, sum, true_divide

import lightning_thunder.unsloth.kernels as kernels


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


unsloth_cross_entropy = unsloth_ex.register_operator(
    "unsloth_cross_entropy", like=unsloth_cross_entropy_like, fn=kernels.cross_entropy_loss._cross_entropy_forward_impl
)


def unsloth_cross_entropy_backward_impl(dlosses, logits, logsumexp, labels):
    # clone() because the kernel writes the grads in the logits.
    # If it works, we can remove this it, but it's not a thing we generally anticipate and support right now.
    return kernels.cross_entropy_loss._cross_entropy_backward_impl(dlosses, logits.clone(), logsumexp, labels)


def unsloth_cross_entropy_backward_like(dlosses, logits, logsumexp, labels):
    return thunder.TensorProxy(like=logits)


unsloth_cross_entropy_backward = unsloth_ex.register_operator(
    "unsloth_cross_entropy_backward", like=unsloth_cross_entropy_backward_like, fn=unsloth_cross_entropy_backward_impl
)


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
    loss, _ = unsloth_cross_entropy(logits, labels)
    if reduction == "none":
        return loss
    # "mean" reduction is not part of the kernel
    # TODO: this doesn't consider that all elements could be masked, causing a division by 0
    n_items = sum(ne(labels, -100))
    return true_divide(sum(loss), n_items)


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
    loss, logsumexp = unsloth_cross_entropy(logits, labels)
    if reduction == "mean":
        # "mean" reduction is not part of the kernel
        n_items = sum(ne(labels, -100))
        loss = true_divide(sum(loss), n_items)

    g = get_grad(loss)

    if reduction == "mean":
        g = mean_backward(logsumexp.ndim, logsumexp.shape, (0,), g)

    logits_grad = unsloth_cross_entropy_backward(g, logits, logsumexp, labels)
    put_grads((logits,), (logits_grad,))

    return loss


# registers as cross entropy implementation, including the execution transform and now a grad transform
unsloth_ex.register_implementation(
    ltorch.cross_entropy,
    checker=unsloth_cross_entropy_checker,
    execution_transform=cross_entropy_to_unsloth,
    grad_transform=unsloth_cross_entropy_grad,
)
