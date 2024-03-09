import torch
from conftest import RunIf


@RunIf(min_cuda_gpus=1, thunder=True)
def test_unsloth_cross_entropy():
    import thunder
    from thunder.core.transforms import grad
    from lightning_thunder.unsloth.executor import unsloth_ex, u_cross_entropy_loss

    logits = torch.randn(64, 128, device="cuda")
    labels = torch.randint(128, (64,), device="cuda")

    jit_u_cross_entropy_loss = thunder.jit(u_cross_entropy_loss, executors=[unsloth_ex])
    actual, _ = jit_u_cross_entropy_loss(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    torch.testing.assert_close(actual, expected)

    def foo(logits, labels):
        # this is the variant supported by unsloth.
        # if different arguments are used, the implementation would no be lowered to unsloth and instead would get
        # decomposed
        return torch.nn.functional.cross_entropy(logits, labels, reduction="mean", ignore_index=-100)

    cfoo = thunder.jit(foo, executors=[unsloth_ex])
    actual = cfoo(logits, labels)
    trace_str = str(thunder.last_traces(cfoo)[-1])
    assert "unsloth_cross_entropy_loss" in trace_str

    expected = foo(logits, labels)
    torch.testing.assert_close(actual, expected)

    logits.requires_grad_()
    cfoo_grad = grad(cfoo)
    actual = cfoo_grad(logits, labels)[0]
    trace_str = str(thunder.last_traces(cfoo_grad)[-1])
    assert "unsloth_cross_entropy_loss_backward" in trace_str
    out = cfoo(logits, labels)
    out.sum().backward()
    expected = logits.grad
    torch.testing.assert_close(actual, expected)


@RunIf(min_cuda_gpus=1, thunder=True)
def test_unsloth_gpt():
    from litgpt import GPT, Config
    from litgpt.utils import chunked_cross_entropy
    import thunder
    from lightning_thunder.unsloth.executor import unsloth_ex

    def forward_and_loss(model, input_ids, targets):
        logits = model(input_ids)
        return chunked_cross_entropy(logits, targets, chunk_size=0)

    cfn = thunder.jit(forward_and_loss, executors=[unsloth_ex])

    device = torch.device("cuda")
    config = Config(
        vocab_size=320,
        padding_multiple=64,
        n_layer=2,
        n_head=4,
        n_embd=64,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=1376,
    )
    with device:
        model = GPT(config)
        input_ids = torch.randint(1, 10, (2, 3))
        targets = torch.randint(0, 10, (2, 3))

    loss = cfn(model, input_ids, targets)
    assert isinstance(loss, torch.Tensor)

    fwd, bwd = thunder.last_traces(cfn)
    print(fwd[-1].python())
