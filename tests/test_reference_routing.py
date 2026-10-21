import importlib.util
from pathlib import Path
import types
import pytest
torch = pytest.importorskip('torch')
spec = importlib.util.spec_from_file_location('reference_routing', Path(__file__).resolve().parents[1] / 'scripts/adapters/vace_reference_routing.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_null_context_changes_reference_channels_only():
    x = torch.arange(2 * 96 * 3 * 2 * 2).reshape(2, 96, 3, 2, 2).float()
    original = x.clone()
    null = module.null_reference(x)
    assert torch.equal(x, original)
    assert not null[:, :16, :1].count_nonzero()
    assert torch.equal(null[:, 16:], x[:, 16:])
    assert torch.equal(null[:, :, 1:], x[:, :, 1:])


def test_temporal_difference_mean_prefix_and_gradient():
    ref = torch.tensor([[[10.], [2.], [4.], [6.]], [[20.], [8.], [10.], [12.]]], requires_grad=True)
    null = torch.tensor([[[99.], [1.], [1.], [1.]], [[99.], [2.], [4.], [6.]]])
    output, metrics = module.route_hints((ref,), (null,), (4, 1, 1), 0)
    assert torch.equal(output[0][:, 0], ref[:, 0])
    assert torch.equal(output[0][0, 1:], torch.tensor([[4.], [4.], [4.]]))
    assert torch.equal(output[0][1, 1:], ref[1, 1:])
    assert torch.equal(output[0][:, 1:].mean(dim=1), ref[:, 1:].mean(dim=1))
    assert metrics['reference_prefix_exact'] and metrics['body_temporal_mean_max_abs_error'] == 0
    assert metrics['body_max_abs_change'] == 2
    output[0].sum().backward()
    assert torch.isfinite(ref.grad).all() and ref.grad.abs().sum() > 0


def test_control_is_original_tensor_not_numerical_roundtrip():
    a = torch.randn(2, 8, 3, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    output, metrics = module.route_hints((a,), (b,), (4, 1, 2), 1)
    assert output[0] is a
    assert metrics['body_max_abs_change'] == 0 and metrics['body_temporal_mean_max_abs_error'] == 0


@pytest.mark.parametrize('bad', [True, -.1, .5, 2, '0', None])
def test_gate_or_layout_rejected(bad):
    a = torch.ones(1, 4, 2)
    with pytest.raises(ValueError): module.route_hints((a,), (a,), (4, 1, 1), bad)


def test_invalid_shapes_and_nonfinite_rejected():
    a = torch.ones(1, 4, 2)
    for grid in ((1, 1, 4), (4, 2, 2)):
        with pytest.raises(ValueError): module.route_hints((a,), (a,), grid, 0)
    with pytest.raises(ValueError): module.null_reference(torch.ones(1, 95, 14, 44, 60))
    a[0, 0, 0] = float('nan')
    with pytest.raises(RuntimeError): module.route_hints((a,), (torch.zeros_like(a),), (4, 1, 1), 0)


class FakeVace(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vace_patch_embedding = types.SimpleNamespace(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.vace_layers = list(range(15))
        self.calls = 0
        self.reference_count = []
    def forward(self, x, context, text, timestep, freqs, **kw):
        self.calls += 1
        self.reference_count.append(int(context[:, :16, :1].count_nonzero().item()))
        return tuple(x for _ in self.vace_layers)


def test_native_calls_and_restore_on_exception():
    model = FakeVace(); original = model.forward
    x = torch.zeros(1, 9240, 1536)
    context = torch.ones(1, 96, 14, 44, 60)
    with pytest.raises(RuntimeError, match='fixture failure'):
        with module.reference_routing(model, 1) as metrics:
            output = model(x, context, None, None, None)
            assert all(r is x for r in output)
            assert model.calls == metrics['native_calls'] == 2 and metrics['outer_calls'] == 1
            assert model.reference_count[0] > 0 and model.reference_count[1] == 0
            raise RuntimeError('fixture failure')
    assert model.forward == original and 'forward' not in model.__dict__


def test_invalid_native_layout_rejected_and_restored():
    model = FakeVace(); original = model.forward
    with pytest.raises(ValueError):
        with module.reference_routing(model, 0): model(torch.zeros(1, 1, 1536), torch.ones(1, 96, 14, 44, 60), None, None, None)
    assert model.forward == original and model.calls == 0
