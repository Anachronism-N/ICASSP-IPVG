"""Experimental reference-difference temporal routing; not a pure-identity encoder."""
from contextlib import contextmanager
import math


def null_reference(context, reference_frames=1):
    if context.ndim != 5 or context.shape[1] != 96 or reference_frames != 1 or context.shape[2] <= 1:
        raise ValueError('Routing requires96-channel context and one reference prefix')
    copy = context.clone()
    copy[:, :16, :reference_frames] = 0
    return copy


def route_hints(reference, null, grid, gate, reference_frames=1):
    import torch
    if isinstance(gate, bool) or gate not in (0, 1):
        raise ValueError('Fixed reference temporal gate must be0or1')
    if reference_frames != 1 or len(grid) != 3 or grid[0] <= reference_frames or any(n <= 0 for n in grid):
        raise ValueError('Invalid routing grid or prefix')
    if len(reference) != len(null) or not reference:
        raise ValueError('Reference/null hint counts differ')
    time, height, width = grid
    spatial = height * width
    routed = []
    zero = reference[0].new_zeros((), dtype=torch.float32)
    maximum_change, maximum_mean_error, maximum_centered = zero, zero, zero
    prefix_exact, finite = True, True
    for left, right in zip(reference, null):
        if (left.ndim != 3 or left.shape != right.shape or left.shape[1] != time * spatial
                or left.device != right.device or left.dtype != right.dtype):
            raise ValueError('Hints differ or tokens do not match time-major layout')
        a = left.reshape(left.shape[0], time, spatial, left.shape[2])
        b = right.reshape_as(a)
        body = a[:, reference_frames:]
        difference = body.float() - b[:, reference_frames:].float()
        centered = difference - difference.mean(dim=1, keepdim=True)
        if gate == 1:
            output = left
            corrected = body
        else:
            corrected = (body.float() - centered).to(left.dtype)
            output = torch.cat((a[:, :reference_frames], corrected), dim=1).reshape_as(left)
        delta = corrected.float() - body.float()
        maximum_change = torch.maximum(maximum_change, delta.abs().max())
        maximum_centered = torch.maximum(maximum_centered, centered.abs().max())
        maximum_mean_error = torch.maximum(maximum_mean_error, delta.mean(dim=1).abs().max())
        prefix_exact = prefix_exact and torch.equal(output.reshape_as(a)[:, :reference_frames], a[:, :reference_frames])
        finite = finite and bool(torch.isfinite(output).all().item())
        routed.append(output)
    if not finite or not prefix_exact:
        raise RuntimeError('Routing produced nonfinite hints or changed reference prefix')
    return tuple(routed), {'body_max_abs_change': float(maximum_change.item()),
                           'reference_temporal_centered_max_abs': float(maximum_centered.item()),
                           'body_temporal_mean_max_abs_error': float(maximum_mean_error.item()),
                           'reference_prefix_exact': prefix_exact}


@contextmanager
def reference_routing(model, gate, reference_frames=1):
    if isinstance(gate, bool) or not isinstance(gate, (int, float)) or not math.isfinite(gate) or gate not in (0, 1):
        raise ValueError('Fixed reference temporal gate must be0or1')
    patch = model.vace_patch_embedding
    if tuple(patch.kernel_size) != (1, 2, 2) or tuple(patch.stride) != (1, 2, 2):
        raise ValueError('Reference routing requires native1x2x2 patches')
    original = model.forward
    owned = 'forward' in model.__dict__
    previous = model.__dict__.get('forward')
    metrics = {'mode': 'reference_difference_temporal_v1', 'gate': gate, 'reference_frames': reference_frames,
               'outer_calls': 0, 'native_calls': 0, 'hint_layers': len(model.vace_layers),
               'body_max_abs_change': 0.0, 'reference_temporal_centered_max_abs': 0.0,
               'body_temporal_mean_max_abs_error': 0.0, 'reference_prefix_exact': True,
               'null_scope': 'only first16channels of one reference-prefix latent frame zeroed'}

    def forward(x, vace_context, context, t_mod, freqs, **kwargs):
        shape = vace_context.shape
        if (vace_context.ndim != 5 or shape[1] != 96 or shape[2] != 14 or shape[3:] != (44, 60)
                or x.ndim != 3 or x.shape[0] != shape[0] or x.shape[1:] != (9240, 1536)
                or len(model.vace_layers) != 15):
            raise ValueError('Routing only verified for full49frame480x352 VACE1.3B tokens')
        null_context = null_reference(vace_context, reference_frames)
        if not bool(vace_context[:, :16, :reference_frames].count_nonzero().item()):
            raise ValueError('Reference prefix has no nonzero conditioning')
        metrics['grid'] = [14, 22, 30]
        metrics['context_shape'] = list(shape)
        metrics['outer_calls'] += 1
        reference = original(x, vace_context, context, t_mod, freqs, **kwargs)
        metrics['native_calls'] += 1
        null = original(x, null_context, context, t_mod, freqs, **kwargs)
        metrics['native_calls'] += 1
        if len(reference) != 15 or len(null) != 15:
            raise ValueError('Expected15 VACE injection hints')
        result, observed = route_hints(reference, null, (14, 22, 30), gate, reference_frames)
        for key in ('body_max_abs_change', 'reference_temporal_centered_max_abs', 'body_temporal_mean_max_abs_error'):
            metrics[key] = max(metrics[key], observed[key])
        metrics['reference_prefix_exact'] = metrics['reference_prefix_exact'] and observed['reference_prefix_exact']
        return result

    model.forward = forward
    try:
        yield metrics
    finally:
        if owned:
            model.forward = previous
        else:
            del model.forward
