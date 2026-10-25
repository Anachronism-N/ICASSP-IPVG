"""Isolated, CPU-tested DyMoS candidate; CUDA feasibility is NOT validated.

Scaled QK logits receive -0.6 only for future query -> first-latent key,
conditional branch, steps 0..9/50, all 30 self-attention layers (phi=1).
Only compiled CUDA flex_attention is allowed: no eager/dense fallback. Compile,
A100/torch-2.7.1 correctness, peak memory and speed need a separately budgeted
one-video <=1 allocated GPUh smoke before the 188-video seed0/1 candidate.
This is a local paper-inspired port, not an official implementation/reproduction;
no usable official repository was established (the checked candidate returned 404).

Use teacher_dymos(pipe, call_kwargs=kwargs, checkpoint="NONE") around exactly
one pipe(**kwargs) call. NONE attests baseline weights; merged LoRA/finetuned
weights cannot be authenticated by this hook. Caller must freeze weight/input/
code hashes separately. No generator, scheduler, frozen plan or DiffSynth source
is changed. Original RMSNorm/RoPE/projections remain upstream of the hook.
"""

from collections.abc import Mapping
from contextlib import contextmanager
import math

import torch


_FIXED_CALL = {
    "sigma_shift": 5.0, "cfg_scale": 5.0, "num_inference_steps": 50,
    "num_frames": 121, "width": 1280, "height": 704,
}
_FALSE_OPTIONS = (
    "cfg_merge", "cfg_zero_star", "cfg_zero_steps", "use_unified_sequence_parallel",
    "use_gradient_checkpointing", "use_gradient_checkpointing_offload", "skip_9th_layer",
)
_ABSENT_OPTIONS = (
    "tea_cache", "tea_cache_l1_thresh", "dit2", "vace", "vace2", "motion_controller",
    "animate_adapter", "vap", "audio_encoder", "lora", "lora_path", "lora_config",
    "positive_only_lora", "negative_only_lora", "end_image", "input_video",
    "input_audio", "audio_embeds", "s2v_pose_video", "s2v_pose_latents", "motion_video",
    "motion_latents", "control_video", "reference_image", "reference_latents",
    "camera_control_direction", "control_camera_latents_input", "vace_video",
    "vace_video_mask", "vace_reference_image", "vace_context", "animate_pose_video",
    "animate_face_video", "animate_inpaint_video", "animate_mask_video", "pose_latents",
    "face_pixel_values", "vap_video", "vap_hidden_state", "vap_clip_feature", "context_vap",
    "motion_bucket_id", "longcat_video", "longcat_latents", "sliding_window_size",
    "sliding_window_stride", "wantodance_music_path", "wantodance_reference_image",
    "wantodance_keyframes", "wantodance_keyframes_mask", "wantodance_refimage_feature",
    "music_feature", "y", "clip_feature",
)
_ACTIVE_PIPELINES = {}


def _reject_combinations(values):
    for name in _FALSE_OPTIONS:
        if values.get(name, False):
            raise ValueError(f"DyMoS does not support {name}")
    for name in _ABSENT_OPTIONS:
        if values.get(name) is not None:
            raise ValueError(f"DyMoS does not support {name}")
    if values.get("denoising_strength", 1.0) != 1.0:
        raise ValueError("DyMoS requires denoising_strength=1")


def _validate_boundary(pipe, call_kwargs, checkpoint):
    if checkpoint != "NONE":
        raise ValueError("DyMoS requires baseline checkpoint=NONE (caller attestation)")
    if not isinstance(call_kwargs, Mapping):
        raise ValueError("DyMoS requires explicit pipeline call_kwargs")
    for name, expected in _FIXED_CALL.items():
        if call_kwargs.get(name) != expected:
            raise ValueError(f"DyMoS requires {name}={expected}")
    if type(call_kwargs.get("seed")) is not int or call_kwargs["seed"] not in (0, 1):
        raise ValueError("DyMoS requires preregistered integer seed0/1")
    for name in ("prompt", "negative_prompt"):
        if not isinstance(call_kwargs.get(name), str):
            raise ValueError(f"DyMoS requires explicit string {name}")
    _reject_combinations(call_kwargs)
    _reject_combinations({name: getattr(pipe, name, None) for name in _ABSENT_OPTIONS})
    if getattr(pipe, "use_unified_sequence_parallel", False) or getattr(pipe, "sp_size", 1) != 1:
        raise ValueError("DyMoS does not support sequence parallelism")
    dit = getattr(pipe, "dit", None)
    if (dit is None or not getattr(dit, "fuse_vae_embedding_in_latents", False)
            or not getattr(dit, "seperated_timestep", False)
            or getattr(dit, "require_vae_embedding", True)
            or getattr(dit, "require_clip_embedding", True)
            or getattr(dit, "dim", None) != 3072 or getattr(dit, "in_dim", None) != 48):
        raise ValueError("DyMoS requires baseline Wan2.2 5B fused-first-latent DiT")
    if tuple(getattr(dit, "patch_size", ())) != (1, 2, 2):
        raise ValueError("DyMoS requires the original (1,2,2) patch geometry")
    patch = getattr(dit, "patch_embedding", None)
    if (not isinstance(patch, torch.nn.Conv3d) or patch.kernel_size != (1, 2, 2)
            or patch.stride != (1, 2, 2) or patch.padding != (0, 0, 0)
            or patch.dilation != (1, 1, 1)):
        raise ValueError("DyMoS rejects non-original patch embedding")
    if getattr(getattr(pipe, "vae", None), "upsampling_factor", None) != 16:
        raise ValueError("DyMoS requires 5B VAE spatial factor16")
    blocks = getattr(dit, "blocks", ())
    if len(blocks) != 30:
        raise ValueError("DyMoS requires all 30 self-attention layers (phi=1)")
    attention = []
    for block in blocks:
        self_attn = getattr(block, "self_attn", None)
        attn = getattr(self_attn, "attn", None)
        if (attn is None or not callable(getattr(attn, "forward", None))
                or getattr(attn, "num_heads", None) != 24
                or getattr(self_attn, "dim", None) != 3072
                or getattr(self_attn, "head_dim", None) != 128):
            raise ValueError("DyMoS requires original 24-head,128-dim AttentionModule")
        if attn is getattr(getattr(block, "cross_attn", None), "attn", None):
            raise ValueError("DyMoS rejects shared self/cross attention modules")
        attention.append(attn)
    if len({id(attn) for attn in attention}) != 30:
        raise ValueError("DyMoS rejects shared self-attention layers")
    for module in dit.modules():
        if any(len(getattr(module, name, ())) for name in ("lora_A_weights", "lora_B_weights")):
            raise ValueError("DyMoS does not support loaded LoRA")
        if any(value for name, value in vars(module).items() if name.startswith("wantodance_enable_")):
            raise ValueError("DyMoS does not support WanToDance adapters")
    model_fn = getattr(pipe, "model_fn", None)
    if not callable(model_fn) or not hasattr(getattr(pipe, "scheduler", None), "timesteps"):
        raise ValueError("DyMoS requires model_fn and scheduler.timesteps")
    if getattr(model_fn, "__self__", pipe) is not pipe:
        raise ValueError("DyMoS rejects another teacher/model_fn hook")
    return attention


def _same_bits(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype and left.device == right.device
            and torch.equal(left.detach().contiguous().view(torch.uint8),
                            right.detach().contiguous().view(torch.uint8)))


def _reference_span(dit, vae, latents, call_kwargs):
    """From real latent shape and Conv3d geometry, not input-pixel token count."""
    if not isinstance(latents, torch.Tensor) or not latents.is_floating_point() or latents.ndim != 5:
        raise ValueError("DyMoS requires floating B,C,F,H,W latents")
    factor = vae.upsampling_factor
    frames = (call_kwargs["num_frames"] - 1) // 4 + 1
    expected = (1, dit.in_dim, frames, call_kwargs["height"] // factor, call_kwargs["width"] // factor)
    if (latents.shape != expected or call_kwargs["height"] % factor
            or call_kwargs["width"] % factor):
        raise ValueError("DyMoS latents do not match registered frame/spatial geometry")
    patch = dit.patch_embedding
    kernel, stride = patch.kernel_size, patch.stride
    if (kernel != tuple(dit.patch_size) or stride != kernel or kernel[0] != 1
            or patch.padding != (0, 0, 0) or patch.dilation != (1, 1, 1)
            or any(size % step for size, step in zip(latents.shape[2:], stride))):
        raise ValueError("DyMoS rejects ambiguous first-latent patch span")
    f, h, w = (size // step for size, step in zip(latents.shape[2:], stride))
    reference, sequence = h * w, f * h * w
    if f < 2:
        raise ValueError("DyMoS requires reference and future latent frames")
    return reference, sequence


def _score_mod(reference_tokens, gamma):
    # torch.where operates on per-score scalar/index tensors inside the fused kernel.
    def modify(score, batch, head, query, key):
        return torch.where((query >= reference_tokens) & (key < reference_tokens), score - gamma, score)
    return modify


class _FusedAttention:
    def __init__(self):
        self.compiled = None

    def __call__(self, q, k, v, num_heads, reference_tokens, gamma):
        if q.device.type != "cuda":
            raise RuntimeError("DyMoS active attention requires compiled CUDA flex_attention; GPU smoke pending")
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("DyMoS CUDA flex_attention requires half/bfloat16/float32")
        if self.compiled is None:
            from torch.nn.attention.flex_attention import flex_attention
            def compiled_only(qh, kh, vh, *, score_mod, scale):
                if not torch.compiler.is_compiling():
                    raise RuntimeError("DyMoS refuses eager flex_attention; compilation is disabled")
                return flex_attention(qh, kh, vh, score_mod=score_mod, scale=scale)
            self.compiled = torch.compile(compiled_only, backend="inductor", fullgraph=True, dynamic=False)
        b, s, width = q.shape
        head_dim = width // num_heads
        qh, kh, vh = (value.reshape(b, s, num_heads, head_dim).transpose(1, 2)
                      for value in (q, k, v))
        # No eager call, block mask or S² bias tensor. Failure propagates, never falls back.
        result = self.compiled(qh, kh, vh, score_mod=_score_mod(reference_tokens, gamma),
                               scale=1.0 / math.sqrt(head_dim))
        if result.shape != (b, num_heads, s, head_dim) or result.dtype != q.dtype or result.device != q.device:
            raise ValueError("DyMoS fused output shape/dtype/device mismatch")
        return result.transpose(1, 2).reshape(b, s, width)


class _DyMoSHook:
    def __init__(self, pipe, call_kwargs, gamma, attention):
        self.pipe, self.call_kwargs, self.gamma = pipe, dict(call_kwargs), gamma
        self.original_model_fn = pipe.model_fn
        self.originals = [(attn, "forward" in vars(attn), attn.forward) for attn in attention]
        self.fused = _FusedAttention()
        self.step_index = 0
        self.phase = "positive"
        self.active = False
        self.reference_tokens = self.sequence_tokens = None
        self.z = self.sample = None
        self.seen_layers = set()
        self.stats = {"completed_steps": 0, "active_steps": 0, "model_forwards": 0,
                      "biased_attention_calls": 0, "original_attention_calls": 0,
                      "reference_tokens": None, "sequence_tokens": None, "bypassed": False}

    def attention_forward(self, index, attn, original):
        def forward(q, k, v):
            if not self.active:
                self.stats["original_attention_calls"] += 1
                return original(q, k, v)
            if index in self.seen_layers:
                raise ValueError("DyMoS rejects repeated self-attention layer in one forward")
            expected = (1, self.sequence_tokens, self.pipe.dit.dim)
            if any(not isinstance(x, torch.Tensor) or x.shape != expected for x in (q, k, v)):
                raise ValueError("DyMoS QKV must have original batch-one B,S,H*D layout")
            if any(x.dtype != q.dtype or x.device != q.device for x in (k, v)):
                raise ValueError("DyMoS QKV dtype/device mismatch")
            result = self.fused(q, k, v, attn.num_heads, self.reference_tokens, self.gamma)
            self.seen_layers.add(index)
            self.stats["biased_attention_calls"] += 1
            return result
        return forward

    def model_fn(self, *args, **kwargs):
        if args:
            raise ValueError("DyMoS requires keyword-only model_fn inputs")
        positive, negative = "prompt" in kwargs, "negative_prompt" in kwargs
        if positive == negative or positive != (self.phase == "positive"):
            raise ValueError("DyMoS requires explicit positive prompt then negative_prompt")
        key = "prompt" if positive else "negative_prompt"
        if kwargs[key] != self.call_kwargs[key]:
            raise ValueError("DyMoS runtime prompt differs from registered call")
        _reject_combinations(kwargs)
        for name, expected in _FIXED_CALL.items():
            if kwargs.get(name) != expected:
                raise ValueError(f"DyMoS runtime {name} differs from registered call")
        if type(kwargs.get("seed")) is not int or kwargs["seed"] != self.call_kwargs["seed"]:
            raise ValueError("DyMoS runtime seed differs from registered call")
        if kwargs.get("dit") is not self.pipe.dit or kwargs.get("fuse_vae_embedding_in_latents") is not True:
            raise ValueError("DyMoS rejects switched DiT/non-fused conditioning")
        timesteps, timestep = self.pipe.scheduler.timesteps, kwargs.get("timestep")
        if len(timesteps) != 50 or self.step_index >= 50:
            raise ValueError("DyMoS requires exactly one 50-step run per context")
        if (not isinstance(timestep, torch.Tensor) or timestep.shape != (1,)
                or not torch.equal(timestep.reshape(()), timesteps[self.step_index].to(timestep).reshape(()))):
            raise ValueError("DyMoS model timestep order mismatch")
        x, z, context = kwargs.get("latents"), kwargs.get("first_frame_latents"), kwargs.get("context")
        reference, sequence = _reference_span(self.pipe.dit, self.pipe.vae, x, self.call_kwargs)
        if not isinstance(z, torch.Tensor) or not _same_bits(x[:, :, 0:1], z):
            raise ValueError("DyMoS first_frame_latents must equal the original locked first frame")
        if self.z is None:
            self.z = z.detach().clone()
            self.reference_tokens, self.sequence_tokens = reference, sequence
            self.stats.update(reference_tokens=reference, sequence_tokens=sequence)
        elif not _same_bits(z, self.z):
            raise ValueError("DyMoS locked reference changed during the run")
        if (not isinstance(context, torch.Tensor) or not context.is_floating_point()
                or context.ndim != 3 or context.shape[0] != 1):
            raise ValueError("DyMoS requires unmerged batch-one text context")
        if positive:
            self.sample = x
        elif x is not self.sample:
            raise ValueError("DyMoS positive/negative must use the same original latents")
        self.active = positive and self.step_index < 10
        self.seen_layers.clear()
        try:
            output = self.original_model_fn(**kwargs)
            if self.active and len(self.seen_layers) != 30:
                raise ValueError("DyMoS active forward must visit all 30 self-attention layers")
        finally:
            self.active = False
        self.stats["model_forwards"] += 1
        if positive:
            self.phase = "negative"
        else:
            self.stats["active_steps"] += int(self.step_index < 10)
            self.step_index += 1
            self.stats["completed_steps"] = self.step_index
            self.phase = "positive"
            self.sample = None
        return output


@contextmanager
def teacher_dymos(pipe, *, call_kwargs, checkpoint, enabled=True, gamma=0.6, phi=1):
    """One fixed baseline run. Off/gamma0 installs no wrappers and never compiles."""
    if type(enabled) is not bool or type(gamma) not in (int, float) or gamma not in (0.0, 0.6):
        raise ValueError("DyMoS supports only enabled bool and gamma=0.6 or identity gamma=0")
    if type(phi) not in (int, float) or phi != 1:
        raise ValueError("DyMoS preregisters phi=1 only")
    attention = _validate_boundary(pipe, call_kwargs, checkpoint)
    identity = id(pipe)
    if identity in _ACTIVE_PIPELINES:
        raise ValueError("DyMoS rejects nested/concurrent contexts on the same pipeline")
    hook = _DyMoSHook(pipe, call_kwargs, gamma, attention)
    model_owned = "model_fn" in vars(pipe)
    wrapped = enabled and gamma != 0
    _ACTIVE_PIPELINES[identity] = pipe
    try:
        if not wrapped:
            hook.stats["bypassed"] = True
            yield hook.stats
            return
        pipe.model_fn = hook.model_fn
        for index, (attn, owned, original) in enumerate(hook.originals):
            attn.forward = hook.attention_forward(index, attn, original)
        yield hook.stats
        if hook.step_index != 50 or hook.phase != "positive":
            raise ValueError("DyMoS context exited without exactly 50 positive/negative pairs")
    finally:
        hook.active = False
        if wrapped:
            if model_owned:
                pipe.model_fn = hook.original_model_fn
            else:
                vars(pipe).pop("model_fn", None)
            for attn, owned, original in hook.originals:
                if owned:
                    attn.forward = original
                else:
                    vars(attn).pop("forward", None)
        _ACTIVE_PIPELINES.pop(identity, None)
