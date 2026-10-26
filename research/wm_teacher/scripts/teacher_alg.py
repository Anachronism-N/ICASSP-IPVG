"""Isolated ALG prototype: U_original + cfg * (C_lowpass - U_lowpass).

Only a baseline fused-first-latent Wan pipeline is supported. This is a port,
not a claim of reproducing the official 5B setup or validating a 14B run.
The caller must pass the SAME call_kwargs to pipe(**call_kwargs) inside the
context. Already-merged LoRA weights cannot be detected: checkpoint="NONE"
is a required caller attestation, not a checkpoint hash verification.
No generator, frozen benchmark, DiffSynth source, or GPU launch is modified.
"""

from contextlib import contextmanager
import hashlib
import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F


_FIXED_CALL = {
    "sigma_shift": 5.0, "cfg_scale": 5.0, "num_inference_steps": 50,
    "num_frames": 121, "width": 1280, "height": 704,
}
_FALSE_OPTIONS = (
    "cfg_merge", "cfg_zero_star", "cfg_zero_steps",
    "use_unified_sequence_parallel", "use_gradient_checkpointing",
    "use_gradient_checkpointing_offload", "skip_9th_layer",
)
_ABSENT_OPTIONS = (
    "tea_cache", "tea_cache_l1_thresh", "dit2", "vace", "vace2",
    "motion_controller", "animate_adapter", "vap", "audio_encoder",
    "lora", "lora_config", "lora_path", "positive_only_lora", "negative_only_lora",
    "end_image", "input_video", "input_audio", "audio_embeds", "s2v_pose_video",
    "s2v_pose_latents", "motion_video", "motion_latents", "control_video",
    "reference_image", "reference_latents", "camera_control_direction",
    "control_camera_latents_input", "vace_video", "vace_video_mask",
    "vace_reference_image", "vace_context", "animate_pose_video", "animate_face_video",
    "animate_inpaint_video", "animate_mask_video", "pose_latents", "face_pixel_values",
    "vap_video", "vap_hidden_state", "vap_clip_feature", "context_vap",
    "motion_bucket_id", "longcat_video", "longcat_latents", "sliding_window_size",
    "sliding_window_stride", "wantodance_music_path", "wantodance_reference_image",
    "wantodance_keyframes", "wantodance_keyframes_mask", "wantodance_refimage_feature",
    "music_feature", "y", "clip_feature",
)
_ACTIVE_PIPELINES = {}


def _finite_tensor(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"ALG requires floating-point Tensor {name}")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"ALG rejects nonfinite {name}")


def _same_bits(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and left.device == right.device
            and torch.equal(left.detach().contiguous().view(torch.uint8),
                            right.detach().contiguous().view(torch.uint8)))


def lowpass_first_latent(z, factor=0.4):
    """B,C,1,H,W bilinear down/up; float32 work preserves half/bfloat16 output."""
    _finite_tensor(z, "first_frame_latents")
    if z.ndim != 5 or z.shape[0] != 1 or z.shape[2] != 1:
        raise ValueError("ALG requires first_frame_latents with shape (1,C,1,H,W)")
    if not isinstance(factor, (int, float)) or not math.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError("ALG lowpass factor must be finite in (0,1]")
    if factor == 1:
        return z
    h, w = z.shape[-2:]
    size = (max(1, round(h * factor)), max(1, round(w * factor)))
    image = z[:, :, 0]
    work = image.float() if image.dtype in (torch.float16, torch.bfloat16) else image
    down = F.interpolate(work, size=size, mode="bilinear", align_corners=False, antialias=True)
    up = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
    result = up.to(dtype=z.dtype).unsqueeze(2)
    _finite_tensor(result, "lowpass first_frame_latents")
    return result


def _reject_combinations(values):
    for name in _FALSE_OPTIONS:
        if values.get(name, False):
            raise ValueError(f"ALG does not support {name}")
    for name in _ABSENT_OPTIONS:
        if values.get(name) is not None:
            raise ValueError(f"ALG does not support {name}")
    if values.get("denoising_strength", 1.0) != 1.0:
        raise ValueError("ALG requires denoising_strength=1")


def _validate_boundary(pipe, call_kwargs, checkpoint):
    if checkpoint != "NONE":
        raise ValueError("ALG requires a baseline checkpoint=NONE; no LoRA or finetuned teacher")
    if not isinstance(call_kwargs, Mapping):
        raise ValueError("ALG requires explicit pipeline call_kwargs")
    for name, expected in _FIXED_CALL.items():
        if call_kwargs.get(name) != expected:
            raise ValueError(f"ALG requires {name}={expected}")
    seed = call_kwargs.get("seed")
    if type(seed) is not int or seed not in (0, 1):
        raise ValueError("ALG requires preregistered integer seed in (0,1)")
    for name in ("prompt", "negative_prompt"):
        if not isinstance(call_kwargs.get(name), str):
            raise ValueError(f"ALG requires explicit string {name}")
    _reject_combinations(call_kwargs)
    _reject_combinations({name: getattr(pipe, name, None) for name in _ABSENT_OPTIONS})
    if getattr(pipe, "use_unified_sequence_parallel", False) or getattr(pipe, "sp_size", 1) != 1:
        raise ValueError("ALG does not support sequence parallelism")
    dit = getattr(pipe, "dit", None)
    if dit is None or not getattr(dit, "fuse_vae_embedding_in_latents", False):
        raise ValueError("ALG requires fused first_frame_latents; non-fused 14B is unsupported")
    if getattr(dit, "require_vae_embedding", False) or getattr(dit, "require_clip_embedding", False):
        raise ValueError("ALG does not support separate VAE/CLIP image conditioning")
    if hasattr(dit, "modules"):
        for module in dit.modules():
            for name in ("lora_A_weights", "lora_B_weights"):
                if len(getattr(module, name, ())) != 0:
                    raise ValueError("ALG does not support loaded LoRA")
    if not callable(getattr(pipe, "model_fn", None)) or not callable(getattr(pipe.scheduler, "step", None)):
        raise ValueError("ALG requires callable model_fn and scheduler.step")


class _ALGHook:
    def __init__(self, pipe, call_kwargs, factor):
        self.pipe = pipe
        self.call_kwargs = dict(call_kwargs)
        self.factor = factor
        self.original_model_fn = pipe.model_fn
        self.original_step = pipe.scheduler.step
        self.step_index = 0
        self.phase = "positive"
        self.z = self.raw_z = self.effective_z = None
        self.z_lowpass = None
        self.sample = None
        self.c_lowpass = None
        self.u_original = None
        self.u_lowpass = None
        self.stats = {"completed_steps": 0, "active_steps": 0, "model_forwards": 0,
                      "first_frame_sha256": None, "raw_first_frame_sha256": None,
                      "effective_first_frame_sha256": None,
                      "raw_first_frame_dtype": None, "effective_first_frame_dtype": None,
                      "bypassed": False}

    def _timestep(self, timestep, model_call=False):
        timesteps = self.pipe.scheduler.timesteps
        if len(timesteps) != 50 or self.step_index >= 50:
            raise ValueError("ALG requires exactly 50 scheduler steps and one run per context")
        _finite_tensor(timestep, "timestep")
        expected = timesteps[self.step_index].to(device=timestep.device, dtype=timestep.dtype)
        if timestep.numel() != 1 or not torch.equal(timestep.reshape(()), expected.reshape(())):
            raise ValueError("ALG model/scheduler timestep order mismatch")
        if model_call and timestep.shape != (1,):
            raise ValueError("ALG requires batch-one model timestep")

    def _inputs(self, kwargs):
        _reject_combinations(kwargs)
        for name, value in _FIXED_CALL.items():
            if kwargs.get(name) != value:
                raise ValueError(f"ALG runtime {name} differs from registered call")
        if type(kwargs.get("seed")) is not int or kwargs["seed"] != self.call_kwargs["seed"]:
            raise ValueError("ALG runtime seed differs from registered call")
        if kwargs.get("dit") is not self.pipe.dit:
            raise ValueError("ALG does not support switched DiT")
        if kwargs.get("fuse_vae_embedding_in_latents") is not True:
            raise ValueError("ALG requires runtime fused first_frame_latents")
        x, z = kwargs.get("latents"), kwargs.get("first_frame_latents")
        _finite_tensor(x, "latents")
        _finite_tensor(z, "first_frame_latents")
        if x.ndim != 5 or x.shape[0] != 1 or x.shape[2] < 2 or z.shape != x[:, :, 0:1].shape:
            raise ValueError("ALG requires batch-one fused first frame and future frames")
        # ImageEmbedderFused casts the assignment into x, but keeps raw z in kwargs.
        effective_z = z.to(dtype=x.dtype, device=x.device)
        if not _same_bits(x[:, :, 0:1], effective_z):
            raise ValueError("ALG original first frame must equal the locked effective original z")
        context = kwargs.get("context")
        _finite_tensor(context, "context")
        if context.ndim != 3 or context.shape[0] != 1:
            raise ValueError("ALG requires unmerged batch-one text context")
        self._timestep(kwargs.get("timestep"), model_call=True)
        if self.z is None:
            self.raw_z = z
            self.z = z.detach().clone()
            self.effective_z = effective_z.detach().clone()
            self.z_lowpass = lowpass_first_latent(effective_z, self.factor)
            raw = self.z.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            effective = self.effective_z.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            self.stats.update(first_frame_sha256=hashlib.sha256(raw).hexdigest(),
                              raw_first_frame_sha256=hashlib.sha256(raw).hexdigest(),
                              effective_first_frame_sha256=hashlib.sha256(effective).hexdigest(),
                              raw_first_frame_dtype=str(z.dtype),
                              effective_first_frame_dtype=str(x.dtype))
        elif not _same_bits(z, self.z) or not _same_bits(self.raw_z, self.z):
            raise ValueError("ALG original first_frame_latents changed during the run")
        elif not _same_bits(effective_z, self.effective_z):
            raise ValueError("ALG effective original first_frame_latents changed during the run")
        return x

    def _forward(self, kwargs, lowpass=False):
        if lowpass:
            kwargs = dict(kwargs)
            x = kwargs["latents"].clone()
            x[:, :, 0:1] = self.z_lowpass
            kwargs["latents"] = x
        output = self.original_model_fn(**kwargs)
        self.stats["model_forwards"] += 1
        _finite_tensor(output, "model prediction")
        x = kwargs["latents"]
        if not _same_bits(kwargs["first_frame_latents"], self.z) or not _same_bits(self.raw_z, self.z):
            raise ValueError("ALG original first_frame_latents changed during model forward")
        expected_z = self.z_lowpass if lowpass else self.effective_z
        if not _same_bits(x[:, :, 0:1], expected_z):
            raise ValueError("ALG model forward changed the fused first frame")
        if output.shape != x.shape or output.dtype != x.dtype or output.device != x.device:
            raise ValueError("ALG model prediction must match latents shape/dtype/device")
        return output

    def model_fn(self, *args, **kwargs):
        if args:
            raise ValueError("ALG requires keyword-only model_fn inputs")
        if self.phase not in ("positive", "negative"):
            raise ValueError("ALG rejects extra model_fn calls before scheduler.step")
        positive, negative = "prompt" in kwargs, "negative_prompt" in kwargs
        expected = self.phase
        if positive == negative or positive != (expected == "positive"):
            raise ValueError("ALG expected explicit positive prompt then negative_prompt")
        prompt_key = "prompt" if positive else "negative_prompt"
        if kwargs[prompt_key] != self.call_kwargs[prompt_key]:
            raise ValueError("ALG prompt differs from registered call")
        x = self._inputs(kwargs)
        active = self.step_index < 10
        if positive:
            self.sample = x
            output = self._forward(kwargs, lowpass=active)
            if active:
                self.c_lowpass = output
            self.phase = "negative"
        else:
            if x is not self.sample:
                raise ValueError("ALG positive/negative must use the same original latents")
            output = self._forward(kwargs)
            if active:
                self.u_original = output
                self.u_lowpass = self._forward(kwargs, lowpass=True)
            self.phase = "scheduler"
        return output

    def step(self, model_output, timestep, sample, *args, **kwargs):
        if self.phase != "scheduler":
            raise ValueError("ALG scheduler.step requires exactly positive/negative calls")
        self._timestep(timestep)
        if (sample is not self.sample or not _same_bits(sample[:, :, 0:1], self.effective_z)
                or not _same_bits(self.raw_z, self.z)):
            raise ValueError("ALG scheduler must step the original sample with original z")
        _finite_tensor(sample, "scheduler sample")
        _finite_tensor(model_output, "incoming CFG prediction")
        if self.step_index < 10:
            model_output = self.u_original + self.call_kwargs["cfg_scale"] * (self.c_lowpass - self.u_lowpass)
            _finite_tensor(model_output, "ALG guidance")
        result = self.original_step(model_output, timestep, sample, *args, **kwargs)
        if not _same_bits(self.raw_z, self.z):
            raise ValueError("ALG original first_frame_latents changed during scheduler.step")
        _finite_tensor(result, "scheduler output")
        if result.shape != sample.shape or result.dtype != sample.dtype or result.device != sample.device:
            raise ValueError("ALG scheduler output must match original sample")
        self.stats["active_steps"] += int(self.step_index < 10)
        self.step_index += 1
        self.stats["completed_steps"] = self.step_index
        self.phase = "positive"
        self.sample = self.c_lowpass = self.u_original = self.u_lowpass = None
        return result


@contextmanager
def teacher_alg(pipe, *, call_kwargs, checkpoint, enabled=True, factor=0.4):
    """Wrap one fixed baseline run; restore both instance attributes on every exit.

    Active steps are 0..9 of 50. Disabled/factor=1 installs no wrappers at all.
    The pipeline's original loop, not this hook, locks z after scheduler.step.
    """
    if not isinstance(enabled, bool):
        raise ValueError("ALG enabled must be bool")
    if factor not in (0.4, 1.0):
        raise ValueError("ALG prototype supports only preregistered factor=0.4 or identity=1")
    _validate_boundary(pipe, call_kwargs, checkpoint)
    key = id(pipe)
    if key in _ACTIVE_PIPELINES:
        raise ValueError("ALG rejects nested/concurrent contexts on the same pipeline")
    hook = _ALGHook(pipe, call_kwargs, factor)
    _ACTIVE_PIPELINES[key] = pipe
    wrapped = enabled and factor != 1
    model_owned = "model_fn" in vars(pipe)
    step_owned = "step" in vars(pipe.scheduler)
    try:
        if not wrapped:
            hook.stats["bypassed"] = True
            yield hook.stats
            return
        pipe.model_fn = hook.model_fn
        pipe.scheduler.step = hook.step
        yield hook.stats
        if hook.step_index != 50 or hook.phase != "positive":
            raise ValueError("ALG context exited without one complete 50-step run")
        if not _same_bits(hook.raw_z, hook.z):
            raise ValueError("ALG original first_frame_latents changed after the last scheduler.step")
    finally:
        if wrapped:
            if model_owned:
                pipe.model_fn = hook.original_model_fn
            else:
                vars(pipe).pop("model_fn", None)
            if step_owned:
                pipe.scheduler.step = hook.original_step
            else:
                vars(pipe.scheduler).pop("step", None)
        _ACTIVE_PIPELINES.pop(key, None)
