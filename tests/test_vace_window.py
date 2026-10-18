import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
SPEC = importlib.util.spec_from_file_location(
    "diffsynth_vace_window", Path(__file__).resolve().parents[1] / "scripts/adapters/diffsynth_vace_window.py"
)
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


TIMELINE = [{"start_s": 0, "end_s": 2, "action": "moves left"},
            {"start_s": 2, "end_s": 4.125, "action": "moves right"}]


def masks():
    return ADAPTER.build_event_masks(33, 8, TIMELINE)


def test_reference_prefix_and_four_frame_occupancy():
    first, second = masks()
    assert isinstance(first, torch.Tensor)
    assert first.shape == second.shape == (1, 1, 10, 1, 1)
    assert first.flatten().tolist() == [0, 1, 1, 1, 1, 0.75, 0, 0, 0, 0]
    assert second.flatten().tolist() == [0, 0, 0, 0, 0, 0.25, 1, 1, 1, 1]
    assert (first + second).flatten().tolist() == [0] + [1] * 9


@pytest.mark.parametrize("reference_prefix", [0, 1, 2])
def test_one_frame_and_reference_prefix(reference_prefix):
    result = ADAPTER.build_event_masks(1, 8, [{"start_s": 0, "end_s": 0.125, "action": "still"}],
                                       reference_prefix=reference_prefix)
    assert result[0].shape == (1, 1, reference_prefix + 1, 1, 1)
    assert result[0].flatten().tolist() == [0] * reference_prefix + [1]


def test_boundary_uses_frame_timestamps_and_not_latent_centers():
    result = ADAPTER.build_event_masks(5, 8, [
        {"start_s": 0, "end_s": 0.25, "action": "first"},
        {"start_s": 0.25, "end_s": 0.625, "action": "second"},
    ])
    assert result[0].flatten().tolist() == [0, 1, 0.25]
    assert result[1].flatten().tolist() == [0, 0, 0.75]


def test_gap_retains_global_prediction_weight():
    result = ADAPTER.build_event_masks(5, 8, [
        {"start_s": 0.125, "end_s": 0.25, "action": "first"},
        {"start_s": 0.5, "end_s": 0.625, "action": "second"},
    ])
    assert result[0].flatten().tolist() == [0, 0, 0.25]
    assert result[1].flatten().tolist() == [0, 0, 0.25]


@pytest.mark.parametrize("start,end", [(-1, 1), (0, 4.126), (2, 1), (0, 0),
                                        (float("nan"), 1), (0, float("inf")), (True, 1), ("0", 1)])
def test_invalid_event_range_fails(start, end):
    with pytest.raises(ValueError, match="timeline"):
        ADAPTER.build_event_masks(33, 8, [{"start_s": start, "end_s": end, "action": "move"}])


@pytest.mark.parametrize("timeline", [[], None, [None], [{"start_s": 0, "end_s": 1}],
                                      [{"start_s": 0, "end_s": 1, "action": " "}],
                                      [{"start_s": 0, "end_s": 2, "action": "a"},
                                       {"start_s": 1.9, "end_s": 4, "action": "b"}]])
def test_invalid_or_overlapping_timeline_fails(timeline):
    with pytest.raises(ValueError, match="timeline"):
        ADAPTER.build_event_masks(33, 8, timeline)


@pytest.mark.parametrize("frames,fps,prefix", [(0, 8, 1), (32, 8, 1), (51, 8, 1), (True, 8, 1),
                                               (33, 0, 1), (33, float("nan"), 1), (33, True, 1),
                                               (33, 8, -1), (33, 8, True)])
def test_invalid_shape_parameters_fail(frames, fps, prefix):
    with pytest.raises(ValueError):
        ADAPTER.build_event_masks(frames, fps, TIMELINE, prefix)


def kwargs(positive=True):
    return {"positive": positive, "latents": torch.zeros(1, 2, 10, 2, 3),
            "timestep": torch.tensor([999.]), "context": torch.tensor([[[2.]]]),
            "vace_context": torch.ones(1, 4, 10, 2, 3), "cfg_merge": False,
            "tea_cache": None, "dit": object(), "vace": object(), "vace_scale": 1.0}


@pytest.mark.parametrize("mode", ADAPTER.MODES)
def test_equal_budget_same_inputs_and_untouched_negative(mode):
    seen = []

    def original(**call):
        seen.append(call)
        return torch.ones_like(call["latents"]) * call["context"].item()

    contexts = [torch.tensor([[[6.]]]), torch.tensor([[[10.]]])]
    wrapped = ADAPTER.WindowGuidance(original, mode, masks(), contexts)
    positive = kwargs()
    result = wrapped(**positive)
    negative = dict(positive, positive=False, context=torch.tensor([[[-7.]]]))
    negative_result = wrapped(**negative)
    assert len(seen) == 4
    assert wrapped.positive_calls == 3 and wrapped.negative_calls == 1
    assert torch.equal(negative_result, torch.full_like(negative_result, -7))
    assert seen[-1].keys() == negative.keys()
    for key, value in negative.items():
        assert seen[-1][key] is value
    for call in seen:
        for key, value in positive.items():
            if key not in ("context", "positive"):
                assert call[key] is value
    if mode == "flatten":
        assert all(call["context"] is positive["context"] for call in seen[:3])
        assert torch.equal(result, torch.full_like(result, 2))
    else:
        assert seen[1]["context"] is contexts[0]
        assert seen[2]["context"] is contexts[1]
        first, second = masks()
        if mode == "swapped":
            first, second = second, first
        expected = torch.full_like(result, 2) + first * 4 + second * 8
        assert torch.equal(result, expected)
    assert result.shape == positive["latents"].shape
    assert torch.equal(result[:, :, 0], torch.full_like(result[:, :, 0], 2))
    observations = wrapped.metrics()
    for name in ("same_latents_identity", "same_timestep_identity", "same_non_context_kwargs_identity",
                 "negative_same_step_inputs_identity"):
        assert observations[name]
    assert observations["mask_event_indices"] == ([1, 0] if mode == "swapped" else [0, 1])


@pytest.mark.parametrize("change", [{"positive": None}, {"cfg_merge": True}, {"tea_cache": object()},
                                   {"latents": torch.zeros(1, 2, 9, 2, 3)}])
def test_invalid_model_call_fails(change):
    wrapped = ADAPTER.WindowGuidance(lambda **call: call["latents"], "window", masks(), [torch.ones(1)] * 2)
    with pytest.raises(ValueError):
        wrapped(**dict(kwargs(), **change))


def test_model_must_not_strip_reference_prefix():
    wrapped = ADAPTER.WindowGuidance(lambda **call: call["latents"][:, :, 1:], "flatten", masks())
    with pytest.raises(RuntimeError, match="complete latent sequence"):
        wrapped(**kwargs())


@pytest.mark.parametrize("mode", ["", "time_swapped", None])
def test_invalid_mode_fails(mode):
    with pytest.raises(ValueError, match="timeline_guidance_mode"):
        ADAPTER.WindowGuidance(lambda **call: call["latents"], mode, masks())


def test_non_torch_and_overweight_masks_fail():
    with pytest.raises(ValueError, match="torch"):
        ADAPTER.WindowGuidance(None, "window", [[0] * 10] * 2)
    with pytest.raises(ValueError, match="occupancy"):
        ADAPTER.WindowGuidance(None, "window", [torch.ones(1, 1, 10, 1, 1)] * 2)


class FakeEmbedder:
    def __init__(self):
        self.prompts = []

    def encode_prompt(self, pipe, prompt):
        assert not torch.is_grad_enabled()
        self.prompts.append(prompt)
        return torch.tensor([[[float(len(self.prompts))]]])

    def process(self, pipe, prompt, positive):
        return {"context": self.encode_prompt(pipe, prompt)}


class FakePipe:
    def __init__(self, embedder, fail=False):
        self.embedder, self.fail = embedder, fail
        self.device, self.torch_dtype = "cpu", torch.float32
        self.calls = []
        self.model_fn = self.original

    def original(self, **call):
        self.calls.append(call)
        return torch.ones_like(call["latents"]) * call["context"].item()

    def __call__(self, **arguments):
        self.arguments = arguments
        assert not torch.is_grad_enabled()
        positive = self.embedder.process(self, arguments["prompt"], True)["context"]
        negative = self.embedder.process(self, arguments["negative_prompt"], False)["context"]
        if self.fail:
            raise RuntimeError("simulated generation failure")
        for step in range(arguments["num_inference_steps"]):
            inputs = kwargs()
            inputs["timestep"] = torch.tensor([1000. - step])
            self.model_fn(**dict(inputs, context=positive))
            self.model_fn(**dict(inputs, positive=False, context=negative))
        return [object() for _ in range(arguments["num_frames"])]


@pytest.mark.parametrize("mode", ADAPTER.MODES)
def test_sample_generation_encodes_once_restores_and_records_80_actual_calls(mode):
    embedder = FakeEmbedder()
    pipe = FakePipe(embedder)
    original_fn, original_process = pipe.model_fn, embedder.process
    row = {"prompt": "A red ball on a gray floor.", "timeline": TIMELINE}
    parameters = dict(ADAPTER.DEFAULTS, timeline_guidance_mode=mode)
    reference = object()
    video, metrics = ADAPTER.generate_sample(pipe, embedder, row, parameters, reference, masks(), 43)
    assert len(video) == 33
    assert pipe.model_fn == original_fn and embedder.process == original_process
    assert len(pipe.calls) == 80
    assert metrics["positive_denoiser_calls"] == 60
    assert metrics["negative_denoiser_calls"] == 20
    assert metrics["latent_shape"] == [1, 2, 10, 2, 3]
    assert metrics["prompt_encoding_calls"] == {"positive": 1, "negative": 1, "event": 0 if mode == "flatten" else 2}
    assert len(embedder.prompts) == (2 if mode == "flatten" else 4)
    if mode != "flatten":
        assert embedder.prompts[:3] == [row["prompt"], row["prompt"] + "\nmoves left", row["prompt"] + "\nmoves right"]
    else:
        assert embedder.prompts[0] == row["prompt"] + "\n0-2s: moves left; 2-4.125s: moves right"
    assert pipe.arguments["vace_reference_image"] is reference
    assert pipe.arguments["seed"] == 43
    assert pipe.arguments["cfg_merge"] is False
    assert pipe.arguments["tea_cache_l1_thresh"] is None
    assert "input_image" not in pipe.arguments and "input_video" not in pipe.arguments


def test_sample_failure_restores_original_attributes():
    embedder = FakeEmbedder()
    pipe = FakePipe(embedder, fail=True)
    original_fn, original_process = pipe.model_fn, embedder.process
    with pytest.raises(RuntimeError, match="simulated"):
        ADAPTER.generate_sample(pipe, embedder, {"prompt": "ball", "timeline": TIMELINE},
                                dict(ADAPTER.DEFAULTS, timeline_guidance_mode="window"), object(), masks(), 42)
    assert pipe.model_fn == original_fn and embedder.process == original_process


def test_seed_override_and_mode_independence():
    row = {"sample_id": "ball_seed42", "seed": 42}
    assert all(ADAPTER.sample_seed(row, {"seed": 1000, "timeline_guidance_mode": mode}) == 42
               for mode in ADAPTER.MODES)
    del row["seed"]
    assert len({ADAPTER.sample_seed(row, {"seed": 42, "timeline_guidance_mode": mode}) for mode in ADAPTER.MODES}) == 1


@pytest.mark.parametrize("seed", [-1, 2**31, True, 42.0, "42"])
def test_invalid_sample_seed_fails(seed):
    with pytest.raises(ValueError, match="seed"):
        ADAPTER.sample_seed({"sample_id": "ball", "seed": seed}, {})
