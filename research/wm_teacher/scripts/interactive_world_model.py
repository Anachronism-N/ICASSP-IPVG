"""Inference-only window chaining and duplex orchestration; not a learned AR/student model.

The caller owns media safety, evidence verification, model loading, GPU authorization,
and hard process deadlines. This module opens no files, network services or device inputs.
"""
import asyncio
from copy import deepcopy
from dataclasses import dataclass
import json
import math


WINDOW_SPEC = dict(sigma_shift=5.0, cfg_scale=5.0, num_inference_steps=50,
                   num_frames=121, width=1280, height=704, tiled=True)
CAPABILITIES = dict(window_chaining=True, branch_memory=True, duplex_input=True,
                    learned_AR=False, student=False, audio_conditioning=False,
                    raw_action_execution=False, realtime_proven=False, sota_proven=False)


def _text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 8000 or '\0' in value:
        raise ValueError('Nonempty bounded text required')
    return value


def _json(value):
    raw = json.dumps(value, allow_nan=False, sort_keys=True)
    if len(raw) > 64000 or r'\u0000' in raw:
        raise ValueError('Bounded non-NUL JSON required')
    return deepcopy(value)


@dataclass(frozen=True)
class WindowRequest:
    branch: str
    revision: int
    reference: object
    prompt: str
    seed: int


@dataclass(frozen=True)
class WindowResult:
    frames: tuple
    fps: int = 24
    provenance: str = 'model_prediction_not_observation'


class TeacherWindowBackend:
    """Wrap an already loaded, externally authorized Wan pipeline; serialize GPU calls.

    Interrupts invalidate publication, not synchronous CUDA compute. Cancellation drains
    the thread before releasing the lock; outer process deadlines remain mandatory.
    """
    def __init__(self, pipe, negative_prompt, budget_check, *, max_calls):
        if not callable(pipe) or not callable(budget_check):
            raise ValueError('Loaded callable pipeline and budget callback required')
        if type(max_calls) is not int or max_calls <= 0:
            raise ValueError('Explicit positive call quota required')
        self.pipe = pipe
        self.negative_prompt = _text(negative_prompt)
        self.budget_check = budget_check
        self.max_calls = max_calls
        self.calls = 0
        self.lock = asyncio.Lock()

    async def generate(self, request):
        if type(request.seed) is not int or request.seed not in (0, 1):
            raise ValueError('Fixed seed0/1 required; no seed search')
        _text(request.prompt)
        if getattr(request.reference, 'size', None) != (1280, 704):
            raise ValueError('Exact preprocessed first image required')
        reference = request.reference.copy()
        async with self.lock:
            self.budget_check()
            if self.calls >= self.max_calls:
                raise ValueError('Call quota exhausted; failures count and no retry')
            self.calls += 1
            task = asyncio.create_task(asyncio.to_thread(
                self.pipe, input_image=reference, prompt=request.prompt,
                negative_prompt=self.negative_prompt, seed=request.seed, **WINDOW_SPEC))
            try:
                frames = await asyncio.shield(task)
            except asyncio.CancelledError:
                # Repeated cancellation must not expose the pipeline while its thread runs.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if task.done() and not task.cancelled():
                    task.exception()
                raise
            finally:
                self.budget_check()
            if not isinstance(frames, (list, tuple)) or len(frames) != 121:
                raise ValueError('Complete 121-frame window required')
            if any(getattr(frame, 'size', None) != (1280, 704) for frame in frames):
                raise ValueError('Full-spec output dimensions required')
            return WindowResult(tuple(frames))


class InteractiveSession:
    """Single-event-loop session. Real observations and predicted frames stay separate."""
    def __init__(self, backend, branch='root'):
        if not callable(getattr(backend, 'generate', None)):
            raise ValueError('Explicit async generation backend required')
        self.backend = backend
        self.branch = _text(branch)
        self.revision = 0
        self.events = []
        self.last_times = {}
        self.memory = {}
        self.observed_reference = None
        self.predicted_reference = None
        self.predictions = []
        self.lock = asyncio.Lock()

    def push(self, kind, timestamp, payload, *, evidence):
        if kind not in ('observation', 'action', 'text', 'audio', 'interrupt'):
            raise ValueError('Unsupported input stream')
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Finite nonnegative timestamp required')
        if timestamp < self.last_times.get(kind, -1):
            raise ValueError('Per-stream timestamps must not go backwards')
        _text(evidence)
        if len(self.events) >= 1024:
            raise ValueError('Session event cap reached; start a new episode')
        if not isinstance(payload, dict):
            raise ValueError('Typed payload required')
        frame = None
        data = dict(payload)
        if kind == 'observation':
            if set(data) != {'frame', 'entities'} or not isinstance(data['entities'], dict):
                raise ValueError('Observation needs frame and evidenced entities')
            frame = deepcopy(data.pop('frame'))
            if frame is None:
                raise ValueError('Observation frame required')
            if any(not isinstance(key, str) or not key or len(key) > 200 for key in data['entities']):
                raise ValueError('Generic nonempty entity IDs required')
            if len(set(self.memory) | set(data['entities'])) > 128:
                raise ValueError('Entity memory cap reached')
        elif kind == 'action':
            if set(data) != {'raw_action', 'semantic_action', 'expected_outcome'}:
                raise ValueError('Keep raw action, semantic action and expected outcome separate')
            _text(data['semantic_action'])
        elif kind in ('text', 'audio'):
            if set(data) != {'text'}:
                raise ValueError('Text or externally transcribed audio only; no native audio model')
            _text(data['text'])
        elif data:
            raise ValueError('Interrupt payload must be empty')
        data = _json(data)
        event = dict(kind=kind, timestamp=timestamp, evidence=evidence, payload=data)
        self.events.append(event)
        self.last_times[kind] = timestamp
        self.revision += 1
        if kind == 'observation':
            self.predicted_reference = None
            self.observed_reference = frame
            for key, value in data['entities'].items():
                self.memory[key] = dict(value=deepcopy(value), observed_at=timestamp,
                                        evidence=evidence, provenance='external_observation')
        return self.revision

    def snapshot(self):
        return deepcopy(dict(branch=self.branch, revision=self.revision, events=self.events,
                             memory=self.memory, last_times=self.last_times,
                             observed_reference=self.observed_reference,
                             predicted_reference=self.predicted_reference))

    def fork(self, branch):
        branch = _text(branch)
        if branch == self.branch:
            raise ValueError('A fork needs a distinct branch name')
        child = InteractiveSession(self.backend, branch)
        state = self.snapshot()
        for key in ('revision', 'events', 'memory', 'last_times',
                    'observed_reference', 'predicted_reference'):
            setattr(child, key, state[key])
        return child

    async def rollout(self, instruction, *, seed=0):
        _text(instruction)
        if type(seed) is not int or seed not in (0, 1):
            raise ValueError('Fixed seed0/1 required')
        async with self.lock:
            if len(self.predictions) >= 32:
                raise ValueError('Rollout window cap reached; start a new episode')
            reference = self.predicted_reference if self.predicted_reference is not None else self.observed_reference
            if reference is None:
                raise ValueError('An evidenced first-frame observation is required')
            revision = self.revision
            action = next((event['payload'] for event in reversed(self.events)
                           if event['kind'] == 'action'), None)
            prompt = instruction
            if self.memory:
                prompt += '\nExternally observed state (not predicted future): ' + json.dumps(self.memory, sort_keys=True)
            if action:
                prompt += '\nRequested action, not observed outcome: ' + action['semantic_action']
            utterance = next((event['payload']['text'] for event in reversed(self.events)
                              if event['kind'] in ('text', 'audio')), None)
            if utterance:
                prompt += '\nUser utterance (not observed state): ' + utterance
            if self.predicted_reference is not None:
                prompt += '\nReference is a prior model prediction, not a new real observation.'
            request = WindowRequest(self.branch, revision, deepcopy(reference), _text(prompt), seed)
            result = await self.backend.generate(request)
            if not isinstance(result, WindowResult) or not result.frames or result.fps != 24 or result.provenance != 'model_prediction_not_observation':
                raise ValueError('Prediction result with explicit provenance required')
            if revision != self.revision:
                return dict(status='stale_discarded', branch=self.branch, revision=revision,
                            current_revision=self.revision, compute_cancelled=False)
            self.predicted_reference = deepcopy(result.frames[-1])
            self.predictions.append(dict(revision=revision, frames=result.frames,
                                         provenance=result.provenance))
            return dict(status='predicted', branch=self.branch, revision=revision,
                        result=result, observed_state_updated=False)


async def cpu_demo():
    """Synthetic scheduling exercise. No video model, device, audio or training is used."""
    class SyntheticBackend:
        def __init__(self):
            self.calls = 0
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.release.set()

        async def generate(self, request):
            self.calls += 1
            self.started.set()
            await self.release.wait()
            return WindowResult((dict(step=request.reference['step'] + 1),))

    backend = SyntheticBackend()
    root = InteractiveSession(backend)
    root.push('observation', 0, dict(frame=dict(step=0), entities={'E1': {'position': 'left'}}),
              evidence='synthetic_external_observation')
    await root.rollout('Move right.')
    await root.rollout('Stay in place.')
    assert root.observed_reference['step'] == 0 and root.predicted_reference['step'] == 2
    branch = root.fork('alternate')
    branch.push('action', 1, dict(raw_action={'key': 'left'}, semantic_action='Move left.',
                                expected_outcome={'position': 'left'}), evidence='synthetic_user_request')
    await branch.rollout('Follow requested movement.')
    assert root.predicted_reference['step'] == 2 and branch.predicted_reference['step'] == 3
    backend.started.clear(); backend.release.clear()
    pending = asyncio.create_task(root.rollout('Move right.'))
    await backend.started.wait()
    root.push('audio', 2, {'text': 'Stop.'}, evidence='synthetic_external_transcript')
    root.push('interrupt', 2, {}, evidence='synthetic_user_interrupt')
    backend.release.set()
    result = await pending
    assert result['status'] == 'stale_discarded' and root.predicted_reference['step'] == 2
    assert root.memory['E1']['value']['position'] == branch.memory['E1']['value']['position'] == 'left'
    return dict(schema='interactive_world_model_CPU_demo_v1', execution='synthetic_CPU_only',
                synthetic_backend_calls=backend.calls, branch_isolation_pass=True,
                two_window_continuation_pass=True, concurrent_input_and_stale_discard_pass=True,
                observation_prediction_separation_pass=True, GPU_used=False,
                training_started=False, generated_video=False, real_audio_processed=False,
                capabilities=deepcopy(CAPABILITIES), model_quality_measured=False, sota_proven=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-demo', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(asyncio.run(cpu_demo()), sort_keys=True))
