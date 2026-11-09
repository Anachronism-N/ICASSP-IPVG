import asyncio
from copy import deepcopy
import threading
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import pytest

import interactive_world_model as wm


class Frame:
    size = (1280, 704)

    def __init__(self, value=0):
        self.value = value

    def copy(self):
        return deepcopy(self)


class Backend:
    def __init__(self):
        self.requests = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def generate(self, request):
        self.requests.append(request)
        self.started.set()
        await self.release.wait()
        return wm.WindowResult((Frame(request.reference.value + 1),))


def observed(backend=None):
    s = wm.InteractiveSession(backend or Backend())
    s.push('observation', 0, {'frame': Frame(), 'entities': {'E1': {'position': 'left'}}}, evidence='verified-local-observation')
    return s


def run(coro):
    return asyncio.run(coro)


def test_capabilities_do_not_claim_learned_AR_student_audio_or_SOTA():
    assert wm.CAPABILITIES['window_chaining'] and wm.CAPABILITIES['duplex_input']
    assert not any(wm.CAPABILITIES[key] for key in ('learned_AR', 'student', 'audio_conditioning', 'raw_action_execution', 'realtime_proven', 'sota_proven'))


def test_two_windows_chain_predictions_without_overwriting_observed_memory():
    async def scenario():
        s = observed()
        before = deepcopy(s.memory)
        a = await s.rollout('The disk moves to the right.')
        b = await s.rollout('The disk remains in place.')
        assert a['status'] == b['status'] == 'predicted'
        assert [r.reference.value for r in s.backend.requests] == [0, 1]
        assert s.observed_reference.value == 0 and s.predicted_reference.value == 2
        assert s.memory == before and not b['observed_state_updated']
    run(scenario())


def test_forks_have_isolated_observed_state_and_prediction_history():
    async def scenario():
        root = observed()
        await root.rollout('The disk moves right.')
        left, right = root.fork('left'), root.fork('right')
        left.push('observation', 1, {'frame': Frame(9), 'entities': {'E1': {'position': 'right'}}}, evidence='verified-new-frame')
        assert root.memory == right.memory != left.memory
        assert not left.predictions and not right.predictions
        assert right.predicted_reference.value == 1
        await left.rollout('The disk stops.')
        assert left.predicted_reference.value == 10
        assert right.predicted_reference.value == root.predicted_reference.value == 1
    run(scenario())


@pytest.mark.parametrize('kind,payload', [
    ('interrupt', {}), ('text', {'text': 'Stop moving.'}),
    ('audio', {'text': 'Stop moving.'}),
    ('action', {'raw_action': {'key': 'left'}, 'semantic_action': 'Move left.', 'expected_outcome': {'position': 'left'}}),
    ('observation', {'frame': Frame(9), 'entities': {'E1': {'position': 'right'}}}),
])
def test_inputs_arrive_during_generation_and_stale_output_is_discarded(kind, payload):
    async def scenario():
        backend = Backend(); backend.release.clear()
        s = observed(backend)
        task = asyncio.create_task(s.rollout('The disk moves right.'))
        await backend.started.wait()
        s.push(kind, 1, payload, evidence='verified-local-event')
        backend.release.set()
        result = await task
        assert result['status'] == 'stale_discarded' and not result['compute_cancelled']
        assert s.predicted_reference is None and not s.predictions
        assert s.memory['E1']['value']['position'] == ('right' if kind == 'observation' else 'left')
    run(scenario())


def test_raw_action_and_expected_outcome_are_not_truth_or_executed_controls():
    async def scenario():
        s = observed()
        payload = {'raw_action': {'key': 'left'}, 'semantic_action': 'Move left.', 'expected_outcome': {'position': 'right'}}
        s.push('action', 1, payload, evidence='user-request')
        payload['raw_action']['key'] = 'changed'
        await s.rollout('Follow the requested movement.')
        assert s.events[-1]['payload']['raw_action']['key'] == 'left'
        assert 'Move left.' in s.backend.requests[0].prompt
        assert 'expected_outcome' not in s.backend.requests[0].prompt
        assert s.memory['E1']['value']['position'] == 'left'
    run(scenario())


@pytest.mark.parametrize('timestamp', [-1, float('nan'), float('inf'), True])
def test_invalid_input_is_atomic(timestamp):
    s = observed(); before = s.snapshot()
    with pytest.raises(ValueError):
        s.push('text', timestamp, {'text': 'hello'}, evidence='input')
    assert s.revision == before['revision'] and s.events == before['events']


def test_stream_ordering_snapshot_and_observation_validation():
    s = observed()
    s.push('text', 5, {'text': 'hello'}, evidence='input')
    s.push('audio', 2, {'text': 'hello'}, evidence='input')
    with pytest.raises(ValueError): s.push('text', 4, {'text': 'hello'}, evidence='input')
    before = s.revision
    with pytest.raises(ValueError): s.push('observation', 2, {'frame': Frame(), 'entities': {'E1': float('nan')}}, evidence='input')
    assert s.revision == before
    state = s.snapshot(); state['memory']['E1']['value']['position'] = 'changed'
    assert s.memory['E1']['value']['position'] == 'left'
    with pytest.raises(ValueError): s.fork('root')


def test_no_automatic_retry_after_backend_error():
    class Fail:
        calls = 0
        async def generate(self, request):
            self.calls += 1
            raise RuntimeError('backend failure')
    async def scenario():
        backend = Fail(); s = observed(backend)
        with pytest.raises(RuntimeError): await s.rollout('Move right.')
        assert backend.calls == 1 and not s.predictions and s.predicted_reference is None
    run(scenario())


def test_native_adapter_passes_exact_frozen_spec_and_counts_failed_calls():
    async def scenario():
        calls, checks = [], []
        def pipe(**kwargs):
            calls.append(kwargs)
            return [Frame()] * 121
        backend = wm.TeacherWindowBackend(pipe, 'fixed negative', lambda: checks.append(True), max_calls=1)
        s = observed(backend)
        result = await s.rollout('Move right.')
        assert len(result['result'].frames) == 121 and len(checks) == 2
        assert all(calls[0][key] == value for key, value in wm.WINDOW_SPEC.items())
        assert calls[0]['seed'] == 0 and calls[0]['input_image'].value == 0
        with pytest.raises(ValueError, match='quota'): await s.rollout('Move right.')
        assert len(calls) == backend.calls == 1
        bad = wm.TeacherWindowBackend(lambda **kw: [], 'fixed', lambda: None, max_calls=1)
        with pytest.raises(ValueError, match='121'): await observed(bad).rollout('Move right.')
        assert bad.calls == 1
    run(scenario())


def test_native_adapter_cancellation_does_not_release_running_thread():
    async def scenario():
        started, release = threading.Event(), threading.Event()
        calls = []
        def pipe(**kwargs):
            calls.append(kwargs); started.set()
            assert release.wait(5)
            return [Frame()] * 121
        backend = wm.TeacherWindowBackend(pipe, 'fixed', lambda: None, max_calls=2)
        request = wm.WindowRequest('a', 0, Frame(), 'Move right.', 0)
        task = asyncio.create_task(backend.generate(request))
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert backend.lock.locked()
        second = asyncio.create_task(backend.generate(request))
        await asyncio.sleep(0)
        assert len(calls) == 1
        release.set()
        with pytest.raises(asyncio.CancelledError): await task
        await second
        assert len(calls) == 2 and not backend.lock.locked()
    run(scenario())


@pytest.mark.parametrize('seed', [-1, 2, True])
def test_seed_search_is_rejected(seed):
    async def scenario():
        s = observed()
        with pytest.raises(ValueError): await s.rollout('Move right.', seed=seed)
        assert not s.backend.requests
    run(scenario())


def test_observation_required_before_rollout():
    async def scenario():
        with pytest.raises(ValueError, match='observation'):
            await wm.InteractiveSession(Backend()).rollout('Move right.')
    run(scenario())


@pytest.mark.parametrize('kind', ['text', 'audio'])
def test_user_utterance_conditions_next_request_without_updating_memory(kind):
    async def scenario():
        s = observed()
        s.push(kind, 1, {'text': 'Stop near the disk.'}, evidence='external-transcript')
        await s.rollout('Follow the instruction.')
        assert 'Stop near the disk.' in s.backend.requests[0].prompt
        assert s.memory['E1']['value']['position'] == 'left'
    run(scenario())


def test_nonfinite_and_NUL_payloads_rejected_before_state_mutation():
    s = observed()
    with pytest.raises(ValueError):
        s.push('action', 1, {'raw_action': {'key': 'bad\0'}, 'semantic_action': 'Move.', 'expected_outcome': {}}, evidence='input')
    assert s.revision == 1


def test_CPU_demo_is_not_video_generation_or_quality_evidence():
    result = run(wm.cpu_demo())
    assert result['synthetic_backend_calls'] == 4
    assert result['concurrent_input_and_stale_discard_pass']
    assert not any(result[key] for key in ('GPU_used', 'training_started', 'generated_video', 'real_audio_processed', 'model_quality_measured', 'sota_proven'))



def test_new_action_continues_from_committed_prediction_not_old_observation():
    async def scenario():
        s = observed()
        await s.rollout('Move right.')
        s.push('text', 1, {'text': 'Now move left.'}, evidence='user-input')
        await s.rollout('Follow the latest instruction.')
        assert s.backend.requests[-1].reference.value == 1
        assert s.observed_reference.value == 0
        assert 'prior model prediction' in s.backend.requests[-1].prompt
    run(scenario())
