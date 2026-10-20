import copy
import importlib.util
from pathlib import Path
import pytest
np = pytest.importorskip('numpy')
from ipvg.common import write_json, sha256_file

spec = importlib.util.spec_from_file_location('real_training_analysis', Path(__file__).resolve().parents[1] / 'scripts/analyze_real_training.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def evidence(folder):
    (folder / 'outputs').mkdir()
    (folder / 'manifest.jsonl').write_text('fixture\n')
    artifact = folder / 'outputs/step-16.safetensors'
    artifact.write_bytes(b'checkpoint fixture')
    config = {'resources': {'gpus': 2, 'max_gpu_hours': 1},
              'parameters': {'learning_rate': 1e-5, 'num_epochs': 1, 'save_steps': 16}}
    write_json(folder / 'config.json', config)
    result = {'status': 'completed', 'task': 'training', 'run_id': 'fixture',
              'selected_manifest_sha256': sha256_file(folder / 'manifest.jsonl'),
              'artifact_sha256': {artifact.name: sha256_file(artifact)},
              'estimated_backend_gpu_hours': 2 * 10 / 3600, 'backend_elapsed_seconds': 10,
              'code_commit': 'fixture', 'source_sha256': 'fixture'}
    write_json(folder / 'result.json', result)
    rank = {'rank': 0, 'world_size': 2, 'status': 'passed', 'actual_optimizer_steps': 16,
            'backward_calls': 16, 'rank_parameters_synchronized': True, 'parameters_finite': True,
            'changed_trainable_tensors': 300, 'parameter_max_abs_change': .001,
            'initial_trainable_sha256': 'initial', 'final_trainable_sha256': 'final',
            'trainable_parameter_count': 10936320,
            'resource_end': {'peak_allocated_bytes': 19e9},
            'losses': [{'value': .05, 'finite': True} for _ in range(16)],
            'steps': [{'optimizer_step': i, 'completed': True, 'gradients_finite': True,
                       'nonzero_gradient_tensors': 150,
                       'learning_rates': [1e-5 / 3 if i <= 3 else 1e-5]} for i in range(1, 17)]}
    for index in range(2):
        row = copy.deepcopy(rank)
        row['rank'] = index
        write_json(folder / f'rank_{index}.json', row)
    return rank, result


def test_native_steps_actual_lr_and_hash_audit(tmp_path):
    evidence(tmp_path)
    report = module.training_summary(tmp_path, 16)
    assert report['actual_optimizer_steps_per_rank'] == 16
    assert report['first_step_learning_rate'] == pytest.approx(1e-5 / 3)
    assert report['last_step_learning_rate'] == pytest.approx(1e-5)
    assert report['epoch_mean_loss_all_ranks'] == pytest.approx([.05])


@pytest.mark.parametrize('problem', ['steps', 'lr', 'gradient', 'loss', 'rank_hash', 'checkpoint', 'empty_checkpoints', 'budget'])
def test_training_evidence_drift_rejected(tmp_path, problem):
    rank, result = evidence(tmp_path)
    rank['rank'] = 1
    if problem == 'steps':
        rank['actual_optimizer_steps'] = 15
    elif problem == 'lr':
        rank['steps'][0]['learning_rates'] = [1e-5]
    elif problem == 'gradient':
        rank['steps'][0]['gradients_finite'] = False
    elif problem == 'loss':
        rank['losses'][0] = {'finite': False, 'value': None}
    elif problem == 'rank_hash':
        rank['final_trainable_sha256'] = 'different'
    elif problem == 'checkpoint':
        (tmp_path / 'outputs/step-16.safetensors').write_bytes(b'drift')
    elif problem == 'empty_checkpoints':
        result['artifact_sha256'] = {}
    elif problem == 'budget':
        result['estimated_backend_gpu_hours'] = 2
    write_json(tmp_path / 'rank_1.json', rank)
    write_json(tmp_path / 'result.json', result)
    with pytest.raises(ValueError):
        module.training_summary(tmp_path, 16)


def test_index_requires_complete_unique_cohort():
    assert module.indexed([{'sample_id': 'one'}], {'one'}) == {'one': {'sample_id': 'one'}}
    for rows in ([], [{'sample_id': 'two'}], [{'sample_id': 'one'}, {'sample_id': 'one'}]):
        with pytest.raises(ValueError):
            module.indexed(rows, {'one'})
