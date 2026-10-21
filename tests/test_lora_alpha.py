"""Alpha validation and real native loader checks without CUDA execution."""
import importlib.util
import os
from pathlib import Path
import sys
import types
import pytest
from ipvg.common import write_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]

def adapter():
    spec = importlib.util.spec_from_file_location('vace_alpha_test', ROOT / 'scripts/adapters/diffsynth_vace.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('parameters', [
    {'lora_alpha': a, 'lora_path': 'checkpoint'} for a in (True, False, None, '0.5', 0, -1, 1.1, float('inf'))
] + [{'lora_alpha': .5}, {'lora_path': 'a', 'adapter_checkpoint': 'b'}])
def test_invalid_alpha_before_heavy_loading(parameters, monkeypatch, tmp_path):
    module = adapter()
    request = {'parameters': parameters, 'repo_root': str(tmp_path)}
    monkeypatch.setattr(module, 'load_json', lambda p: request)
    monkeypatch.setattr(module, 'local_model_paths', lambda *a: pytest.fail('Heavy loading reached'))
    monkeypatch.setattr(sys, 'argv', ['adapter', '--request', 'unused'])
    with pytest.raises(ValueError): module.main()


@pytest.mark.parametrize('parameters', [{}, {'lora_path': 'c'}, {'lora_path': 'c', 'lora_alpha': 1},
                                         {'lora_path': 'c', 'lora_alpha': .5}, {'lora_path': 'c', 'lora_alpha': .25}])
def test_valid_alpha_reaches_normal_request_validation(parameters, monkeypatch):
    module = adapter()
    monkeypatch.setattr(module, 'load_json', lambda p: {'parameters': parameters, 'repo_root': '.', 'task': 'training'})
    monkeypatch.setattr(sys, 'argv', ['adapter', '--request', 'unused'])
    with pytest.raises(ValueError, match='only supports inference'): module.main()


def native_loader():
    torch = pytest.importorskip('torch')
    upstream = os.environ.get('IPVG_TEST_DIFFSYNTH')
    if not upstream: pytest.skip('Explicit fixed local DiffSynth directory required')
    import subprocess
    revision = subprocess.check_output(['git', '-C', upstream, 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == '4e3496991455b300fd60ab023113422cd16899f5'
    spec = importlib.util.spec_from_file_location('native_alpha_loader', Path(upstream) / 'diffsynth/utils/lora/general.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return torch, module.GeneralLoRALoader


@pytest.mark.parametrize('alpha', [1, .5, .25])
def test_native_fusion_and_fresh_base(alpha):
    torch, loader_class = native_loader()
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    with torch.no_grad(): model[0].weight.fill_(1)
    original = model[0].weight.detach().clone()
    a = torch.tensor([[1., 2.]])
    b = torch.tensor([[3.], [4.]])
    state = {'0.lora_A.weight': a, '0.lora_B.weight': b}
    loader = loader_class(device='cpu', torch_dtype=torch.float32)
    loader.fuse_lora_to_base_model(model, state, alpha=alpha)
    assert torch.equal(model[0].weight, original + alpha * (b @ a))
    fresh = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    with torch.no_grad(): fresh[0].weight.copy_(original)
    loader.fuse_lora_to_base_model(fresh, state, alpha=alpha)
    assert torch.equal(fresh[0].weight, model[0].weight)
    loader.fuse_lora_to_base_model(model, state, alpha=alpha)
    assert torch.equal(model[0].weight, original + 2 * alpha * (b @ a))


@pytest.mark.parametrize('alpha', [None, 1, .5, .25])
def test_adapter_forwards_alpha_and_records_actual_reload(alpha, monkeypatch, tmp_path):
    torch, loader_class = native_loader()
    pytest.importorskip('PIL')
    module = adapter()
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    state = {'0.lora_A.weight': torch.ones((1, 2)), '0.lora_B.weight': torch.ones((2, 1))}
    checkpoint = tmp_path / 'checkpoint.safetensors'; checkpoint.write_bytes(b'fixture')
    forwarded = []
    pipe = types.SimpleNamespace(vace=model, torch_dtype=torch.float32, device='cpu',
                                 lora_loader=loader_class)
    def merge(model, path, alpha):
        forwarded.append(alpha)
        loader_class().fuse_lora_to_base_model(model, state, alpha=alpha)
    pipe.load_lora = merge
    pipeline = types.ModuleType('diffsynth.pipelines.wan_video')
    pipeline.WanVideoPipeline = types.SimpleNamespace(from_pretrained=lambda **kw: pipe)
    pipeline.ModelConfig = lambda **kw: kw
    core = types.ModuleType('diffsynth.core'); core.load_state_dict = lambda *a, **kw: state
    data = types.ModuleType('diffsynth.utils.data'); data.save_video = lambda *a, **kw: None
    for name, value in [('diffsynth.pipelines.wan_video', pipeline), ('diffsynth.core', core), ('diffsynth.utils.data', data)]:
        monkeypatch.setitem(sys.modules, name, value)
    for name in ('reset_peak_memory_stats', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ('max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'CPU fixture')
    monkeypatch.setattr(module, 'local_model_paths', lambda *a: ([], 'tokenizer'))
    monkeypatch.setattr(module, 'check_upstream', lambda *a: tmp_path)
    parameters = {'lora_path': str(checkpoint)}
    if alpha is not None: parameters['lora_alpha'] = alpha
    request = {'task': 'inference', 'parameters': parameters, 'repo_root': str(tmp_path),
               'resources': {'gpus': 1}, 'samples': [], 'run_dir': str(tmp_path), 'output_dir': str(tmp_path / 'out')}
    write_json(tmp_path / 'request.json', request)
    monkeypatch.setattr(sys, 'argv', ['adapter', '--request', str(tmp_path / 'request.json')])
    module.main()
    from ipvg.common import load_json
    reload = load_json(tmp_path / 'backend_metrics.json')['lora_reload']
    expected = 1 if alpha is None else alpha
    assert forwarded == [expected] and reload['alpha'] == expected
    assert reload['checkpoint_sha256'] == sha256_file(checkpoint)
    assert reload['matched_pairs'] == reload['changed_base_tensors'] == 1
