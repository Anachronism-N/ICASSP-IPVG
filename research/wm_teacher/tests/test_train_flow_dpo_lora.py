import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_dpo_inside_term_sign_and_magnitude():
    import torch
    beta = 100.0
    def L(mw, rw, ml, rl):
        inside = -0.5 * beta * ((mw - rw) - (ml - rl))
        return float(-torch.nn.functional.logsigmoid(torch.tensor(inside))), inside
    l0, i0 = L(0.5, 0.5, 0.5, 0.5)
    assert abs(l0 - 0.6931) < 1e-3 and i0 == 0
    better, ib = L(0.49, 0.5, 0.51, 0.5)   # policy improves winner, worsens loser
    worse, iw = L(0.51, 0.5, 0.49, 0.5)
    assert ib > 0 and better < l0 < worse


def test_trainer_reuses_verified_e3_pieces():
    src = (Path(__file__).resolve().parents[1] / 'scripts/train_flow_dpo_lora.py').read_text()
    for name in ('tr.native_state', 'tr.adapters_disabled', 'tr.draw_t_noise', 'tr.export_lora', "Adapters differ across DDP ranks"):
        assert name in src
    assert '[:, :, 1:]' in src
