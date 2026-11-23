import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_nft_lora as nft  # noqa: E402


class RewardWeightTests(unittest.TestCase):
    def test_official_normalisation(self):
        self.assertEqual(nft.reward_weight(0.0, 5), 0.5)
        self.assertEqual(nft.reward_weight(5.0, 5), 1.0)
        self.assertEqual(nft.reward_weight(-10.0, 5), 0.0)
        self.assertAlmostEqual(nft.reward_weight(2.5, 5), 0.75)


class NftTermsTests(unittest.TestCase):
    def setUp(self):
        import torch
        g = torch.Generator().manual_seed(0)
        self.torch = torch
        self.x0 = torch.randn(1, 4, 3, 5, 6, generator=g)
        eps = torch.randn(1, 4, 3, 5, 6, generator=g)
        self.sigma = 0.6
        self.xt = (1 - self.sigma) * self.x0 + self.sigma * eps
        self.u = eps - self.x0                       # true velocity
        self.v_old = self.u + 0.5 * torch.randn(1, 4, 3, 5, 6, generator=g)
        self.v_ref = self.v_old.clone()

    def grad_at_old(self, r, beta=1.0):
        v = self.v_old.clone().requires_grad_(True)
        out = nft.nft_terms(v, self.v_old, self.v_ref, self.xt, self.x0, self.sigma, r, beta, 5.0, 0.0)
        out['total'].backward()
        return v.grad, out

    def test_positive_pulls_toward_data_negative_pushes_away(self):
        for beta in (1.0, 0.1):
            err0 = float(((self.v_old - self.u) ** 2).mean())
            g_pos, _ = self.grad_at_old(1.0, beta)
            g_neg, _ = self.grad_at_old(0.0, beta)
            step = 1e-3 / float(g_pos.abs().max())
            self.assertLess(float(((self.v_old - step * g_pos - self.u) ** 2).mean()), err0)
            self.assertGreater(float(((self.v_old - step * g_neg - self.u) ** 2).mean()), err0)

    def test_neutral_reward_has_no_policy_gradient_at_old_policy(self):
        g, out = self.grad_at_old(0.5)
        self.assertLess(float(g.abs().max()), 1e-6)
        self.assertLess(float(out['old_deviate']), 1e-12)

    def test_kl_term_pulls_toward_reference(self):
        v = (self.v_ref + 1.0).requires_grad_(True)
        out = nft.nft_terms(v, v.detach(), self.v_ref, self.xt, self.x0, self.sigma, 0.5, 1.0, 5.0, 1.0)
        out['total'].backward()
        self.assertGreater(float(v.grad.mean()), 0)  # descending reduces v back toward v_ref
        self.assertAlmostEqual(float(out['kl']), 1.0, places=5)


class SwapTests(unittest.TestCase):
    def test_swapped_restores_exactly(self):
        import torch
        params = [torch.nn.Parameter(torch.randn(3, 2)), torch.nn.Parameter(torch.randn(4))]
        live = [p.detach().clone() for p in params]
        other = [torch.zeros(3, 2), torch.ones(4)]
        with nft.swapped(params, other):
            self.assertTrue(all(torch.equal(p, o) for p, o in zip(params, other)))
        self.assertTrue(all(torch.equal(p, l) for p, l in zip(params, live)))
        with self.assertRaises(RuntimeError):
            with nft.swapped(params, other):
                raise RuntimeError('boom')
        self.assertTrue(all(torch.equal(p, l) for p, l in zip(params, live)))


class ArgsAndSourceTests(unittest.TestCase):
    def test_validate(self):
        base = ['--samples', 's', '--cache-index', 'c', '--out-dir', 'o', '--models-dir', 'm', '--max-steps', '20',
                '--max-gpu-hours', '4']
        nft.validate(nft.parser().parse_args(base))
        for extra in (['--nft-beta', '0'], ['--nft-beta', '1.5'], ['--adv-clip-max', '0'], ['--kl-beta', '-1'],
                      ['--warmup', '20'], ['--lr', '0']):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                nft.validate(nft.parser().parse_args(base + extra))

    def test_trainer_reuses_verified_e3_pieces(self):
        src = (Path(__file__).resolve().parents[1] / 'scripts/train_nft_lora.py').read_text()
        for name in ('tr.native_state', 'tr.adapters_disabled', 'tr.draw_t_noise', 'tr.export_lora',
                     'Adapters differ across DDP ranks', 'require_same_frozen_base_across_nodes', 'check_shard='):
            self.assertIn(name, src)
        self.assertIn('[:, :, 1:]', src)


class LoadRoundTests(unittest.TestCase):
    def test_round_samples_must_be_cached_with_float_advantage(self):
        entries = [{'sample_id': 'a_s1'}, {'sample_id': 'a_s2'}]
        idx = {'samples': entries, 'index_sha256': 'x'}
        with tempfile.TemporaryDirectory() as d, patch.object(nft.tr, 'load_cache_index', return_value=idx) as lci:
            p = Path(d) / 'samples.jsonl'
            p.write_text(json.dumps({'sample_id': 'a_s1', 'advantage': 0.5}) + '\n')
            data, _ = nft.load_round(p, 'idx', check_shard=(1, 4))
            self.assertEqual(data, [(entries[0], 0.5)])
            self.assertEqual(lci.call_args.kwargs['check_shard'], (1, 4))
            for bad in ({'sample_id': 'zz_s1', 'advantage': 0.5}, {'sample_id': 'a_s1', 'advantage': 1}):
                p.write_text(json.dumps(bad) + '\n')
                with self.assertRaises(ValueError):
                    nft.load_round(p, 'idx')


if __name__ == '__main__':
    unittest.main()
