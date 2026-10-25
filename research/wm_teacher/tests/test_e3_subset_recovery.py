import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

S=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(S))
spec=importlib.util.spec_from_file_location('recovery',S/'run_e3_subset_recovery.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.folder=self.root/'eval/e3_subset_20261025';self.folder.mkdir(parents=True)
        self.state=self.root/'state';self.state.mkdir()
        self.cache={'sha256':'c'}
        for i in range(4):
            (self.folder/f'node_{i}_preflight.json').write_text(json.dumps({'model_content_sha256':'same','plan_sha256':'approved','checkpoint_cache':self.cache,'hostname':os.uname().nodename,'models':[]}))
        for i in (1,2):
            (self.folder/f'node_{i}.finished.json').write_text(json.dumps({'exit_code':1,'owned_tree_released':True,'cap_exceeded':False,'gpu_hours':.201}))
    def tearDown(self):self.tmp.cleanup()
    def run_recovery(self, rc=0):
        args=['runner','--node-rank','1','--approved-plan-sha256','approved']
        with patch.object(r.n,'ROOT',self.root),patch.object(r.n,'STATE',self.state),patch.object(r.n,'verify_inputs',return_value=({},[],self.cache)),patch.object(r.n,'command_for',return_value=([sys.executable,'-c',f'raise SystemExit({rc})'],dict(os.environ))),patch.object(sys,'argv',args):
            with self.assertRaises(SystemExit) as e:r.main()
            return e.exception.code
    def test_success_preserves_old_receipts(self):
        original=(self.folder/'node_1.finished.json').read_bytes()
        self.assertEqual(self.run_recovery(),0)
        report=json.loads((self.folder/'node_1_recovery_a1.finished.json').read_text())
        self.assertTrue(report['owned_tree_released']);self.assertGreater(report['gpu_hours'],0)
        self.assertEqual(report['gpu_hours'],8*report['elapsed_seconds']/3600)
        self.assertEqual((self.folder/'node_1.finished.json').read_bytes(),original)
        self.assertFalse(list(self.state.glob('pause.*')))
    def test_failure_counted(self):
        self.assertEqual(self.run_recovery(3),1)
        report=json.loads((self.folder/'node_1_recovery_a1.finished.json').read_text())
        self.assertEqual(report['exit_code'],3);self.assertGreater(report['gpu_hours'],0)
    def test_duplicate_recovery_rejected(self):
        self.run_recovery()
        with self.assertRaisesRegex(AssertionError,'one recovery'):
            self.run_recovery()
    def test_budget_refuses_before_launch(self):
        (self.folder/'node_1.finished.json').write_text(json.dumps({'exit_code':1,'owned_tree_released':True,'cap_exceeded':False,'gpu_hours':2}))
        with self.assertRaisesRegex(AssertionError,'generation cap'):
            self.run_recovery()
        self.assertFalse((self.folder/'node_1_recovery_a1.launch.json').exists())
    def test_wrong_host_refuses(self):
        p=self.folder/'node_1_preflight.json';d=json.loads(p.read_text());d['hostname']='other';p.write_text(json.dumps(d))
        with self.assertRaisesRegex(AssertionError,'wrong recovery node'):
            self.run_recovery()

if __name__=='__main__':unittest.main()
