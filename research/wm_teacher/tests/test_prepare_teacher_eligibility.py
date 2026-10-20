import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('eligibility', Path(__file__).resolve().parents[1] / 'scripts/prepare_teacher_eligibility.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EligibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'training_metadata/ft2').mkdir(parents=True)
        (self.root / 'eval/diag').mkdir(parents=True)
        self.out = self.root / 'training_metadata/new_candidate'
        self.rows = [{'video': 'dataset/wan22_5s/egoexo4d_5s/a/window_w00' + str(i) + '.mp4',
                      'prompt': '  original caption ' + str(i) + chr(10) + ' second line  '}
                     for i in range(3)]
        self.scope = {'status': 'completed_metadata_potential_collision_scope_NOT_proof_of_media_overwrite',
                      'raw_data_modified': False, 'GPU_used': False,
                      'coverage': {'ft2': {'total_records': 3, 'ambiguous_namespace_records': 2,
                                          'multiple_candidate_run_records': 1}},
                      'private_cases': {'ft2': [{'video': row['video'], 'dataset': 'egoexo4d',
                                                'multiple_candidate_runs': i == 0}
                                               for i, row in enumerate(self.rows[:2])]}}
        raw = module.csv_bytes(self.rows)
        scope_raw = json.dumps(self.scope).encode()
        (self.root / module.CSV_REL).write_bytes(raw)
        (self.root / module.SCOPE_REL).write_bytes(scope_raw)
        for name, value in [('EXPECTED_ROWS', 3), ('EXPECTED_QUARANTINE', 1),
                            ('CSV_SHA', module.digest(raw)), ('SCOPE_SHA', module.digest(scope_raw))]:
            patcher = mock.patch.object(module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_partition_excludes_only_multiple_candidate_rows(self):
        kept, isolated = module.partition(self.rows, self.scope)
        self.assertEqual(kept, self.rows[1:])
        self.assertEqual(isolated, self.rows[:1])
        self.assertEqual(kept[0]['prompt'], self.rows[1]['prompt'])

    def test_real_private_csv_roundtrip_original_unchanged_and_not_approved(self):
        original = (self.root / module.CSV_REL).read_bytes()
        result = module.prepare(self.root, self.out)
        self.assertFalse(result['training_approved'])
        self.assertEqual(result['candidate_rows'], 2)
        self.assertEqual(result['quarantined_rows'], 1)
        self.assertEqual((self.root / module.CSV_REL).read_bytes(), original)
        with (self.out / 'namespace_candidates.csv').open(newline='', encoding='utf-8') as handle:
            self.assertEqual(list(csv.DictReader(handle)), self.rows[1:])
        self.assertEqual(self.out.stat().st_mode & 0o777, 0o700)
        for path in self.out.iterdir():
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads((self.out / 'eligibility_summary.json').read_text()), result)

    def test_existing_directory_or_symlink_is_not_overwritten(self):
        self.out.mkdir()
        (self.out / 'user_work').write_text('preserve')
        with self.assertRaises(ValueError):
            module.prepare(self.root, self.out)
        self.assertEqual((self.out / 'user_work').read_text(), 'preserve')
        link = self.root / 'training_metadata/link'
        link.symlink_to(self.out, target_is_directory=True)
        with self.assertRaises(ValueError):
            module.prepare(self.root, link)

    def test_modified_input_fails_before_destination_creation(self):
        path = self.root / module.CSV_REL
        path.write_bytes(path.read_bytes() + b'x')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            module.prepare(self.root, self.out)
        self.assertFalse(self.out.exists())

    def test_scope_bad_flags_counts_or_unknown_identity_rejected(self):
        for kind in ['flag', 'count', 'missing', 'duplicate']:
            scope = copy.deepcopy(self.scope)
            if kind == 'flag': scope['private_cases']['ft2'][0]['multiple_candidate_runs'] = 1
            elif kind == 'count': scope['coverage']['ft2']['multiple_candidate_run_records'] = 2
            elif kind == 'missing': scope['private_cases']['ft2'][1]['video'] = 'dataset/wan22_5s/egoexo4d_5s/a/missing.mp4'
            else: scope['private_cases']['ft2'][1] = scope['private_cases']['ft2'][0]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                module.partition(self.rows, scope)

    def test_duplicate_rows_and_escaping_paths_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[1] = rows[0]
        with self.assertRaises(ValueError):
            module.partition(rows, self.scope)
        for value in ['../private.mp4', '/abs.mp4', 'dataset/wan22_5s/a_5s/../bad.mp4', 'a' + chr(92) + 'b.mp4']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.dataset(value)

    def test_output_outside_training_metadata_rejected(self):
        with self.assertRaises(ValueError):
            module.prepare(self.root, self.root / 'dataset/new')
        self.assertFalse((self.root / 'dataset/new').exists())


if __name__ == '__main__':
    unittest.main()
