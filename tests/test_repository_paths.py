from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dcad.fs import EventLogFile
from dcad.variant_ratio import pool, split


ROOT = Path(__file__).resolve().parents[1]


class RepositoryPathTests(unittest.TestCase):
    def test_default_data_paths_resolve_inside_the_repository(self):
        self.assertEqual(
            EventLogFile('BPIC12_custom_test').path,
            ROOT / 'data/processed/custom_test/BPIC12_custom_test.json.gz',
        )
        self.assertEqual(split.parse_args([]).input_dir, ROOT / 'data/processed/raw')
        self.assertEqual(pool.parse_args([]).pool_config, ROOT / 'data/variant_ratio/pool_config.json')
        for relative in pool.IMPLEMENTATIONS:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_data_commands_work_from_another_directory_without_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            for script in ('split_by_variant_ratio.py', 'build_variant_ratio_testsets.py',
                           'validate_variant_ratio.py'):
                with self.subTest(script=script):
                    result = subprocess.run(
                        [sys.executable, '-I', str(ROOT / 'scripts' / script), '--help'],
                        cwd=directory, text=True, capture_output=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('--', result.stdout)
