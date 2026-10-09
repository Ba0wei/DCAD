import json
from pathlib import Path
import tempfile
import unittest

from dcad.variant_ratio import release, split


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'release'
        self.data.mkdir()
        self.file = self.data / 'splits' / 'example.csv'
        self.file.parent.mkdir()
        self.file.write_text('case_id,name\n1,A\n')
        self.index = {
            'base_seed': 42, 'repeat': 1,
            'splits': [{'dataset': d, 'ratio': r} for d in split.DATASETS for r in split.RATIOS],
            'files': [{'path': 'splits/example.csv', 'bytes': self.file.stat().st_size,
                       'sha256': split.file_sha256(self.file)}],
        }
        self.save()

    def save(self):
        (self.data / 'release_index.json').write_text(json.dumps(self.index))

    def test_portable_index_and_tampered_data(self):
        release.validate_index(self.data)
        self.file.write_text('case_id,name\n1,B\n')
        with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
            release.validate_index(self.data)

    def test_extra_repetition_file_is_rejected(self):
        (self.data / 'repeat_2.csv').write_text('unpublished')
        with self.assertRaisesRegex(ValueError, 'exactly'):
            release.validate_index(self.data)

    def test_index_path_escape_and_duplicates_are_rejected(self):
        self.index['files'][0]['path'] = '../outside.csv'
        self.save()
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            release.validate_index(self.data)
        self.index['files'][0]['path'] = 'splits/example.csv'
        self.index['files'] *= 2
        self.save()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            release.validate_index(self.data)

    def test_rebuild_compares_bytes_but_not_new_manifests(self):
        rebuilt = self.root / 'rebuilt'
        (rebuilt / 'splits').mkdir(parents=True)
        target = rebuilt / 'splits/example.csv'
        target.write_bytes(self.file.read_bytes())
        (rebuilt / 'splits/split_manifest.json').write_text('{"new_provenance": true}')
        release.compare_rebuilt(rebuilt, self.data)
        target.write_text('different')
        with self.assertRaisesRegex(ValueError, 'differs'):
            release.compare_rebuilt(rebuilt, self.data)

    def test_default_scope_is_one_repetition(self):
        args = split.parse_args([])
        self.assertEqual((args.base_seed, args.repeats), (42, 1))
        self.assertEqual(args.output_dir, split.ROOT / 'data/variant_ratio/splits')
