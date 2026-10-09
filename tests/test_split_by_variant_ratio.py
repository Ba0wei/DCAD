import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from dcad.variant_ratio import split


class VariantRatioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input"
        self.source.mkdir()
        self.output = self.root / "output"

    def write_log(self, sequences):
        path = self.source / "BPIC12.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["case_id", "name", "concept:name", "extra"])
            for i, sequence in enumerate(sequences):
                for activity in ["▶", *sequence, "■"]:
                    writer.writerow([str(i), activity, activity.split("+")[0].strip(), 'quoted,"value"\nline'])
        return path

    def fixture(self):
        return self.write_log(
            [(" A+START ", "B")] * 40 + [("A+COMPLETE", "B"),
                                           ("A+START", "B", "B"),
                                           ("A+COMPLETE", "B", "B")]
        )

    def args(self, *extra):
        return ["--datasets", "BPIC12", "--observed-ratios", "0.5", "0.75", "--base-seed", "42", "--repeats", "1",
                "--input-dir", str(self.source), "--output-dir", str(self.output), *extra]

    def test_historical_audit_is_compared_after_module_relocation(self):
        self.fixture()
        split.main(self.args())
        path = next(self.output.glob('BPIC12/seed_*/observed_*/split_manifest.json'))
        m = json.loads(path.read_text())
        m['validation_audit']['implementation_sha256'] = {'old_reader.py': 'historical'}
        path.write_text(json.dumps(m))
        split.main(self.args('--audit-only'))
        m['validation_audit']['gradient_train']['case_ids'] = ['incorrect-membership']
        path.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, 'validation audit is not reproducible'):
            split.main(self.args('--audit-only'))

    def test_name_variants_frequency_and_nested_coverage(self):
        log = split.load_log(self.fixture())
        self.assertEqual(len(log.variants), 4)  # concept:name would merge these into two variants
        discovery = split.variant_order(log, 42, [0.5, 0.75])
        order = discovery.variants
        self.assertEqual(len(discovery.startup_case_ids), 2)
        small, large = set(order[:2]), set(order[:3])
        self.assertLess(small, large)
        metrics = split.statistics(log, small, 0.5)
        self.assertEqual(metrics["observed_variants"], 2)
        self.assertNotEqual(metrics["train_case_ratio"], 0.5)
        self.assertEqual(metrics["unknown_test_activities"], 0)
        self.assertEqual(split.variant_order(log, 42, [0.5, 0.75]), discovery)
        cases = {case.case_id: case for case in log.cases}
        self.assertEqual(order, list(dict.fromkeys(cases[cid].sequence for cid in discovery.trace_case_ids)))
        self.assertEqual(set(discovery.trace_case_ids), set(cases))
        self.assertEqual(len(discovery.trace_case_ids), len(cases))
        self.assertEqual(discovery.trace_case_ids[:2], discovery.startup_case_ids)

    def test_generate_read_back_overwrite_and_reproducibility(self):
        self.fixture()
        split.main(self.args())
        original = {str(p.relative_to(self.output)): split.file_sha256(p)
                    for p in self.output.rglob("*") if p.is_file()}
        with self.assertRaises(FileExistsError):
            split.main(self.args())
        split.main(self.args("--overwrite"))
        repeated = {str(p.relative_to(self.output)): split.file_sha256(p)
                    for p in self.output.rglob("*") if p.is_file()}
        self.assertEqual(original, repeated)
        seed = split.derive_seed("BPIC12", 42, 1)
        manifest = json.loads((self.output / f"BPIC12/seed_{seed}/observed_50/split_manifest.json").read_text())
        self.assertEqual(manifest["activity_col"], "name")
        self.assertTrue(manifest["validation"]["read_back_verified"])
        self.assertEqual(len(manifest["variants"]), 4)
        self.assertEqual(manifest["validation_audit"]["unknown_normal_test_activities"], [])
        split.main(self.args("--audit-only"))
        self.assertEqual(original, {str(p.relative_to(self.output)): split.file_sha256(p)
                                    for p in self.output.rglob("*") if p.is_file()})
        # Re-auditing a changed truncation must actually replay preprocessing.
        with self.assertRaisesRegex(ValueError, "unknown activities after preprocessing/validation"):
            split.main(self.args("--audit-only", "--max-len", "1"))

    def test_missing_name_does_not_fall_back(self):
        path = self.source / "BPIC12.csv"
        path.write_text("case_id,concept:name\n1,A\n")
        with self.assertRaisesRegex(ValueError, "'name'"):
            split.load_log(path)

    def test_infeasible_greedy_cover_is_explicit(self):
        log = split.load_log(self.write_log([("A",), ("B",), ("C",), ("D",)]))
        with self.assertRaisesRegex(ValueError, "smallest budget is 1"):
            split.variant_order(log, 42, [0.25, 0.75])

    def test_read_back_detects_non_activity_column_changes(self):
        path = self.fixture()
        fields, cases = split.read_cases(path)
        path.write_text(path.read_text(encoding="utf-8-sig").replace("value", "changed"), encoding="utf-8-sig")
        with self.assertRaisesRegex(ValueError, "read-back validation failed"):
            split.validate_csv(path, fields, cases)

    def test_noncontiguous_cases_are_rejected(self):
        path = self.source / "BPIC12.csv"
        path.write_text("case_id,name\n1,A\n2,B\n1,C\n")
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            split.load_log(path)

    def test_rounding_and_ratio_names(self):
        log = split.load_log(self.write_log([("A",) * n for n in range(1, 6)]))
        order = split.variant_order(log, 42, [0.5]).variants
        metrics = split.statistics(log, set(order[:round(0.5 * 5)]), 0.5)
        self.assertEqual(metrics["observed_variants"], 2)
        self.assertEqual(metrics["actual_observed_ratio"], 0.4)
        self.assertEqual(split.ratio_directory(0.9), "observed_90")
        self.assertEqual(split.ratio_directory(0.05), "observed_5")
        self.assertEqual(split.ratio_directory(0.333), "observed_33p3")

    def test_late_observed_cases_are_still_training_members(self):
        log = split.load_log(self.fixture())
        discovery = split.variant_order(log, 42, [0.5, 0.75])
        observed = set(discovery.variants[:2])
        cases = {case.case_id: case for case in log.cases}
        seen = set()
        for cutoff, cid in enumerate(discovery.trace_case_ids):
            seen.add(cases[cid].sequence)
            if len(seen) == 2:
                break
        late = {cid for cid in discovery.trace_case_ids[cutoff + 1:] if cases[cid].sequence in observed}
        self.assertTrue(late)
        train, test = split.split_cases(log, observed)
        self.assertLessEqual(late, {case.case_id for case in train})
        self.assertFalse(late & {case.case_id for case in test})

    def test_trace_shuffle_is_frequency_weighted_not_variant_shuffle(self):
        log = split.load_log(self.write_log([("A",)] * 40 + [("A",) * n for n in range(2, 6)]))
        rng = random.Random(42)
        remaining = list(log.cases)
        rng.shuffle(remaining)
        startup = remaining.pop(0)  # All candidates cover the same activity; seeded tie order wins.
        rng.shuffle(remaining)
        expected = [startup, *remaining]
        discovery = split.variant_order(log, 42, [0.2, 0.8])
        self.assertEqual(discovery.trace_case_ids, [case.case_id for case in expected])
        self.assertEqual(discovery.variants, list(dict.fromkeys(case.sequence for case in expected)))

    def test_derived_seeds_independent_of_dataset_iteration_order(self):
        self.assertEqual(split.derive_seed("BPIC12", 42, 1), 307694762)
        seeds = [split.derive_seed(dataset, 42, repeat) for dataset in split.DATASETS for repeat in (1, 2, 3)]
        self.assertEqual(len(set(seeds)), 21)
        self.assertNotEqual(split.derive_seed("BPIC12", 42, 1), split.derive_seed("BPIC12", 43, 1))

    def test_validation_coverage_uses_actual_gradient_sequences(self):
        train_path = self.write_log([("A", "rare"), ("A",), ("A", "A"), ("A", "A", "A")])
        train_log = split.load_log(train_path)
        test_path = self.root / "test.csv"
        test_path.write_text("case_id,name\nt,A\nt,rare\n")
        _, test = split.read_cases(test_path)
        ids = {variant: f"v{i}" for i, variant in enumerate(train_log.variants)}
        ids[test[0].sequence] = ids.get(test[0].sequence, "test_variant")
        audit = split.audit_validation(train_path, test_path, train_log.cases, test, ids,
                                       split.AuditConfig(validation_ratio=0.5))
        self.assertGreater(audit["validation"]["case_count"], 0)
        self.assertIn("0", audit["gradient_train"]["case_ids"])
        self.assertIn("rare", audit["gradient_train"]["activities"])
        self.assertEqual(audit["unknown_validation_activities"], [])
        # All original activities still exist in the CSV, but truncation removes rare.
        with self.assertRaisesRegex(ValueError, "normal_test=.*rare"):
            split.audit_validation(train_path, test_path, train_log.cases, test, ids,
                                   split.AuditConfig(max_len=2))

    def test_model_reader_case_id_collision_rejected(self):
        path = self.source / "BPIC12.csv"
        path.write_text("case_id,name\n1,A\n 1 ,B\n")
        with self.assertRaisesRegex(ValueError, "collide"):
            split.load_log(path)


if __name__ == "__main__":
    unittest.main()
