#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).with_name("test-longcat-next-local-parity.py")
SPEC = importlib.util.spec_from_file_location("longcat_next_parity", SCRIPT)
PARITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARITY)


class ParityHarnessTests(unittest.TestCase):
    def test_direct_capture_inventory_is_exact(self):
        expected = {"inp_embd", "inp_embd_ngram", "h_nextn", "final_logits",
                    "l_out-0", "l_out-1", "l_out-2", "l_out-27"}
        expected.update(f"ngram_proj-{index}" for index in range(12))
        self.assertEqual(PARITY.DIRECT_NAMES, expected)
        self.assertEqual(len(PARITY.DIRECT_NAMES), 20)

    def test_all_block_manifest_and_localization_with_attention_mask(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = []
            references = {}
            mask = [0, 1]
            for block in range(28):
                reference = np.zeros((1, 2, 3), np.float32)
                candidate = reference.copy()
                candidate[0, 0, :] = 1000  # masked and therefore ignored
                if block >= 7:
                    candidate[0, 1, 0] = 1
                raw = root / f"all_blocks_l_out-{block}.raw"
                candidate.reshape(-1).tofile(raw)
                manifest.append(f"l_out-{block}\tf32\t3,2,1,1\t{raw.name}")
                references[f"physical_block_{block:02d}"] = reference
            (root / "all-blocks-diagnostics.tsv").write_text(
                "\n".join(manifest) + "\n", encoding="ascii")
            reference_path = root / "physical-blocks.npz"
            np.savez(reference_path, **references)
            with np.load(reference_path, allow_pickle=False) as archive:
                report = PARITY.compare_all_blocks(
                    archive, root, mask, {"atol": 0.125, "rtol": 0.03125})
            self.assertEqual(report["first_block_exceeding_diagnostic_criterion"], 7)
            self.assertEqual(report["last_block_within_diagnostic_criterion_before_failure"], 6)
            self.assertEqual(len(report["blocks"]), 28)
            self.assertEqual(report["blocks"][0]["maximum_absolute_error"], 0)
            self.assertFalse(report["accepted"])

    def test_all_block_manifest_rejects_duplicate_and_malformed_rows(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "all-blocks-diagnostics.tsv"
            path.write_text("bad\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "malformed"):
                PARITY.read_all_blocks_manifest(path)
            rows = [f"l_out-{block}\tf32\t3,1,1,1\tx.raw" for block in range(28)]
            path.write_text("\n".join(rows + [rows[0]]) + "\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                PARITY.read_all_blocks_manifest(path)
            rows[-1] = "l_out-28\tf32\t3,1,1,1\tx.raw"
            path.write_text("\n".join(rows) + "\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "capture-name mismatch"):
                PARITY.read_all_blocks_manifest(path)

    def test_all_block_cli_requires_one_case_and_reference(self):
        required = ["--model", "m", "--reference-dir", "r", "--precision", "bf16",
                    "--output-dir", "o", "--capture-exe", "capture"]
        parser = PARITY.build_parser()
        disabled = parser.parse_args(required)
        PARITY.validate_all_blocks_options(disabled)
        for extra in (["--all-blocks-diagnostic", "1"],
                      ["--all-blocks-diagnostic", "1", "--case", "a"],
                      ["--all-blocks-reference-npz", "blocks.npz"]):
            with self.assertRaises(ValueError):
                PARITY.validate_all_blocks_options(parser.parse_args(required + extra))
        enabled = parser.parse_args(required + ["--all-blocks-diagnostic", "1",
                                    "--all-blocks-reference-npz", "blocks.npz",
                                    "--case", "eos_window_position_2"])
        PARITY.validate_all_blocks_options(enabled)

    def test_reference_finiteness_preflight_writes_report(self):
        class Fixture:
            files = ["finite", "invalid", "integer"]
            data = {"finite": np.array([1.0]), "invalid": np.array([0.0, np.nan]),
                    "integer": np.array([1], dtype=np.int64)}
            def __getitem__(self, key): return self.data[key]
        with tempfile.TemporaryDirectory() as td:
            report = PARITY.validate_reference_finiteness(Fixture(), Path(td))
            self.assertFalse(report["passed"])
            self.assertEqual(report["invalid_arrays"]["invalid"]["non_finite_count"], 1)
            self.assertTrue((Path(td) / "reference-validation.json").is_file())

    def test_finite_reference_preflight_passes(self):
        class Fixture:
            files = ["finite"]
            def __getitem__(self, key): return np.array([1.0, 2.0])
        with tempfile.TemporaryDirectory() as td:
            validation = PARITY.validate_reference_finiteness(Fixture(), Path(td))
            self.assertTrue(validation["passed"])
            called = []
            PARITY.run_capture_after_validation(
                validation, ["capture"], runner=lambda *args, **kwargs: called.append((args, kwargs)))
            self.assertEqual(called[0][0], (["capture"],))
            self.assertTrue(called[0][1]["check"])

    def test_invalid_reference_prevents_capture_invocation(self):
        called = []
        with self.assertRaisesRegex(ValueError, "bad/key"):
            PARITY.run_capture_after_validation(
                {"passed": False, "invalid_arrays": {"bad/key": {}}}, ["capture"],
                runner=lambda *args, **kwargs: called.append(args))
        self.assertEqual(called, [])

    def raw(self, values, dtype, dims, kind="hidden"):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.raw"
            if dtype == "bf16":
                data = (np.asarray(values, np.float32).view(np.uint32) >> 16).astype(np.uint16)
            else:
                data = np.asarray(values, {"f16": np.float16, "f32": np.float32}[dtype])
            data.tofile(path)
            return PARITY.decode_raw(dtype, dims, path, kind)

    def test_hidden_layout_retains_batch_and_token_rank(self):
        one = self.raw([1, 2, 3], "f32", (3, 1, 1, 1))
        many = self.raw(range(6), "f32", (3, 2, 1, 1))
        self.assertEqual(one.shape, (1, 1, 3))
        self.assertEqual(many.shape, (1, 2, 3))
        np.testing.assert_array_equal(many, [[[0, 1, 2], [3, 4, 5]]])

    def test_final_normalized_multi_token_rank_is_not_output_selected(self):
        final = self.raw(range(12), "f32", (4, 3, 1, 1))
        self.assertEqual(final.shape, (1, 3, 4))
        self.assertEqual(PARITY.CPP_TO_REFERENCE["h_nextn"], "final_normalized_hidden_state")
        self.assertNotIn("result_norm", PARITY.CPP_TO_REFERENCE)
        self.assertIn("final_logits", PARITY.DIRECT_NAMES)

    def test_logits_retains_batch_rank(self):
        self.assertEqual(self.raw(range(5), "f32", (5, 1, 1, 1), "logits").shape, (1, 5))

    def test_all_raw_float_formats(self):
        for dtype in ("bf16", "f16", "f32"):
            got = self.raw([1.0, 2.0], dtype, (2, 1, 1, 1))
            np.testing.assert_allclose(got, [[[1.0, 2.0]]], rtol=0, atol=0)

    def test_combined_tolerance_pass_and_failure(self):
        ref = np.array([1.0], np.float32)
        self.assertTrue(PARITY.floating_result(ref, np.array([1.1]), {"atol": .05, "rtol": .05})["passed"])
        failed = PARITY.floating_result(ref, np.array([1.1001]), {"atol": .05, "rtol": .05})
        self.assertFalse(failed["passed"])
        self.assertGreater(failed["max_normalized_tolerance_violation"], 1)

    def test_exact_zero_policy_and_finite_requirement(self):
        self.assertFalse(PARITY.floating_result(np.array([1.]), np.array([1.001]), {"atol": 0, "rtol": 0})["passed"])
        with self.assertRaisesRegex(ValueError, "finite"):
            PARITY.floating_result(np.array([1.]), np.array([np.inf]), {"atol": 1, "rtol": 1})
        with self.assertRaisesRegex(ValueError, "complete C\\+\\+ direct logits"):
            PARITY.require_finite_logits(np.array([[0.0, np.nan]]), "case")

    def test_reversed_argsort_tie_prefers_larger_id(self):
        logits = np.array([[0., 2., 2., 1.]])
        self.assertEqual(np.argsort(logits, axis=-1)[:, ::-1][0, 0], 2)

    def test_manifest_rejects_duplicate_malformed_and_unexpected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "captures.tsv"
            path.write_text("bad\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "malformed"):
                PARITY.read_manifest(path)
            path.write_text("inp_embd\tf32\t1,1,1,1\ta\ninp_embd\tf32\t1,1,1,1\tb\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                PARITY.read_manifest(path)
            path.write_text("surprise\tf32\t1,1,1,1\ta\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "capture-name mismatch"):
                PARITY.read_manifest(path)

    def test_manifest_rejects_missing_capture(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "captures.tsv"
            path.write_text("inp_embd\tf32\t1,1,1,1\ta\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "missing"):
                PARITY.read_manifest(path)

    def test_case_manifest_preserves_reference_inputs_and_multiple_cases(self):
        class FakeNPZ:
            files = []
            def __init__(self):
                self.data = {}
                for case in ("a", "bos_left_zero"):
                    for suffix, value in {"input_ids": [[0, 7]], "attention_mask": [[0, 1]],
                                          "position_ids": [[1, 0]], "cache_position": [0, 1]}.items():
                        key = f"{case}/{suffix}"; self.files.append(key); self.data[key] = np.asarray(value)
            def __getitem__(self, key): return self.data[key]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "cases.json"
            cases = PARITY.make_case_manifest(FakeNPZ(), out)
            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[1]["attention_mask"], [0, 1])
            self.assertEqual(cases[1]["position_ids"], [1, 0])
            self.assertEqual(json.loads(out.read_text())["cases"], cases)

    def test_shape_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            PARITY.floating_result(np.zeros((1, 2, 3)), np.zeros((1, 1, 3)), {"atol": 1, "rtol": 1})

    def test_integer_and_greedy_mismatch_fail_exactly(self):
        self.assertTrue(PARITY.exact_result([1, 2, 3], [1, 2, 3])["passed"])
        self.assertFalse(PARITY.exact_result([1, 2, 3], [1, 2, 4])["passed"])
        self.assertFalse(PARITY.exact_result([10] * 8, [10] * 7 + [11])["passed"])

    def test_runtime_placement_argument_defaults_and_overrides(self):
        required = ["--model", "m", "--reference-dir", "r", "--precision", "bf16",
                    "--output-dir", "o", "--capture-exe", "capture"]
        defaults = PARITY.build_parser().parse_args(required)
        self.assertEqual((defaults.n_gpu_layers, defaults.threads), (0, 0))
        custom = PARITY.build_parser().parse_args(required + ["--n-gpu-layers", "99", "--threads", "8"])
        self.assertEqual((custom.n_gpu_layers, custom.threads), (99, 8))


if __name__ == "__main__":
    unittest.main()
