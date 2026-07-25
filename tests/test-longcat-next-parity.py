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


if __name__ == "__main__":
    unittest.main()
