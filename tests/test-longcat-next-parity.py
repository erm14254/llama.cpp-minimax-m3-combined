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
    def test_component_manifest_and_three_way_router_diagnostics(self):
        names = PARITY.component_names()
        self.assertEqual(len(names), 110)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); rows = []; default = {}; math = {}
            cpp_values = {}
            for name in names:
                block = int(name[15:17]); suffix = name.split("__", 1)[1]
                width = (384 if suffix in {"router_logits", "router_probabilities", "router_selection_scores"}
                         else 12 if suffix in {"router_topk_indices", "router_topk_weights"}
                         else 1 if suffix == "identity_weight_sum" else 3072)
                is_indices = suffix == "router_topk_indices"
                value = np.zeros((1, 2, width), np.int32 if is_indices else np.float32)
                if is_indices:
                    value[:] = np.arange(12)
                default[name] = value.copy(); math[name] = value.copy(); cpp = value.copy()
                if name == "physical_block_00__router_topk_indices":
                    cpp[..., [0, 1]] = cpp[..., [1, 0]]  # ordered mismatch, identical set
                if name == "physical_block_00__router_topk_weights":
                    default[name][..., :2] = [0.25, 0.5]
                    math[name][..., :2] = [0.25, 0.5]
                    cpp[..., :2] = [0.5, 0.25]  # aligns exactly by expert ID after index swap
                if name == "physical_block_03__attention_output":
                    cpp[:, 0, :] = 1000  # masked
                    cpp[:, 1, 0] = 1     # first material attended discrepancy
                cpp_values[name] = cpp
                reverse = {value: key for key, value in PARITY.COMPONENT_CPP_BASE.items()}
                base = reverse[suffix]
                cpp_name = f"{base}-{block}"
                raw = root / f"{cpp_name}.raw"
                cpp.reshape(-1).tofile(raw)
                dims = "1,12,2,1" if suffix == "router_topk_weights" else f"{width},2,1,1"
                rows.append(f"{cpp_name}\t{'i32' if is_indices else 'f32'}\t{dims}\t{raw.name}")
            (root / "block-components-diagnostics.tsv").write_text(
                "\n".join(rows) + "\n", encoding="ascii")
            default_path = root / "default.npz"; math_path = root / "math.npz"
            np.savez(default_path, **default); np.savez(math_path, **math)
            with np.load(default_path, allow_pickle=False) as default_npz, \
                 np.load(math_path, allow_pickle=False) as math_npz:
                report = PARITY.compare_block_components(
                    default_npz, math_npz, root, [0, 1], {"atol": .125, "rtol": .03125})
            indices = next(row for row in report["components"]
                           if row["name"] == "physical_block_00__router_topk_indices")
            self.assertFalse(indices["exact_ordered_equality"])
            self.assertTrue(indices["per_token_selected_set_equality"])
            weights = next(row for row in report["components"]
                           if row["name"] == "physical_block_00__router_topk_weights")
            self.assertGreater(weights["returned_topk_order"]["cpp_vs_python_default"]["maximum_absolute_error"], 0)
            self.assertEqual(weights["expert_id_aligned"]["cpp_vs_python_default"]["maximum_absolute_error"], 0)
            self.assertEqual(report["first_component_outside_diagnostic_criterion_vs_both_backends"],
                             "physical_block_03__attention_output")
            self.assertEqual(report["first_large_discrepancy_classification"], "attention output")
            self.assertTrue(all(row["representation"] == "compact_logical"
                                for row in report["component_decode_provenance"]))
            with np.load(default_path, allow_pickle=False) as default_npz, \
                 np.load(math_path, allow_pickle=False) as math_npz:
                attribution = PARITY.numerical_attribution_report(
                    default_npz, math_npz, root, [0, 1], {"atol": .125, "rtol": .03125})
            self.assertEqual(len(attribution["router_semantics"]), 5)
            self.assertEqual(len(attribution["floating_components"]), 105)
            self.assertEqual(set(attribution["residual_reconstruction"]),
                             {"cpp", "python_default", "python_math"})
            self.assertEqual(attribution["revised_attribution_classification"],
                             "attention output divergence")

            # Replay the same complete 110-component capture using the precise
            # legacy 384-wide argsort-view storage span for every top-k index row.
            for block in range(0, 10, 2):
                name = f"physical_block_{block:02d}__router_topk_indices"
                values = cpp_values[name].reshape(2, 12)
                legacy = np.full((PARITY.ROUTED_EXPERT_COUNT + 12,), -777, np.int32)
                legacy[:12] = values[0]
                legacy[PARITY.ROUTED_EXPERT_COUNT:PARITY.ROUTED_EXPERT_COUNT + 12] = values[1]
                legacy.tofile(root / f"ffn_moe_topk-{block}.raw")
            with np.load(default_path, allow_pickle=False) as default_npz, \
                 np.load(math_path, allow_pickle=False) as math_npz:
                legacy_report = PARITY.compare_block_components(
                    default_npz, math_npz, root, [0, 1], {"atol": .125, "rtol": .03125})
            self.assertTrue(all(row["representation"] == "legacy_strided_argsort_view"
                                for row in legacy_report["component_decode_provenance"]))
            legacy_weights = next(row for row in legacy_report["components"]
                                  if row["name"] == "physical_block_00__router_topk_weights")
            self.assertEqual(
                legacy_weights["expert_id_aligned"]["cpp_vs_python_default"]["maximum_absolute_error"], 0)

    def test_router_topk_weight_production_layout_decodes_token_expert_order(self):
        name = "physical_block_00__router_topk_weights"
        values = np.arange(24, dtype=np.float32).reshape(1, 2, 12)
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "weights.raw"
            values.reshape(-1).tofile(raw)
            decoded = PARITY.decode_component_raw(name, "f32", (1, 12, 2, 1), raw)
            self.assertEqual(decoded.shape, (1, 2, 12))
            np.testing.assert_array_equal(decoded, values)
            for malformed in ((12, 2, 1, 1), (1, 2, 12, 1), (1, 12, 2, 2)):
                with self.assertRaisesRegex(ValueError, r"\[1,12,tokens,1\]"):
                    PARITY.decode_component_raw(name, "f32", malformed, raw)

    def test_router_topk_indices_compact_and_legacy_strided_replay(self):
        name = "physical_block_00__router_topk_indices"
        expected = np.arange(24, dtype=np.int32).reshape(1, 2, 12)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            compact = root / "compact.raw"
            expected.tofile(compact)
            provenance = []
            np.testing.assert_array_equal(
                PARITY.decode_component_raw(name, "i32", (12, 2, 1, 1), compact, provenance), expected)
            self.assertEqual(provenance[-1]["representation"], "compact_logical")

            legacy = root / "legacy.raw"
            storage = np.full((PARITY.ROUTED_EXPERT_COUNT + 12,), 9999, np.int32)
            storage[:12] = expected[0, 0]
            storage[PARITY.ROUTED_EXPERT_COUNT:PARITY.ROUTED_EXPERT_COUNT + 12] = expected[0, 1]
            storage.tofile(legacy)
            np.testing.assert_array_equal(
                PARITY.decode_component_raw(name, "i32", (12, 2, 1, 1), legacy, provenance), expected)
            self.assertEqual(provenance[-1]["representation"], "legacy_strided_argsort_view")
            for count in (23, 25, PARITY.ROUTED_EXPERT_COUNT + 11,
                          PARITY.ROUTED_EXPERT_COUNT + 13, 2 * PARITY.ROUTED_EXPERT_COUNT):
                invalid = root / f"invalid-{count}.raw"
                np.zeros(count, np.int32).tofile(invalid)
                with self.assertRaisesRegex(ValueError, "neither compact"):
                    PARITY.decode_component_raw(name, "i32", (12, 2, 1, 1), invalid)

    def test_component_manifest_rejects_duplicate_missing_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "block-components-diagnostics.tsv"
            path.write_text("bad\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "malformed"):
                PARITY.read_component_manifest(path)
            path.write_text("block_in-0\tf32\t3072,1,1,1\ta\n" * 2, encoding="ascii")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                PARITY.read_component_manifest(path)
        with self.assertRaisesRegex(ValueError, "finite"):
            PARITY.all_block_result(np.zeros((1, 1, 1)), np.full((1, 1, 1), np.nan),
                                    {"atol": 1, "rtol": 1})

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

    def test_component_replay_mode_forbids_capture_invocation_arguments(self):
        parser = PARITY.build_parser()
        replay = ["--reference-dir", "r", "--precision", "bf16", "--output-dir", "existing",
                  "--component-replay-only", "1", "--block-components-diagnostic", "1",
                  "--block-components-default-npz", "default.npz",
                  "--block-components-math-npz", "math.npz",
                  "--case", "eos_window_position_2"]
        PARITY.validate_all_blocks_options(parser.parse_args(replay))
        attribution = replay.copy()
        attribution[attribution.index("--component-replay-only")] = "--component-attribution-replay-only"
        PARITY.validate_all_blocks_options(parser.parse_args(attribution))
        for extra in (["--model", "model.gguf"], ["--capture-exe", "capture"],
                      ["--all-blocks-diagnostic", "1"]):
            with self.assertRaises(ValueError):
                PARITY.validate_all_blocks_options(parser.parse_args(replay + extra))

    def test_bf16_round_to_nearest_even_specials_and_subnormals(self):
        bits = np.array([
            0x3f808000,  # halfway, even upper BF16 LSB -> down
            0x3f818000,  # halfway, odd upper BF16 LSB -> up
            0x7f800000, 0xff800000, 0x7f800001,
            0x00000000, 0x80000000, 0x00008000, 0x00008001,
        ], dtype=np.uint32)
        rounded = PARITY.bf16_round_to_float32(bits.view(np.float32)).view(np.uint32)
        self.assertEqual(int(rounded[0]), 0x3f800000)
        self.assertEqual(int(rounded[1]), 0x3f820000)
        self.assertEqual(int(rounded[2]), 0x7f800000)
        self.assertEqual(int(rounded[3]), 0xff800000)
        self.assertTrue(np.isnan(rounded[4:5].view(np.float32))[0])
        self.assertEqual(int(rounded[5]), 0x00000000)
        self.assertEqual(int(rounded[6]), 0x80000000)
        self.assertEqual(int(rounded[7]), 0x00000000)
        self.assertEqual(int(rounded[8]), 0x00010000)

    def test_bf16_rounding_improves_worsens_and_changes_pass_state(self):
        raw = np.array([1.003], np.float32)
        rounded = PARITY.bf16_round_to_float32(raw)
        criterion = {"atol": 0.001, "rtol": 0.0}
        improved = PARITY.rounding_comparison(rounded, raw, criterion, True)
        self.assertEqual(improved["metric_changes_after_bf16_rounding"]["rms_error"], "improved")
        self.assertTrue(improved["pass_state_changed"])
        worsened = PARITY.rounding_comparison(raw, raw, criterion, True)
        self.assertEqual(worsened["metric_changes_after_bf16_rounding"]["rms_error"], "worsened")
        self.assertTrue(worsened["pass_state_changed"])

    def test_semantic_router_identity_and_real_expert_differences(self):
        sums = np.ones((1, 1, 1), np.float32)
        identity_only = PARITY.semantic_router_report(
            np.array([[[256, 300]]]), np.array([[[270, 301]]]), np.array([[[280, 302]]]),
            sums, sums, sums)
        self.assertTrue(identity_only["has_identity_only_id_substitution"])
        self.assertFalse(identity_only["has_real_expert_set_difference"])
        self.assertFalse(identity_only["has_identity_presence_difference"])
        real = PARITY.semantic_router_report(
            np.array([[[1, 256]]]), np.array([[[2, 256]]]), np.array([[[2, 256]]]),
            sums, sums, sums)
        self.assertTrue(real["has_real_expert_set_difference"])
        self.assertEqual(real["first_token_with_real_expert_set_mismatch"], 0)
        presence = PARITY.semantic_router_report(
            np.array([[[1, 2]]]), np.array([[[1, 256]]]), np.array([[[1, 256]]]),
            sums, sums, sums)
        self.assertTrue(presence["has_identity_presence_difference"])

    def test_router_classification_requires_semantic_difference_and_failing_shortcut(self):
        identity_router = [{"physical_block": 0, "has_real_expert_set_difference": False,
                            "has_identity_presence_difference": False,
                            "shortcut_within_criterion_vs_both": True}]
        local_passes = [{"suffix": suffix, "outside_vs_both": False,
                         "python_backends_outside_criterion": False}
                        for suffix in ("attention_output", "dense_output", "moe_shortcut", "block_output")]
        self.assertEqual(PARITY.classify_numerical_attribution(identity_router, local_passes, 7),
                         "cumulative numerical drift with no discrete local operator failure")
        real_router = [{"physical_block": 0, "has_real_expert_set_difference": True,
                        "has_identity_presence_difference": False,
                        "shortcut_within_criterion_vs_both": False}]
        self.assertEqual(PARITY.classify_numerical_attribution(real_router, local_passes, 7),
                         "real router-selection divergence")

    def test_residual_reconstruction_f32_bf16_and_odd_groupings(self):
        target_f32 = np.array([[[-2100.0]]], np.float32)
        residual = np.array([[[-1000.0]]], np.float32)
        dense = np.array([[[-1000.0]]], np.float32)
        shortcut = np.array([[[-100.0]]], np.float32)
        first = residual + dense
        reports = PARITY.reconstruction_metrics(target_f32, {
            "f32": first + shortcut,
            "two_stage_bf16": PARITY.bf16_round_to_float32(
                PARITY.bf16_round_to_float32(first) + shortcut),
            "old_bf16": PARITY.bf16_round_to_float32(
                PARITY.bf16_round_to_float32(dense + shortcut) + residual),
        })
        self.assertIn("f32", reports["byte_exact_alternatives"])
        self.assertNotEqual(
            reports["alternatives"]["two_stage_bf16"]["maximum_absolute_error"],
            reports["alternatives"]["old_bf16"]["maximum_absolute_error"])
        bf_target = PARITY.bf16_round_to_float32(first + shortcut)
        bf_report = PARITY.reconstruction_metrics(bf_target, {
            "final_bf16": PARITY.bf16_round_to_float32(first + shortcut)})
        self.assertIn("final_bf16", bf_report["byte_exact_alternatives"])

        arrays = {}
        for block in range(10):
            prefix = f"physical_block_{block:02d}__"
            arrays[prefix + "block_input"] = np.array([[[1.0]]], np.float32)
            arrays[prefix + "attention_output"] = np.array([[[2.0]]], np.float32)
            arrays[prefix + "post_attention_residual"] = np.array([[[3.0]]], np.float32)
            arrays[prefix + "dense_output"] = np.array([[[4.0]]], np.float32)
            if block % 2 == 0:
                arrays[prefix + "moe_shortcut"] = np.array([[[5.0]]], np.float32)
                arrays[prefix + "block_output"] = np.array([[[7.0]]], np.float32)
            else:
                arrays[prefix + "block_output"] = np.array([[[12.0]]], np.float32)
        reconstructed = PARITY.reconstruct_component_boundaries(arrays)
        self.assertIn("pure_f32_addition",
                      reconstructed[0]["attention_residual"]["byte_exact_alternatives"])
        self.assertIn("pure_f32_addition",
                      reconstructed[0]["block_output"]["byte_exact_alternatives"])
        self.assertIn("all_f32_official_association",
                      reconstructed[1]["block_output"]["byte_exact_alternatives"])

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
