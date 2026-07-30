#!/usr/bin/env python3
import importlib.util
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).with_name("test-longcat-next-local-parity.py")
SPEC = importlib.util.spec_from_file_location("longcat_next_parity", SCRIPT)
PARITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARITY)

EXTRACTOR_PATH = Path(__file__).parents[1] / "scripts/longcat-next/extract-router-linear-diagnostic.py"
EXTRACTOR_SPEC = importlib.util.spec_from_file_location("longcat_router_extractor", EXTRACTOR_PATH)
EXTRACTOR = importlib.util.module_from_spec(EXTRACTOR_SPEC)
EXTRACTOR_SPEC.loader.exec_module(EXTRACTOR)
NORM_EXTRACTOR_PATH = Path(__file__).parents[1] / "scripts/longcat-next/extract-ffn-rmsnorm-diagnostic.py"
NORM_EXTRACTOR_SPEC = importlib.util.spec_from_file_location("longcat_norm_extractor", NORM_EXTRACTOR_PATH)
NORM_EXTRACTOR = importlib.util.module_from_spec(NORM_EXTRACTOR_SPEC)
NORM_EXTRACTOR_SPEC.loader.exec_module(NORM_EXTRACTOR)


class ParityHarnessTests(unittest.TestCase):
    def test_ffn_rmsnorm_names_vector_extraction_and_epsilon(self):
        expected = {(5, 10): ("model.layers.5.post_attention_layernorm.0.weight", "blk.10.ffn_norm.weight"),
                    (6, 12): ("model.layers.6.post_attention_layernorm.0.weight", "blk.12.ffn_norm.weight")}
        for logical, physical in NORM_EXTRACTOR.LAYERS:
            self.assertEqual((EXTRACTOR.resolve_tensor_name(NORM_EXTRACTOR.PYTHON_TEMPLATE, logical, physical),
                              EXTRACTOR.resolve_tensor_name(NORM_EXTRACTOR.GGUF_TEMPLATE, logical, physical)),
                             expected[(logical, physical)])
        vector, transposed = NORM_EXTRACTOR.canonical_norm_weight(np.zeros((1, 3072), np.float32))
        self.assertEqual(vector.shape, (3072,))
        self.assertTrue(transposed)
        with self.assertRaises(ValueError): NORM_EXTRACTOR.canonical_norm_weight(np.zeros((2, 1536), np.float32))
        record = NORM_EXTRACTOR.make_norm_weight_record(
            5, 10, "gguf", "blk.10.ffn_norm.weight", vector, "F32", True, "synthetic")
        self.assertEqual(record["canonical_tensor_orientation"], "hidden_vector")
        self.assertEqual(record["canonical_shape"], [3072])
        self.assertTrue(record["source_was_reshaped"])
        self.assertFalse(record["source_was_transposed"])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text('{"rms_norm_eps": 1e-6}')
            value, key = NORM_EXTRACTOR.python_epsilon(root)
            self.assertEqual((value, key), (1e-6, "rms_norm_eps"))

    def test_selective_bf16_and_f32_norm_vector_safetensor_extraction(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shard = root / "norm.safetensors"
            bf16 = torch.linspace(-1, 1, 3072).to(torch.bfloat16)
            f32 = torch.ones(3072)
            save_file({"norm.bf16": bf16, "norm.f32": f32, "unrequested": torch.zeros(3072)}, shard)
            index = {"weight_map": {name: shard.name for name in ("norm.bf16", "norm.f32", "unrequested")}}
            got, dtype, _ = EXTRACTOR.safetensor_weight(root, index, "norm.bf16")
            canonical, transposed = NORM_EXTRACTOR.canonical_norm_weight(got)
            self.assertEqual((dtype, canonical.shape, transposed), ("torch.bfloat16", (3072,), False))
            np.testing.assert_array_equal(canonical, bf16.float().numpy())
            got, dtype, _ = EXTRACTOR.safetensor_weight(root, index, "norm.f32")
            self.assertEqual((dtype, got.dtype), ("torch.float32", np.float32))

    def test_rmsnorm_diagnostics_equivalence_and_variants(self):
        weight = np.linspace(.5, 1.5, 32, dtype=np.float32)
        gguf = PARITY.bf16_round_to_float32(weight)
        equivalent = PARITY.norm_weight_epsilon_equivalence(weight, gguf, 1e-6, 1e-6)
        self.assertTrue(equivalent["shared_components_valid"])
        self.assertEqual(equivalent["weight_representation_classification"],
                         "BF16 runtime candidate and GGUF exactly equal")
        raw_equal = PARITY.norm_weight_epsilon_equivalence(weight, weight.copy(), 1e-6, 1e-6)
        self.assertEqual(raw_equal["weight_representation_classification"],
                         "raw checkpoint and GGUF exactly equal")
        self.assertFalse(raw_equal["shared_components_valid"])
        mismatch = gguf.copy()
        mismatch[0] += 1
        self.assertFalse(PARITY.norm_weight_epsilon_equivalence(weight, mismatch, 1e-6, 1e-6)[
            "shared_components_valid"])
        self.assertFalse(PARITY.norm_weight_epsilon_equivalence(weight, gguf, 1e-6, 2e-6)[
            "epsilon"]["operational_float32_epsilon_exact_equality"])
        operational = PARITY.norm_weight_epsilon_equivalence(
            weight, gguf, 1e-6, float(np.float32(1e-6)))
        self.assertFalse(operational["epsilon"]["raw_epsilon_exact_equality"])
        self.assertTrue(operational["epsilon"]["operational_float32_epsilon_exact_equality"])
        self.assertTrue(operational["shared_components_valid"])
        value = np.linspace(-2, 2, 64, dtype=np.float32).reshape(2, 32)
        outputs = {}
        for variant in PARITY.RMSNORM_VARIANTS:
            outputs[variant], scalars = PARITY.diagnostic_rmsnorm(value, weight, 1e-6, variant)
            self.assertTrue(np.isfinite(outputs[variant]).all())
            self.assertEqual(len(scalars["mean_square"]), 2)
        manual_x = np.asarray(value, np.float32)
        ms = np.mean(np.asarray(manual_x * manual_x, np.float32), axis=-1, keepdims=True, dtype=np.float32)
        inv = np.asarray(1 / np.sqrt(np.asarray(ms + np.float32(1e-6), np.float32)), np.float32)
        manual = PARITY.bf16_round_to_float32(PARITY.bf16_round_to_float32(manual_x * inv)
                                              * PARITY.bf16_round_to_float32(weight))
        np.testing.assert_array_equal(outputs["python_bf16_input_bf16_weight_contract"], manual)

    def test_torch_validates_pinned_python_rmsnorm_dtype_contracts(self):
        import torch
        epsilon = 1e-6

        def pinned(hidden, weight):
            dtype = hidden.dtype
            x32 = hidden.to(torch.float32)
            variance = x32.pow(2).mean(-1, keepdim=True)
            normalized = x32 * torch.rsqrt(variance + epsilon)
            return weight * normalized.to(dtype)
        hidden_bf16 = torch.linspace(-2, 2, 64).reshape(2, 32).to(torch.bfloat16)
        weight_bf16 = torch.linspace(.5, 1.5, 32).to(torch.bfloat16)
        actual = pinned(hidden_bf16, weight_bf16)
        diagnostic, _ = PARITY.diagnostic_rmsnorm(hidden_bf16.float().numpy(), weight_bf16.float().numpy(),
                                                  epsilon, "python_bf16_input_bf16_weight_contract")
        self.assertEqual(actual.dtype, torch.bfloat16)
        np.testing.assert_array_equal(actual.float().numpy(), diagnostic)
        weight_f32 = weight_bf16.float() + torch.linspace(0, .001, 32)
        promoted = pinned(hidden_bf16, weight_f32)
        promoted_diagnostic, _ = PARITY.diagnostic_rmsnorm(
            hidden_bf16.float().numpy(), weight_f32.numpy(), epsilon, "python_bf16_input_f32_weight_contract")
        self.assertEqual(promoted.dtype, torch.float32)
        np.testing.assert_allclose(promoted.numpy(), promoted_diagnostic, rtol=0, atol=0)
        hidden_f32 = hidden_bf16.float()
        actual_f32 = pinned(hidden_f32, weight_f32)
        f32_diagnostic, _ = PARITY.diagnostic_rmsnorm(
            hidden_f32.numpy(), weight_f32.numpy(), epsilon, "python_f32_input_f32_weight_contract")
        self.assertEqual(actual_f32.dtype, torch.float32)
        np.testing.assert_allclose(actual_f32.numpy(), f32_diagnostic, rtol=0, atol=0)

    def _write_norm_artifact(self, root):
        arrays = {key: np.ones((3072,), np.float32) for key in PARITY.FFN_RMSNORM_ARRAY_KEYS}
        npz = root / "norm.npz"
        EXTRACTOR.deterministic_npz(npz, arrays)
        records = []
        for logical, physical in ((5, 10), (6, 12)):
            for short, source in (("python", "python_checkpoint"), ("gguf", "gguf")):
                key = f"physical_block_{physical:02d}__{short}_ffn_norm_weight"
                records.append(NORM_EXTRACTOR.make_norm_weight_record(
                    logical, physical, source,
                    (f"model.layers.{logical}.post_attention_layernorm.0.weight" if short == "python" else
                     f"blk.{physical}.ffn_norm.weight"), arrays[key], "BF16", False, "synthetic"))
        metadata = {"schema_version": 1, "bounded_physical_blocks": [10, 12],
                    "model_instantiated": False, "inference_executed": False, "weight_records": records,
                    "python_epsilon": 1e-6, "gguf_epsilon": 1e-6, "python_epsilon_key": "rms_norm_eps",
                    "gguf_epsilon_key": "longcat-next.attention.layer_norm_rms_epsilon",
                    "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest()}
        js = root / "norm.json"
        js.write_text(json.dumps(metadata))
        return npz, js

    def test_rmsnorm_artifact_binding(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            npz, js = self._write_norm_artifact(root)
            arrays, metadata = PARITY.load_ffn_rmsnorm_artifacts(npz, js)
            self.assertEqual(set(arrays), PARITY.FFN_RMSNORM_ARRAY_KEYS)
            self.assertEqual(metadata["python_epsilon"], 1e-6)
            with self.assertRaises(ValueError): PARITY.load_ffn_rmsnorm_artifacts(js, npz)
            damaged = json.loads(js.read_text())
            damaged["npz_sha256"] = "0" * 64
            js.write_text(json.dumps(damaged))
            with self.assertRaises(ValueError): PARITY.load_ffn_rmsnorm_artifacts(npz, js)
            for mutation in ("extra", "missing", "shape", "dtype", "nonfinite"):
                values = {key: value.copy() for key, value in arrays.items()}
                first = next(iter(values))
                if mutation == "extra": values["extra"] = np.zeros(1, np.float32)
                elif mutation == "missing": values.pop(first)
                elif mutation == "shape": values[first] = np.zeros(3071, np.float32)
                elif mutation == "dtype": values[first] = values[first].astype(np.float64)
                else: values[first][0] = np.nan
                candidate = root / f"{mutation}.npz"
                EXTRACTOR.deterministic_npz(candidate, values)
                candidate_metadata = json.loads(json.dumps(metadata))
                candidate_metadata["npz_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
                candidate_json = root / f"{mutation}.json"
                candidate_json.write_text(json.dumps(candidate_metadata))
                with self.assertRaises(ValueError):
                    PARITY.load_ffn_rmsnorm_artifacts(candidate, candidate_json)

    def _rmsnorm_sides(self, pre_delta, norm_residual_delta, boundary_gap):
        hidden = 32
        prefix = "physical_block_10__"
        weight = np.ones(hidden, np.float32)
        inputs = {"cpp": np.ones((1, 1, hidden), np.float32),
                  "default": np.ones((1, 1, hidden), np.float32),
                  "math": np.ones((1, 1, hidden), np.float32)}
        inputs["default"][..., 0] += pre_delta
        inputs["math"][..., 0] += pre_delta
        inputs["default"] = PARITY.bf16_round_to_float32(inputs["default"])
        inputs["math"] = PARITY.bf16_round_to_float32(inputs["math"])
        outputs = {}
        for name in inputs:
            outputs[name], _ = PARITY.diagnostic_rmsnorm(
                inputs[name], weight, 1e-6, "python_bf16_input_bf16_weight_contract")
        outputs["default"][..., 0] += norm_residual_delta
        outputs["math"][..., 0] += norm_residual_delta
        outputs["default"] = PARITY.bf16_round_to_float32(outputs["default"])
        outputs["math"] = PARITY.bf16_round_to_float32(outputs["math"])
        router_weight = np.zeros((384, hidden), np.float32)
        router_weight[12, 0] = 1
        cpp_linear = PARITY.diagnostic_router_linear(outputs["cpp"].reshape(1, hidden), router_weight,
                                                     "float32_matmul").reshape(1, 1, 384)
        base = np.full((1, 1, 384), -10, np.float32)
        base[..., :11] = 10
        base[..., 11] = np.float32(boundary_gap)
        base[..., 12] = 0
        linear_residual = base - cpp_linear
        sides = {}
        for name in inputs:
            logits = PARITY.diagnostic_router_linear(outputs[name].reshape(1, hidden), router_weight,
                                                     "float32_matmul").reshape(1, 1, 384) + linear_residual
            probabilities = PARITY.diagnostic_softmax_stable_float32(logits)
            indices = PARITY._score_topk_indices(probabilities.reshape(384)).reshape(1, 1, 12)[..., ::-1]
            sides[name] = {prefix + "post_attention_residual": inputs[name], prefix + "ffn_norm": outputs[name],
                           prefix + "router_logits": logits, prefix + "router_probabilities": probabilities,
                           prefix + "router_selection_scores": probabilities.copy(), prefix + "router_topk_indices": indices}
        return sides, weight, router_weight

    def test_rmsnorm_end_to_end_membership_and_delta_projection(self):
        cases = ((1.0, 0.0, .1, "pre-norm input component sufficient"),
                 (0.0, .25, .1, "RMSNorm diagnostic residual sufficient"),
                 (1.0, .25, .1, "both components independently sufficient"),
                 (.02, .02, .03, "requires both components"),
                 (0.0, 0.0, .1, "native outcomes already equal"))
        reports = []
        for pre, residual, gap, expected in cases:
            sides, weight, router_weight = self._rmsnorm_sides(pre, residual, gap)
            report = PARITY.ffn_rmsnorm_block_decomposition(
                sides, "physical_block_10__", [0], weight, PARITY.bf16_round_to_float32(weight),
                1e-6, 1e-6, router_weight)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["source_lineage"]["capture_input_surface"], "post_attention_residual")
            self.assertFalse(report["source_lineage"]["block_input_is_norm_input"])
            classifications = {softmax["classification"] for norm in report["references"].values()
                               for matmul in norm["router_matmul_references"].values() for token in matmul
                               for softmax in token["softmax_references"].values()}
            self.assertEqual(classifications, {expected})
            for norm in report["references"].values():
                self.assertLess(norm["ffn_norm_delta_decomposition"][0]["rms_closure_error"], 1e-6)
                for matmul in norm["router_matmul_references"].values():
                    projected = matmul[0]["router_projected_decomposition"]
                    self.assertLess(projected["total_closure_rms"], 1e-6)
                    for softmax in matmul[0]["softmax_references"].values():
                        self.assertTrue(all(len(row["selected_expert_set"]) == 12
                                            for row in softmax["coalitions"].values()))
            reports.append(report)
        primary = PARITY.primary_ffn_rmsnorm_summary([
            {"physical_block": 10, "analysis": reports[0]},
            {"physical_block": 12, "analysis": reports[-1]}], 10)
        self.assertTrue(primary["physical_block_12_remains_aligned_control"])
        self.assertTrue(primary["cross_rmsnorm_agreement"])
        self.assertTrue(primary["cross_router_matmul_agreement"])
        self.assertTrue(primary["cross_router_softmax_agreement"])
        self.assertTrue(primary["cross_rmsnorm_matmul_softmax_agreement"])
        self.assertTrue(primary["all_required_references_agree"])
        self.assertEqual(primary["tokens_explained_by_pre_norm_input"], [0])
        self.assertIsNone(primary["stop_condition_conclusion"])
        negative = PARITY.primary_ffn_rmsnorm_summary([
            {"physical_block": 10, "analysis": reports[0]},
            {"physical_block": 12, "analysis": reports[0]}], 10)
        self.assertFalse(negative["physical_block_12_remains_aligned_control"])

    def test_rmsnorm_captured_membership_mismatch_is_not_decisive(self):
        sides, weight, router_weight = self._rmsnorm_sides(1.0, 0.0, .1)
        sides["default"]["physical_block_10__router_topk_indices"][0, 0, -1] = 383
        report = PARITY.ffn_rmsnorm_block_decomposition(
            sides, "physical_block_10__", [0], weight, weight, 1e-6, 1e-6, router_weight)
        for norm in report["references"].values():
            for tokens in norm["router_matmul_references"].values():
                for softmax in tokens[0]["softmax_references"].values():
                    self.assertFalse(softmax["native_python_membership_reconstruction_valid"])
                    self.assertFalse(softmax["classification_decisive"])
                    self.assertEqual(softmax["classification"], "analysis not decisive")

    def test_rmsnorm_distinct_weight_fallback_retains_four_combinations(self):
        sides, _, router_weight = self._rmsnorm_sides(1.0, 0.0, .1)
        varied_input = np.linspace(.123, 2.789, 32, dtype=np.float32).reshape(1, 1, 32)
        sides["cpp"]["physical_block_10__post_attention_residual"] = varied_input
        sides["default"]["physical_block_10__post_attention_residual"] = varied_input.copy()
        weight = np.linspace(.5001, 1.4999, 32, dtype=np.float32)
        gguf = weight.copy()
        audit = PARITY.norm_weight_epsilon_equivalence(weight, gguf, 1e-6, 1e-6)
        self.assertTrue(audit["raw"]["exact_byte_equality"])
        self.assertFalse(audit["operational_python_runtime_weight_equals_gguf"])
        report = PARITY.ffn_rmsnorm_block_decomposition(
            sides, "physical_block_10__", [0], weight, gguf, 1e-6, 1e-6, router_weight)
        fallback = report["distinct_weight_fallback"]
        self.assertFalse(fallback["two_component_input_vs_norm_residual_verdict_emitted"])
        for combinations in fallback["references"].values():
            self.assertEqual(len(combinations), 4)
            for evidence in combinations.values():
                self.assertEqual(set(evidence["captured_ffn_norm_comparisons"]), {"cpp", "default"})
                self.assertEqual(set(evidence["downstream_router_membership"]),
                                 set(PARITY.DIAGNOSTIC_LINEAR_VARIANTS))
                for softmaxes in evidence["downstream_router_membership"].values():
                    self.assertEqual(set(softmaxes), set(PARITY.DIAGNOSTIC_SOFTMAX_REFERENCES))
                    self.assertEqual(len(softmaxes["stable_float32"][0]["selected_expert_set"]), 12)
            gguf_case = combinations["cpp_input_gguf_weight"]
            python_case = combinations["cpp_input_python_bf16_runtime_weight"]
            self.assertEqual(gguf_case["operational_weight_dtype"], "float32")
            self.assertFalse(gguf_case["weight_was_bf16_rounded_before_multiplication"])
            self.assertEqual(gguf_case["epsilon_source"], "gguf")
            self.assertEqual(python_case["operational_weight_dtype"],
                             "bfloat16 represented as float32 values")
            self.assertTrue(python_case["weight_was_bf16_rounded_before_multiplication"])
            self.assertEqual(python_case["epsilon_source"], "python")
            self.assertNotEqual(gguf_case["diagnostic_ffn_norm_sha256"],
                                python_case["diagnostic_ffn_norm_sha256"])

    def test_rmsnorm_operational_weight_contract_preserves_off_bf16_grid_values(self):
        value = np.linspace(.123, 2.789, 32, dtype=np.float32).reshape(1, 32)
        raw_weight = np.linspace(.5001, 1.4999, 32, dtype=np.float32)
        rounded_weight = PARITY.bf16_round_to_float32(raw_weight)
        self.assertFalse(np.array_equal(raw_weight, rounded_weight))
        gguf, gguf_metadata = PARITY.diagnostic_rmsnorm_with_operational_weight(
            value, raw_weight, 1e-6, "float32_rmsnorm_then_final_bf16",
            "gguf_float32_runtime_weight")
        python, python_metadata = PARITY.diagnostic_rmsnorm_with_operational_weight(
            value, raw_weight, 1e-6, "float32_rmsnorm_then_final_bf16",
            "python_bf16_runtime_weight")
        self.assertFalse(np.array_equal(gguf, python))
        self.assertEqual(gguf_metadata["operational_weight_dtype"], "float32")
        self.assertFalse(gguf_metadata["weight_was_bf16_rounded_before_multiplication"])
        self.assertEqual(gguf_metadata["operational_weight_sha256"],
                         hashlib.sha256(raw_weight.tobytes()).hexdigest())
        self.assertTrue(python_metadata["weight_was_bf16_rounded_before_multiplication"])
        self.assertEqual(python_metadata["operational_weight_sha256"],
                         hashlib.sha256(rounded_weight.tobytes()).hexdigest())

    def _side_specific_rmsnorm_sides(self, input_delta=.25):
        hidden = 32
        prefix = "physical_block_10__"
        weight = np.linspace(.5001, 1.4999, hidden, dtype=np.float32)
        cpp_input = np.linspace(.123, 2.789, hidden, dtype=np.float32).reshape(1, hidden)
        python_input = cpp_input.copy()
        python_input[0, 0] += input_delta
        cpp_output, _ = PARITY.diagnostic_rmsnorm_with_operational_weight(cpp_input, weight, 1e-6,
                                                                          "float32_rmsnorm_then_final_bf16", "gguf_float32_runtime_weight")
        python_output, _ = PARITY.diagnostic_rmsnorm_with_operational_weight(python_input, weight, 1e-6,
                                                                             "python_bf16_input_bf16_weight_contract", "python_bf16_runtime_weight")
        router_weight = np.zeros((384, hidden), np.float32)
        router_weight[12, 0] = 1
        cpp_linear = PARITY.diagnostic_router_linear(cpp_output, router_weight, "float32_matmul")
        python_linear = PARITY.diagnostic_router_linear(python_output, router_weight, "float32_matmul")
        threshold = np.float32((cpp_linear[0, 12] + python_linear[0, 12]) / 2)
        base = np.full((1, 384), -10, np.float32)
        base[:, :11] = 10
        base[:, 11] = threshold
        residual = base - cpp_linear
        sides = {}
        for name, input_value, output in (("cpp", cpp_input, cpp_output),
                                          ("default", python_input, python_output), ("math", python_input, python_output)):
            logits = PARITY.diagnostic_router_linear(output, router_weight, "float32_matmul") + residual
            probabilities = PARITY.diagnostic_softmax_stable_float32(logits)
            indices = PARITY._score_topk_indices(probabilities[0]).reshape(1, 1, 12)[..., ::-1]
            sides[name] = {prefix + "post_attention_residual": input_value.reshape(1, 1, hidden),
                           prefix + "ffn_norm": output.reshape(1, 1, hidden),
                           prefix + "router_logits": logits.reshape(1, 1, 384),
                           prefix + "router_probabilities": probabilities.reshape(1, 1, 384),
                           prefix + "router_selection_scores": probabilities.reshape(1, 1, 384),
                           prefix + "router_topk_indices": indices}
        return sides, weight, router_weight

    def test_side_specific_rmsnorm_paths_and_native_contracts(self):
        sides, weight, router_weight = self._side_specific_rmsnorm_sides()
        report = PARITY.ffn_rmsnorm_side_specific_contract_decomposition(
            sides, "physical_block_10__", [1], weight, weight.copy(), 1e-6, 1e-6, router_weight)
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["per_token_applicability"][0]["applicable_cpp_arithmetic_references"])
        for reference in report["cpp_arithmetic_references"].values():
            self.assertTrue(reference["native_python_ffn_norm_reconstruction_by_token"][0][
                "native_python_ffn_norm_exact_for_token"])
            paths = reference["exact_path_decompositions"][0]
            self.assertLess(paths["input_first"]["exact_closure_rms"], 1e-6)
            self.assertLess(paths["contract_first"]["exact_closure_rms"], 1e-6)
            for rows in reference["router_matmul_references"].values():
                projected = rows[0]["router_projected_path_decomposition"]
                self.assertLess(projected["input_first_closure_rms"], 1e-6)
                self.assertLess(projected["contract_first_closure_rms"], 1e-6)
                for expert in projected["disputed_experts"]:
                    self.assertLess(abs(expert["input_first_closure_residual"]), 1e-6)
                    self.assertLess(abs(expert["contract_first_closure_residual"]), 1e-6)
                for softmax in rows[0]["softmax_references"].values():
                    if softmax["reference_applicable_to_token"]:
                        self.assertTrue(softmax["classification_decisive"])
                    else:
                        self.assertEqual(softmax["classification"],
                                         "analysis not decisive — native C++ surface not reconstructed")
                    self.assertEqual(len(softmax["coalitions"]["native_cpp"]["selected_expert_set"]), 12)
        broken = {side: dict(values) for side, values in sides.items()}
        broken["default"] = dict(broken["default"])
        broken["default"]["physical_block_10__ffn_norm"] = np.zeros_like(
            broken["default"]["physical_block_10__ffn_norm"])
        invalid = PARITY.ffn_rmsnorm_side_specific_contract_decomposition(
            broken, "physical_block_10__", [1], weight, weight.copy(), 1e-6, 1e-6, router_weight)
        for reference in invalid["cpp_arithmetic_references"].values():
            for rows in reference["router_matmul_references"].values():
                for softmax in rows[0]["softmax_references"].values():
                    self.assertFalse(softmax["classification_decisive"])
                    self.assertTrue(softmax["classification"].startswith("analysis not decisive"))

    def test_side_specific_float64_references_are_distinct(self):
        names = PARITY.SIDE_SPECIFIC_CPP_ARITHMETIC_REFERENCES
        self.assertEqual(len(names), 3)
        self.assertEqual(len(set(names)), 3)
        value = np.linspace(.123, 2.789, 3072, dtype=np.float32).reshape(1, 3072)
        weight = np.linspace(.5001, 1.4999, 3072, dtype=np.float32)
        before_multiply, _ = PARITY.diagnostic_rmsnorm_with_operational_weight(
            value, weight, 1e-6, names[1], "gguf_float32_runtime_weight")
        through_multiply, _ = PARITY.diagnostic_rmsnorm_with_operational_weight(
            value, weight, 1e-6, names[2], "gguf_float32_runtime_weight")
        self.assertFalse(np.array_equal(before_multiply, through_multiply))

    def test_side_specific_rmsnorm_classification_categories(self):
        classify = PARITY.classify_side_specific_rmsnorm
        self.assertEqual(classify(False, True, True, False), "pre-norm input component sufficient")
        self.assertEqual(classify(False, True, False, True),
                         "RMSNorm operational-contract component sufficient")
        self.assertEqual(classify(False, True, True, True), "both components independently sufficient")
        self.assertEqual(classify(False, True, False, False), "requires both components")
        self.assertEqual(classify(True, True, False, False), "native outcomes already equal")
        self.assertEqual(classify(False, False, True, True), "analysis not decisive")

    def _side_specific_summary_analysis(self, classifications, include_non_applicable=True,
                                        no_applicable=False):
        references = {}
        names = PARITY.SIDE_SPECIFIC_CPP_ARITHMETIC_REFERENCES
        for index, name in enumerate(names):
            applicable = not no_applicable and (index == 0 or not include_non_applicable)
            rows = []
            cpp_native = []
            python_native = []
            for token, classification in classifications.items():
                report = {"reference_applicable_to_token": applicable,
                          "classification_decisive": applicable,
                          "classification": (classification if applicable else
                                             "analysis not decisive — native C++ surface not reconstructed"),
                          "classification_before_required_reference_agreement": (classification if applicable else
                                                                                 "analysis not decisive — native C++ surface not reconstructed"),
                          "captured_topk_membership_reconstruction_valid": True,
                          "native_python_ffn_norm_reconstruction_valid": True}
                rows.append({"attended_token": token, "softmax_references": {
                    "stable_float32": dict(report), "stable_float64_then_float32": dict(report)},
                    "router_projected_path_decomposition": {}})
                cpp_native.append({"attended_token": token,
                                   "native_cpp_ffn_norm_exact_for_token": applicable})
                python_native.append({"attended_token": token,
                                      "native_python_ffn_norm_exact_for_token": True})
            references[name] = {"native_cpp_ffn_norm_reconstruction_by_token": cpp_native,
                                "native_python_ffn_norm_reconstruction_by_token": python_native,
                                "exact_path_decompositions": [], "router_matmul_references": {
                                    "float32_matmul": rows, "float64_matmul_then_float32": copy.deepcopy(rows)}}
        applicability = [{"attended_token": token,
                          "applicable_cpp_arithmetic_references": ([] if no_applicable else [names[0]]),
                          "non_applicable_cpp_arithmetic_references": (list(names) if no_applicable else list(names[1:])),
                          "applicable_reference_classification_agreement": not no_applicable}
                         for token in classifications]
        return {"status": "complete", "cpp_arithmetic_references": references,
                "per_token_applicability": applicability}

    def test_side_specific_summary_uses_only_applicable_decisive_results(self):
        aligned = self._side_specific_summary_analysis({1: "native outcomes already equal",
                                                        3: "native outcomes already equal"})
        all_input = self._side_specific_summary_analysis({
            1: "pre-norm input component sufficient", 3: "pre-norm input component sufficient"})
        summary = PARITY.primary_side_specific_rmsnorm_summary([
            {"physical_block": 10, "analysis": all_input},
            {"physical_block": 12, "analysis": aligned}], 10)
        self.assertTrue(summary["physical_block_12_remains_aligned_control"])
        self.assertEqual(summary["decisive_tokens_explained_by_pre_norm_input"], [1, 3])
        self.assertEqual(summary["decisive_tokens_explained_by_operational_contract"], [])
        self.assertIn("supersedes that result", summary["restrained_final_conclusion"])
        self.assertEqual(summary["next_tracks_not_implemented"],
                         ["upstream localization of post_attention_residual"])
        joint = self._side_specific_summary_analysis({
            1: "pre-norm input component sufficient", 3: "requires both components"})
        joint_summary = PARITY.primary_side_specific_rmsnorm_summary([
            {"physical_block": 10, "analysis": joint},
            {"physical_block": 12, "analysis": aligned}], 10)
        self.assertEqual(joint_summary["decisive_tokens_explained_by_pre_norm_input"], [1])
        self.assertEqual(joint_summary["decisive_tokens_requiring_both"], [3])
        self.assertIn("jointly caused", joint_summary["restrained_final_conclusion"])
        bad_control = self._side_specific_summary_analysis(
            {1: "native outcomes already equal", 3: "native outcomes already equal"}, no_applicable=True)
        self.assertFalse(PARITY.primary_side_specific_rmsnorm_summary([
            {"physical_block": 10, "analysis": all_input},
            {"physical_block": 12, "analysis": bad_control}], 10)[
                "physical_block_12_remains_aligned_control"])

    def test_residual_add_contracts_coalitions_and_minimal_sets(self):
        block_cpp = np.array([[-.9535515, -.8060354]], np.float32)
        attention_cpp = np.array([[1.2569029, -1.6323363]], np.float32)
        block_python = np.array([[.4004021, .9142421]], np.float32)
        attention_python = np.array([[-1.2483957, -1.7794135]], np.float32)
        cpp = PARITY.residual_add_contract(block_cpp, attention_cpp, "cpp_float32_add")
        python = PARITY.residual_add_contract(
            block_python, attention_python, "python_final_bf16_rne_add")
        np.testing.assert_array_equal(cpp, np.asarray(block_cpp + attention_cpp, np.float32))
        np.testing.assert_array_equal(python, PARITY.bf16_round_to_float32(
            np.asarray(block_python + attention_python, np.float32)))
        coalitions = PARITY.residual_three_factor_coalitions(
            block_cpp, attention_cpp, block_python, attention_python)
        self.assertEqual(set(coalitions), {f"{b}{a}{c}" for b in (0, 1) for a in (0, 1) for c in (0, 1)})
        np.testing.assert_array_equal(coalitions["101"], PARITY.residual_add_contract(
            block_python, attention_cpp, "python_final_bf16_rne_add"))
        self.assertFalse(np.array_equal(coalitions["101"] - coalitions["100"],
                                        coalitions["001"] - coalitions["000"]))
        cases = [({"100": True, "111": True}, "block-input component sufficient"),
                 ({"010": True, "111": True}, "attention-output component sufficient"),
                 ({"001": True, "111": True}, "residual-add operational-contract component sufficient"),
                 ({"110": True, "111": True}, "requires block-input and attention-output"),
                 ({"101": True, "111": True}, "requires block-input and add contract"),
                 ({"011": True, "111": True}, "requires attention-output and add contract"),
                 ({"111": True}, "requires all three components"),
                 ({"100": True, "010": True, "111": True},
                  "multiple single components independently sufficient"),
                 ({"110": True, "101": True, "111": True}, "multiple minimal sufficient coalitions")]
        for partial, expected in cases:
            truth = {key: partial.get(key, False) for key in coalitions}
            _, classification = PARITY.minimal_sufficient_residual_coalitions(truth)
            self.assertEqual(classification, expected)
        _, failed = PARITY.minimal_sufficient_residual_coalitions({key: False for key in coalitions})
        self.assertEqual(failed, "full Python operand coalition does not reproduce Python membership")
        _, invalid_equal = PARITY.minimal_sufficient_residual_coalitions(
            {key: key == "111" for key in coalitions}, native_equal=True, decisive=False)
        self.assertEqual(invalid_equal, "analysis not decisive")

    def _residual_three_factor_sides(self):
        hidden = 32
        prefix = "physical_block_10__"
        weight = np.ones(hidden, np.float32)
        cpp_block = np.linspace(.2, 1.8, hidden, dtype=np.float32).reshape(1, hidden)
        cpp_attention = np.linspace(-.1, .1, hidden, dtype=np.float32).reshape(1, hidden)
        python_block = cpp_block.copy()
        python_block[0, 0] += .5
        python_attention = cpp_attention.copy()
        python_attention[0, 0] += .25
        cpp_post = PARITY.residual_add_contract(cpp_block, cpp_attention, "cpp_float32_add")
        python_post = PARITY.residual_add_contract(
            python_block, python_attention, "python_final_bf16_rne_add")
        cpp_ffn, _ = PARITY.diagnostic_rmsnorm_with_operational_weight(cpp_post, weight, 1e-6,
                                                                       PARITY.SIDE_SPECIFIC_CPP_ARITHMETIC_REFERENCES[0], "gguf_float32_runtime_weight")
        python_downstream_ffn, _ = PARITY.diagnostic_rmsnorm_with_operational_weight(
            python_post, weight, 1e-6, PARITY.SIDE_SPECIFIC_CPP_ARITHMETIC_REFERENCES[0],
            "gguf_float32_runtime_weight")
        router_weight = np.zeros((384, hidden), np.float32)
        router_weight[12, 0] = 1
        cpp_linear = PARITY.diagnostic_router_linear(cpp_ffn, router_weight, "float32_matmul")
        python_linear = PARITY.diagnostic_router_linear(python_downstream_ffn, router_weight, "float32_matmul")
        threshold = np.float32((cpp_linear[0, 12] + python_linear[0, 12]) / 2)
        base = np.full((1, 384), -10, np.float32)
        base[:, :11] = 10
        base[:, 11] = threshold
        residual = base - cpp_linear
        sides = {}
        for name, block, attention, post, ffn in (("cpp", cpp_block, cpp_attention, cpp_post, cpp_ffn),
                                                  ("default", python_block, python_attention, python_post, python_downstream_ffn),
                                                  ("math", python_block, python_attention, python_post, python_downstream_ffn)):
            logits = PARITY.diagnostic_router_linear(ffn, router_weight, "float32_matmul") + residual
            probabilities = PARITY.diagnostic_softmax_stable_float32(logits)
            indices = PARITY._score_topk_indices(probabilities[0]).reshape(1, 1, 12)[..., ::-1]
            sides[name] = {prefix + "block_input": block.reshape(1, 1, hidden),
                           prefix + "attention_output": attention.reshape(1, 1, hidden),
                           prefix + "post_attention_residual": post.reshape(1, 1, hidden),
                           prefix + "ffn_norm": ffn.reshape(1, 1, hidden),
                           prefix + "router_logits": logits.reshape(1, 1, 384),
                           prefix + "router_probabilities": probabilities.reshape(1, 1, 384),
                           prefix + "router_selection_scores": probabilities.reshape(1, 1, 384),
                           prefix + "router_topk_indices": indices}
        return sides, weight, router_weight

    @staticmethod
    def _residual_profile(boundary=False, hidden=False):
        return {"longcat_bf16_boundary_rounding": boundary,
                "longcat_bf16_hidden_surface_rounding": hidden}

    def _equal_contract_residual_analysis(self, endpoint_failure=False):
        sides, weight, router_weight = self._residual_three_factor_sides()
        prefix = "physical_block_10__"
        old_cpp_ffn = sides["cpp"][prefix + "ffn_norm"].reshape(1, -1)
        linear_residual = sides["cpp"][prefix + "router_logits"].reshape(1, 384) - \
            PARITY.diagnostic_router_linear(old_cpp_ffn, router_weight, "float32_matmul")
        rounded_post = PARITY.residual_add_contract(
            sides["cpp"][prefix + "block_input"].reshape(1, -1),
            sides["cpp"][prefix + "attention_output"].reshape(1, -1),
            "python_final_bf16_rne_add")
        rounded_ffn, _ = PARITY.diagnostic_rmsnorm_with_operational_weight(
            rounded_post, weight, 1e-6, PARITY.SIDE_SPECIFIC_CPP_ARITHMETIC_REFERENCES[0],
            "gguf_float32_runtime_weight")
        rounded_logits = PARITY.diagnostic_router_linear(
            rounded_ffn, router_weight, "float32_matmul") + linear_residual
        rounded_probabilities = PARITY.diagnostic_softmax_stable_float32(rounded_logits)
        sides["cpp"][prefix + "post_attention_residual"] = rounded_post.reshape(1, 1, -1)
        sides["cpp"][prefix + "ffn_norm"] = rounded_ffn.reshape(1, 1, -1)
        sides["cpp"][prefix + "router_logits"] = rounded_logits.reshape(1, 1, 384)
        sides["cpp"][prefix + "router_probabilities"] = rounded_probabilities.reshape(1, 1, 384)
        sides["cpp"][prefix + "router_selection_scores"] = rounded_probabilities.reshape(1, 1, 384)
        sides["cpp"][prefix + "router_topk_indices"] = PARITY._score_topk_indices(
            rounded_probabilities[0]).reshape(1, 1, 12)[..., ::-1]
        if endpoint_failure:
            for target in ("default", "math"):
                replacement = np.zeros((1, 1, 384), np.float32)
                replacement[0, 0, 100:112] = np.arange(1, 13, dtype=np.float32)
                sides[target][prefix + "router_probabilities"] = replacement.copy()
                sides[target][prefix + "router_selection_scores"] = replacement.copy()
                sides[target][prefix + "router_topk_indices"] = PARITY._score_topk_indices(
                    replacement[0, 0]).reshape(1, 1, 12)[..., ::-1]
        return PARITY.post_attention_residual_three_factor_decomposition(
            sides, prefix, [1], weight, weight.copy(), 1e-6, 1e-6, router_weight,
            self._residual_profile(True, True))

    def _differing_contract_endpoint_failure_analysis(self):
        sides, weight, router_weight = self._residual_three_factor_sides()
        prefix = "physical_block_10__"
        for target in ("default", "math"):
            replacement = np.zeros((1, 1, 384), np.float32)
            replacement[0, 0, 100:112] = np.arange(1, 13, dtype=np.float32)
            sides[target][prefix + "router_probabilities"] = replacement.copy()
            sides[target][prefix + "router_selection_scores"] = replacement.copy()
            sides[target][prefix + "router_topk_indices"] = PARITY._score_topk_indices(
                replacement[0, 0]).reshape(1, 1, 12)[..., ::-1]
        return PARITY.post_attention_residual_three_factor_decomposition(
            sides, prefix, [1], weight, weight.copy(), 1e-6, 1e-6, router_weight,
            self._residual_profile())

    def _bias_provenance_analysis(self, invalidate_native_python=False):
        sides, weight, router_weight = self._residual_three_factor_sides()
        prefix = "physical_block_10__"
        for target in ("default", "math"):
            probabilities = sides[target][prefix + "router_probabilities"].copy()
            bias = np.zeros((1, 1, 384), np.float32)
            bias[0, 0, 100:112] = 20
            selection = probabilities + bias
            sides[target][prefix + "router_selection_scores"] = selection
            indices = PARITY._score_topk_indices(selection[0, 0]).reshape(1, 1, 12)[..., ::-1]
            if invalidate_native_python: indices[0, 0, 0] = 383
            sides[target][prefix + "router_topk_indices"] = indices
        return PARITY.post_attention_residual_three_factor_decomposition(
            sides, prefix, [1], weight, weight.copy(), 1e-6, 1e-6, router_weight,
            self._residual_profile())

    def test_residual_three_factor_downstream_and_six_paths(self):
        sides, weight, router_weight = self._residual_three_factor_sides()
        report = PARITY.post_attention_residual_three_factor_decomposition(
            sides, "physical_block_10__", [1], weight, weight.copy(), 1e-6, 1e-6, router_weight,
            self._residual_profile())
        self.assertEqual(report["status"], "complete")
        for target in report["python_targets"].values():
            self.assertTrue(target["native_residual_add_reconstruction"][0]["cpp_native_exact"])
            self.assertTrue(target["native_residual_add_reconstruction"][0]["python_native_exact"])
            self.assertTrue(target["per_token_applicability"][0]["applicable_cpp_rmsnorm_references"])
            for reference in target["references"].values():
                for rows in reference["router_matmul_references"].values():
                    row = rows[0]
                    self.assertEqual(len(row["six_factor_order_paths"]), 6)
                    for path in row["six_factor_order_paths"]:
                        self.assertTrue(path["residual_surface"]["closure_exact_equality"])
                        self.assertTrue(path["fixed_cpp_rmsnorm_output"]["closure_exact_equality"])
                        self.assertTrue(path["evaluated_router_logits"]["closure_exact_equality"])
                        self.assertLess(path["residual_surface"]["closure_rms"], 1e-6)
                        self.assertLess(path["fixed_cpp_rmsnorm_output"]["closure_rms"], 1e-6)
                        self.assertLess(path["evaluated_router_logits"]["closure_rms"], 1e-6)
                        for expert in path["evaluated_router_logits"]["disputed_experts"]:
                            self.assertLess(abs(expert["closure_residual"]), 1e-6)
                    for softmax in row["softmax_references"].values():
                        self.assertEqual(len(softmax["coalitions"]), 8)
                        self.assertTrue(softmax["coalitions"]["000"]["equals_captured_cpp_membership"])
                        self.assertTrue(softmax["coalitions"]["111"]["equals_captured_python_membership"])
                        for coalition in softmax["coalitions"].values():
                            self.assertTrue(coalition["residual_surface_finite"])
                            self.assertTrue(coalition["rmsnorm_output_finite"])
                            self.assertTrue(coalition["router_logits_finite"])
                            self.assertTrue(coalition["softmax_probabilities_finite"])
                        self.assertTrue(coalition["all_evaluated_stages_finite"])

    def test_metadata_native_boundary_contract_and_equal_contract_mode(self):
        disabled = PARITY.validate_capture_residual_profile(self._residual_profile())
        enabled = PARITY.validate_capture_residual_profile(self._residual_profile(True, True))
        self.assertEqual(disabled["metadata_native_cpp_residual_contract"], "cpp_float32_add")
        self.assertEqual(enabled["metadata_native_cpp_residual_contract"], "python_final_bf16_rne_add")
        self.assertFalse(enabled["native_residual_contracts_differ"])
        with self.assertRaises(ValueError): PARITY.validate_capture_residual_profile(self._residual_profile(False, True))
        report = self._equal_contract_residual_analysis()
        self.assertEqual(report["active_residual_factors"], ["block_input", "attention_output"])
        self.assertEqual((report["native_cpp_coalition_key"], report["native_python_coalition_key"]),
                         ("00", "11"))
        for target in report["python_targets"].values():
            self.assertTrue(target["native_cpp_residual_reconstruction"][0]["cpp_native_exact"])
            self.assertTrue(target["per_token_applicability"][0]["applicable_cpp_rmsnorm_references"])
            for reference in target["references"].values():
                for rows in reference["router_matmul_references"].values():
                    self.assertEqual(len(rows[0]["native_path_decompositions"]), 2)
                    for path in rows[0]["native_path_decompositions"]:
                        self.assertTrue(path["residual_surface"]["closure_exact_equality"])
                        self.assertTrue(path["fixed_cpp_rmsnorm_output"]["closure_exact_equality"])
                        self.assertTrue(path["evaluated_router_logits"]["closure_exact_equality"])
                    for softmax in rows[0]["softmax_references"].values():
                        self.assertEqual(set(softmax["native_factor_truth_table"]), {"00", "01", "10", "11"})
                        self.assertEqual(len(softmax["descriptive_alternate_contract_truth_table"]), 8)

    def test_residual_three_factor_prerequisite_and_invalid_equal_native_are_non_decisive(self):
        sides, weight, router_weight = self._residual_three_factor_sides()
        mismatch = PARITY.post_attention_residual_three_factor_decomposition(
            sides, "physical_block_10__", [1], weight, weight.copy(), 1e-6, 2e-6, router_weight,
            self._residual_profile())
        self.assertEqual(mismatch["status"], "analysis not decisive")
        self.assertFalse(mismatch["shared_component_prerequisite"]["shared_components_valid"])
        equal_but_invalid = copy.deepcopy(sides)
        for target in ("cpp", "default", "math"):
            equal_but_invalid[target]["physical_block_10__router_topk_indices"][0, 0, -1] = 383
        report = PARITY.post_attention_residual_three_factor_decomposition(
            equal_but_invalid, "physical_block_10__", [1], weight, weight.copy(), 1e-6, 1e-6,
            router_weight, self._residual_profile())
        for target in report["python_targets"].values():
            for reference in target["references"].values():
                for rows in reference["router_matmul_references"].values():
                    for softmax in rows[0]["softmax_references"].values():
                        if softmax["reference_applicable_to_token"]:
                            self.assertFalse(softmax["classification_decisive"])
                            self.assertEqual(softmax["classification"], "analysis not decisive")

    def test_native_python_endpoint_failure_is_preserved_and_non_causal(self):
        expected = "full Python operand coalition does not reproduce Python membership"
        for analysis, python_key in ((self._equal_contract_residual_analysis(True), "11"),
                                     (self._differing_contract_endpoint_failure_analysis(), "111")):
            self.assertEqual(analysis["native_python_coalition_key"], python_key)
            self.assertTrue(analysis["native_factor_truth_tables"])
            self.assertTrue(analysis["descriptive_alternate_contract_truth_tables"])
            for target in analysis["python_targets"].values():
                applicability = target["per_token_applicability"][0]
                self.assertTrue(applicability["applicable_reference_diagnostic_outcome_agreement"])
                self.assertFalse(applicability["applicable_reference_causal_classification_agreement"])
                for name in applicability["applicable_cpp_rmsnorm_references"]:
                    for rows in target["references"][name]["router_matmul_references"].values():
                        for report in rows[0]["softmax_references"].values():
                            self.assertTrue(report["structural_prerequisites_valid"])
                            self.assertFalse(report["native_python_endpoint_membership_reconstruction_valid"])
                            self.assertEqual(report["diagnostic_outcome"], expected)
                            self.assertIsNone(report["definitive_causal_classification"])
                            self.assertFalse(report["causal_classification_decisive"])
                            self.assertEqual(report["classification"], expected)
            summary = PARITY.primary_post_attention_residual_three_factor_summary([
                {"physical_block": 10, "analysis": analysis},
                {"physical_block": 12, "analysis": analysis}], 10)
            self.assertIsNone(summary["global_definitive_result"])
            self.assertEqual(summary["definitive_localization_factors"], [])
            self.assertEqual(summary["restrained_next_localization_targets"], [])
            self.assertTrue(summary["cross_rmsnorm_diagnostic_outcome_agreement"])
            self.assertFalse(summary["cross_rmsnorm_definitive_causal_agreement"])
            self.assertTrue(summary["cross_router_matmul_diagnostic_outcome_agreement"])
            self.assertFalse(summary["cross_router_matmul_definitive_causal_agreement"])
            self.assertTrue(summary["cross_router_softmax_diagnostic_outcome_agreement"])
            self.assertFalse(summary["cross_router_softmax_definitive_causal_agreement"])
            for result_name in ("primary_oracle_result", "sensitivity_control_result"):
                row = summary[result_name]["result"][1]
                self.assertEqual(row["diagnostic_outcome"], expected)
                self.assertIsNone(row["definitive_causal_classification"])
                self.assertTrue(row["target_token_structurally_valid"])
                self.assertFalse(row["native_python_endpoint_membership_reconstruction_valid"])
                self.assertFalse(row["target_token_causally_decisive"])
                self.assertEqual(row["non_decisive_reason"],
                                 "native Python operand endpoint does not reproduce captured Python routing membership "
                                 "under the fixed C++ downstream contract")

    def test_own_side_bias_reconstruction_is_separate_from_fixed_cpp_cross_evaluation(self):
        analysis = self._bias_provenance_analysis()
        for target in analysis["python_targets"].values():
            self.assertFalse(target["native_correction_bias_exact_equality"])
            self.assertGreater(target["native_correction_bias_maximum_absolute_difference"], 0)
            self.assertGreater(target["native_correction_bias_rms_difference"], 0)
            self.assertIn("candidates", target["native_correction_bias_dtype_grid_audit"])
            self.assertEqual(set(target["probability_bias_decomposition"]["tokens"][0]["coalitions"]),
                             {"native_left", "right_probabilities_only", "right_bias_only", "native_right"})
            applicability = target["per_token_applicability"][0]
            for name in applicability["applicable_cpp_rmsnorm_references"]:
                for rows in target["references"][name]["router_matmul_references"].values():
                    for report in rows[0]["softmax_references"].values():
                        self.assertTrue(report["native_cpp_capture_membership_reconstruction_valid"])
                        self.assertTrue(report["native_python_capture_membership_reconstruction_valid"])
                        self.assertTrue(report["native_capture_memberships_reconstruct"])
                        self.assertFalse(report[
                            "fixed_cpp_bias_applied_to_captured_python_probabilities_matches_python_membership"])
                        self.assertTrue(report["structural_prerequisites_valid"])
                        self.assertEqual(report["diagnostic_outcome"],
                                         "full Python operand coalition does not reproduce Python membership")
                        self.assertFalse(report["causal_classification_decisive"])
        invalid = self._bias_provenance_analysis(True)
        for target in invalid["python_targets"].values():
            applicability = target["per_token_applicability"][0]
            for name in applicability["applicable_cpp_rmsnorm_references"]:
                for rows in target["references"][name]["router_matmul_references"].values():
                    for report in rows[0]["softmax_references"].values():
                        self.assertFalse(report["native_python_capture_membership_reconstruction_valid"])
                        self.assertFalse(report["structural_prerequisites_valid"])
                        self.assertEqual(report["diagnostic_outcome"], "analysis not decisive")

    def test_equal_native_bias_provenance_is_exact(self):
        analysis = self._differing_contract_endpoint_failure_analysis()
        for target in analysis["python_targets"].values():
            self.assertTrue(target["native_correction_bias_exact_equality"])
            self.assertEqual(target["native_correction_bias_maximum_absolute_difference"], 0)
            self.assertEqual(target["native_correction_bias_rms_difference"], 0)

    def test_exact_native_path_closure_gates_primary_result(self):
        for analysis, expected_paths in ((self._equal_contract_residual_analysis(), 2),
                                         (self._differing_contract_endpoint_failure_analysis(), 6)):
            for target in analysis["python_targets"].values():
                for reference in target["references"].values():
                    for rows in reference["router_matmul_references"].values():
                        self.assertEqual(len(rows[0]["native_path_decompositions"]), expected_paths)
                        for path in rows[0]["native_path_decompositions"]:
                            for stage in ("residual_surface", "fixed_cpp_rmsnorm_output",
                                          "evaluated_router_logits"):
                                self.assertTrue(path[stage]["closure_exact_equality"])
            self.assertTrue(analysis["all_native_paths_close_exactly"])
        analysis = self._equal_contract_residual_analysis()
        target = analysis["python_targets"]["default"]
        applicable = target["per_token_applicability"][0]["applicable_cpp_rmsnorm_references"]
        for name in applicable:
            for rows in target["references"][name]["router_matmul_references"].values():
                for report in rows[0]["softmax_references"].values():
                    report["all_native_paths_close_exactly"] = False
                    report["structural_prerequisites_valid"] = False
                    report["causal_classification_decisive"] = False
                    report["classification_decisive"] = False
        summary = PARITY.primary_post_attention_residual_three_factor_summary([
            {"physical_block": 10, "analysis": analysis},
            {"physical_block": 12, "analysis": self._equal_contract_residual_analysis()}], 10)
        self.assertIsNone(summary["global_definitive_result"])
        self.assertEqual(summary["definitive_localization_factors"], [])
        self.assertEqual(summary["python_reference_results"]["default"][1]["non_decisive_reason"],
                         "structural prerequisites or exact native path closure failed")

    def test_residual_primary_summary_rejects_missing_and_cross_python_mismatch(self):
        sides, weight, router_weight = self._residual_three_factor_sides()
        analysis = PARITY.post_attention_residual_three_factor_decomposition(
            sides, "physical_block_10__", [1], weight, weight.copy(), 1e-6, 1e-6, router_weight,
            self._residual_profile())
        positive = PARITY.primary_post_attention_residual_three_factor_summary([
            {"physical_block": 10, "analysis": analysis},
            {"physical_block": 12, "analysis": analysis}], 10)
        self.assertTrue(positive["physical_block_12_remains_aligned_control"])
        missing = copy.deepcopy(analysis)
        missing["python_targets"]["default"]["per_token_applicability"][0][
            "applicable_cpp_rmsnorm_references"] = []
        summary = PARITY.primary_post_attention_residual_three_factor_summary([
            {"physical_block": 10, "analysis": missing}, {"physical_block": 12, "analysis": analysis}], 10)
        self.assertFalse(summary["all_applicable_results_decisive"])
        self.assertEqual(summary["tokens_without_applicable_cpp_rmsnorm_reference"],
                         [{"target": "default", "attended_token": 1}])
        self.assertEqual(summary["restrained_next_localization_targets"], [])
        disagreement = copy.deepcopy(analysis)
        applicable = disagreement["python_targets"]["math"]["per_token_applicability"][0][
            "applicable_cpp_rmsnorm_references"]
        for name in applicable:
            for rows in disagreement["python_targets"]["math"]["references"][name][
                    "router_matmul_references"].values():
                for softmax in rows[0]["softmax_references"].values():
                    softmax["minimal_sufficient_factor_sets"] = [["attention_output"]]
        summary = PARITY.primary_post_attention_residual_three_factor_summary([
            {"physical_block": 10, "analysis": disagreement},
            {"physical_block": 12, "analysis": analysis}], 10)
        self.assertFalse(summary["cross_python_minimal_factor_set_agreement"])
        self.assertFalse(summary["cross_python_reference_agreement"])
        self.assertEqual(summary["definitive_localization_factors"], [])

        def control_flag(mutator):
            control = copy.deepcopy(analysis)
            mutator(control)
            return PARITY.primary_post_attention_residual_three_factor_summary([
                {"physical_block": 10, "analysis": analysis},
                {"physical_block": 12, "analysis": control}], 10)[
                    "physical_block_12_remains_aligned_control"]
        self.assertFalse(control_flag(lambda control: control["python_targets"]["default"][
            "per_token_applicability"][0].update({"applicable_cpp_rmsnorm_references": []})))

        def mutate_report(control, operation):
            applicability = control["python_targets"]["default"]["per_token_applicability"][0]
            name = applicability["applicable_cpp_rmsnorm_references"][0]
            rows = next(iter(control["python_targets"]["default"]["references"][name][
                "router_matmul_references"].values()))
            operation(next(iter(rows[0]["softmax_references"].values())))
        self.assertFalse(control_flag(lambda control: mutate_report(control,
                                                                    lambda report: report["coalitions"]["111"].update({"equals_captured_python_membership": False}))))
        self.assertFalse(control_flag(lambda control: mutate_report(control,
                                                                    lambda report: report.update({"captured_topk_membership_reconstruction_valid": False}))))
        self.assertFalse(control_flag(lambda control: control["python_targets"]["default"][
            "native_residual_add_reconstruction"][0].update({"python_native_exact": False})))
        self.assertTrue(control_flag(lambda control: control["python_targets"]["default"][
            "per_token_applicability"][0].update({"applicable_reference_classification_agreement": False})))

    def test_equal_contract_block12_uses_dynamic_native_keys(self):
        analysis = self._equal_contract_residual_analysis()
        self.assertEqual((analysis["native_cpp_coalition_key"], analysis["native_python_coalition_key"]),
                         ("00", "11"))

        def control_flag(control):
            return PARITY.primary_post_attention_residual_three_factor_summary([
                {"physical_block": 10, "analysis": analysis},
                {"physical_block": 12, "analysis": control}], 10)[
                    "physical_block_12_remains_aligned_control"]
        positive_summary = PARITY.primary_post_attention_residual_three_factor_summary([
            {"physical_block": 10, "analysis": analysis},
            {"physical_block": 12, "analysis": copy.deepcopy(analysis)}], 10)
        self.assertTrue(positive_summary["physical_block_12_remains_aligned_control"])
        self.assertEqual(positive_summary["physical_block_12_control_failure_reasons"], [])
        self.assertTrue(positive_summary["physical_block_12_per_target_token_control_status"])
        self.assertTrue(all(row["final_control_status"] for row in
                            positive_summary["physical_block_12_per_target_token_control_status"]))
        for key, field in (("00", "equals_captured_cpp_membership"),
                           ("11", "equals_captured_python_membership")):
            control = copy.deepcopy(analysis)
            for target in control["python_targets"].values():
                applicability = target["per_token_applicability"][0]
                name = applicability["applicable_cpp_rmsnorm_references"][0]
                rows = next(iter(target["references"][name]["router_matmul_references"].values()))
                next(iter(rows[0]["softmax_references"].values()))["coalitions"][key][field] = False
                break
            negative = PARITY.primary_post_attention_residual_three_factor_summary([
                {"physical_block": 10, "analysis": analysis},
                {"physical_block": 12, "analysis": control}], 10)
            self.assertFalse(negative["physical_block_12_remains_aligned_control"])
            self.assertTrue(negative["physical_block_12_control_failure_reasons"])
        for field, expected_reason in (
                ("native_python_capture_membership_reconstruction_valid",
                 "native Python capture membership reconstruction failed"),
                ("fixed_cpp_downstream_python_operand_endpoint_matches_python_membership",
                 "fixed-C++ Python operand endpoint reconstruction failed")):
            control = copy.deepcopy(analysis)
            target = control["python_targets"]["default"]
            name = target["per_token_applicability"][0]["applicable_cpp_rmsnorm_references"][0]
            rows = next(iter(target["references"][name]["router_matmul_references"].values()))
            next(iter(rows[0]["softmax_references"].values()))[field] = False
            negative = PARITY.primary_post_attention_residual_three_factor_summary([
                {"physical_block": 10, "analysis": analysis},
                {"physical_block": 12, "analysis": control}], 10)
            self.assertFalse(negative["physical_block_12_remains_aligned_control"])
            self.assertTrue(any(expected_reason in reason for reason in
                                negative["physical_block_12_control_failure_reasons"]))
        stale = copy.deepcopy(analysis)
        target = stale["python_targets"]["default"]
        name = target["per_token_applicability"][0]["applicable_cpp_rmsnorm_references"][0]
        rows = next(iter(target["references"][name]["router_matmul_references"].values()))
        next(iter(rows[0]["softmax_references"].values()))["minimal_sufficient_factor_sets"] = [
            ["block_input"]]
        stale_summary = PARITY.primary_post_attention_residual_three_factor_summary([
            {"physical_block": 10, "analysis": analysis},
            {"physical_block": 12, "analysis": stale}], 10)
        self.assertFalse(stale_summary["physical_block_12_remains_aligned_control"])
        self.assertTrue(any("minimal sufficient factor sets disagree" in reason for reason in
                            stale_summary["physical_block_12_control_failure_reasons"]))

    def test_primary_oracle_local_factors_survive_non_global_sensitivity_disagreement(self):
        control = self._equal_contract_residual_analysis()
        analysis = copy.deepcopy(control)
        for target_name, classification, factors in (
                ("default", "block-input component sufficient", [["block_input"]]),
                ("math", "attention-output component sufficient", [["attention_output"]])):
            target = analysis["python_targets"][target_name]
            for applicability in target["per_token_applicability"]:
                for name in applicability["applicable_cpp_rmsnorm_references"]:
                    for rows in target["references"][name]["router_matmul_references"].values():
                        for report in rows[0]["softmax_references"].values():
                            report.update({"diagnostic_outcome": classification,
                                           "classification": classification,
                                           "definitive_causal_classification": classification,
                                           "minimal_sufficient_factor_sets": factors,
                                           "structural_prerequisites_valid": True,
                                           "native_python_endpoint_membership_reconstruction_valid": True,
                                           "causal_classification_decisive": True,
                                           "classification_decisive": True})
        summary = PARITY.primary_post_attention_residual_three_factor_summary([
            {"physical_block": 10, "analysis": analysis},
            {"physical_block": 12, "analysis": control}], 10)
        self.assertFalse(summary["cross_python_reference_agreement"])
        self.assertEqual(summary["primary_oracle_localization_factors"], ["block_input"])
        self.assertTrue(summary["primary_oracle_next_localization_targets"])
        self.assertFalse(summary["primary_oracle_is_global"])
        self.assertEqual(summary["sensitivity_control_localization_factors"], ["attention_output"])
        self.assertFalse(summary["sensitivity_control_is_global"])
        self.assertIsNone(summary["global_definitive_result"])
        self.assertEqual(summary["definitive_localization_factors"], [])
        self.assertEqual(summary["restrained_next_localization_targets"], [])

    def test_block12_requires_exact_nonempty_default_and_math_targets(self):
        pristine = self._equal_contract_residual_analysis()
        for missing_target in ("default", "math"):
            control = copy.deepcopy(pristine)
            control["python_targets"].pop(missing_target)
            summary = PARITY.primary_post_attention_residual_three_factor_summary([
                {"physical_block": 10, "analysis": pristine},
                {"physical_block": 12, "analysis": control}], 10)
            self.assertFalse(summary["physical_block_12_remains_aligned_control"])
            self.assertFalse(summary["block_12_python_target_inventory_complete"])
            self.assertEqual(summary["missing_block_12_python_targets"], [missing_target])
            status = next(row for row in summary["physical_block_12_per_target_token_control_status"]
                          if row["python_target"] == missing_target)
            self.assertIsNone(status["attended_token"])
            self.assertIn("required block-12 Python target is missing", status["failure_reasons"])
            self.assertIsNone(summary["global_definitive_result"])
        for empty_target in ("default", "math"):
            control = copy.deepcopy(pristine)
            control["python_targets"][empty_target] = {}
            summary = PARITY.primary_post_attention_residual_three_factor_summary([
                {"physical_block": 10, "analysis": pristine},
                {"physical_block": 12, "analysis": control}], 10)
            self.assertFalse(summary["physical_block_12_remains_aligned_control"])
            status = next(row for row in summary["physical_block_12_per_target_token_control_status"]
                          if row["python_target"] == empty_target)
            self.assertIsNone(status["attended_token"])
            self.assertFalse(status["attended_token_inventory_complete"])
        control = copy.deepcopy(pristine)
        control["python_targets"]["unexpected"] = control["python_targets"].pop("math")
        summary = PARITY.primary_post_attention_residual_three_factor_summary([
            {"physical_block": 10, "analysis": pristine}, {"physical_block": 12, "analysis": control}], 10)
        self.assertEqual(summary["missing_block_12_python_targets"], ["math"])
        self.assertEqual(summary["unexpected_block_12_python_targets"], ["unexpected"])
        self.assertFalse(summary["physical_block_12_remains_aligned_control"])
        control = copy.deepcopy(pristine)
        control["python_targets"]["unexpected"] = copy.deepcopy(control["python_targets"]["default"])
        summary = PARITY.primary_post_attention_residual_three_factor_summary([
            {"physical_block": 10, "analysis": pristine}, {"physical_block": 12, "analysis": control}], 10)
        self.assertFalse(summary["block_12_python_target_inventory_complete"])
        self.assertFalse(summary["physical_block_12_remains_aligned_control"])

    def test_canonical_minimal_factor_sets_ignore_order_but_preserve_coalitions(self):
        left = [["block_input"], ["attention_output", "residual_add_contract"]]
        reordered = [["residual_add_contract", "attention_output"], ["block_input"]]
        different = [["block_input"], ["block_input", "attention_output"]]
        self.assertEqual(PARITY.canonical_minimal_factor_sets(left),
                         PARITY.canonical_minimal_factor_sets(reordered))
        self.assertNotEqual(PARITY.canonical_minimal_factor_sets(left),
                            PARITY.canonical_minimal_factor_sets(different))

    def test_missing_applicable_reference_report_suppresses_local_causality(self):
        control = self._equal_contract_residual_analysis()
        analysis = copy.deepcopy(control)
        target = analysis["python_targets"]["default"]
        name = target["per_token_applicability"][0]["applicable_cpp_rmsnorm_references"][0]
        rows = next(iter(target["references"][name]["router_matmul_references"].values()))
        rows[0]["softmax_references"].pop(next(iter(rows[0]["softmax_references"])))
        summary = PARITY.primary_post_attention_residual_three_factor_summary([
            {"physical_block": 10, "analysis": analysis},
            {"physical_block": 12, "analysis": control}], 10)
        row = summary["python_reference_results"]["default"][1]
        self.assertFalse(row["applicable_reference_inventory_complete"])
        self.assertFalse(row["target_token_causally_decisive"])
        self.assertEqual(row["non_decisive_reason"], "applicable reference evidence is incomplete")
        self.assertFalse(summary["primary_oracle_locally_decisive"])
        self.assertEqual(summary["primary_oracle_localization_factors"], [])

    def test_exact_reference_inventory_rejects_missing_unexpected_and_duplicate_coordinates(self):
        pristine = self._equal_contract_residual_analysis()
        for target in pristine["python_targets"].values():
            inventory = PARITY.collect_applicable_residual_reference_reports(target, 1)
            self.assertTrue(inventory["inventory_complete"])
            self.assertEqual(inventory["missing_coordinates"], [])
            self.assertEqual(inventory["unexpected_coordinates"], [])
            self.assertEqual(inventory["duplicate_coordinates"], [])

        def coordinates(target):
            applicability = target["per_token_applicability"][0]
            rmsnorm = applicability["applicable_cpp_rmsnorm_references"][0]
            matmuls = target["references"][rmsnorm]["router_matmul_references"]
            matmul = next(iter(matmuls))
            rows = matmuls[matmul]
            row = next(item for item in rows if item["attended_token"] == 1)
            softmax = next(iter(row["softmax_references"]))
            return applicability, rmsnorm, matmuls, matmul, rows, row, softmax

        def missing_rmsnorm(target):
            _, rmsnorm, *_ = coordinates(target)
            target["references"].pop(rmsnorm)

        def missing_matmul(target):
            _, _, matmuls, matmul, *_ = coordinates(target)
            matmuls.pop(matmul)

        def missing_softmax(target):
            *_, row, softmax = coordinates(target)
            row["softmax_references"].pop(softmax)

        def missing_row(target):
            _, _, _, _, rows, _, _ = coordinates(target)
            rows[:] = []

        def duplicate_row(target):
            _, _, _, _, rows, row, _ = coordinates(target)
            rows.append(copy.deepcopy(row))

        def replace_matmul(target):
            _, _, matmuls, matmul, *_ = coordinates(target)
            matmuls["unexpected_matmul"] = matmuls.pop(matmul)

        def replace_softmax(target):
            *_, row, softmax = coordinates(target)
            row["softmax_references"]["unexpected_softmax"] = row["softmax_references"].pop(softmax)

        def unknown_rmsnorm(target):
            applicability, *_ = coordinates(target)
            applicability["applicable_cpp_rmsnorm_references"].append("unknown_rmsnorm")

        def populated_unknown_rmsnorm(target):
            applicability, rmsnorm, *_ = coordinates(target)
            target["references"]["unknown_rmsnorm"] = copy.deepcopy(target["references"][rmsnorm])
            applicability["applicable_cpp_rmsnorm_references"].append("unknown_rmsnorm")

        def replace_with_unknown_rmsnorm(target):
            applicability, rmsnorm, *_ = coordinates(target)
            target["references"]["unknown_rmsnorm"] = target["references"].pop(rmsnorm)
            applicability["applicable_cpp_rmsnorm_references"] = [
                "unknown_rmsnorm" if name == rmsnorm else name
                for name in applicability["applicable_cpp_rmsnorm_references"]]

        def duplicate_rmsnorm(target):
            applicability, rmsnorm, *_ = coordinates(target)
            applicability["applicable_cpp_rmsnorm_references"].append(rmsnorm)

        def extra_matmul(target):
            _, _, matmuls, matmul, *_ = coordinates(target)
            matmuls["unexpected_matmul"] = copy.deepcopy(matmuls[matmul])

        def extra_softmax(target):
            *_, row, softmax = coordinates(target)
            row["softmax_references"]["unexpected_softmax"] = copy.deepcopy(
                row["softmax_references"][softmax])

        cases = ((missing_rmsnorm, "missing_rmsnorm_references"),
                 (missing_matmul, "missing_coordinates"),
                 (missing_softmax, "missing_coordinates"),
                 (missing_row, "missing_token_rows"),
                 (duplicate_row, "duplicate_token_rows"),
                 (replace_matmul, "unexpected_coordinates"),
                 (replace_softmax, "unexpected_coordinates"),
                 (unknown_rmsnorm, "unknown_applicable_rmsnorm_references"),
                 (populated_unknown_rmsnorm, "unknown_applicable_rmsnorm_references"),
                 (replace_with_unknown_rmsnorm, "unknown_applicable_rmsnorm_references"),
                 (duplicate_rmsnorm, "duplicate_applicable_rmsnorm_references"),
                 (extra_matmul, "unexpected_coordinates"),
                 (extra_softmax, "unexpected_coordinates"))
        for mutate, expected_field in cases:
            malformed = copy.deepcopy(pristine)
            for target in malformed["python_targets"].values(): mutate(target)
            for target in malformed["python_targets"].values():
                inventory = PARITY.collect_applicable_residual_reference_reports(target, 1)
                self.assertFalse(inventory["inventory_complete"], mutate.__name__)
                self.assertTrue(inventory[expected_field], mutate.__name__)
                if mutate in (populated_unknown_rmsnorm, replace_with_unknown_rmsnorm):
                    self.assertIn("unknown_rmsnorm", inventory["unexpected_rmsnorm_reference_keys"])
            summary = PARITY.primary_post_attention_residual_three_factor_summary([
                {"physical_block": 10, "analysis": malformed},
                {"physical_block": 12, "analysis": malformed}], 10)
            for target_name in ("default", "math"):
                row = summary["python_reference_results"][target_name][1]
                self.assertFalse(row["applicable_reference_inventory_complete"], mutate.__name__)
                self.assertFalse(row["target_token_causally_decisive"], mutate.__name__)
                self.assertIsNone(row["definitive_causal_classification"])
                self.assertEqual(row["minimal_sufficient_factor_sets"], [])
                self.assertEqual(row["non_decisive_reason"],
                                 "applicable reference evidence is incomplete")
            self.assertFalse(summary["primary_oracle_locally_decisive"])
            self.assertFalse(summary["sensitivity_control_locally_decisive"])
            self.assertEqual(summary["primary_oracle_localization_factors"], [])
            self.assertEqual(summary["sensitivity_control_localization_factors"], [])
            self.assertIsNone(summary["global_definitive_result"])
            self.assertFalse(summary["physical_block_12_remains_aligned_control"])
            self.assertTrue(summary["physical_block_12_control_failure_reasons"])
            self.assertTrue(any(not status["applicable_reference_inventory_complete"] for status in
                                summary["physical_block_12_per_target_token_control_status"]))

    def test_same_broad_classification_with_different_minimal_sets_is_not_locally_decisive(self):
        control = self._equal_contract_residual_analysis()
        analysis = copy.deepcopy(control)
        target = analysis["python_targets"]["default"]
        reports = []
        for applicability in target["per_token_applicability"]:
            for name in applicability["applicable_cpp_rmsnorm_references"]:
                for rows in target["references"][name]["router_matmul_references"].values():
                    reports.extend(rows[0]["softmax_references"].values())
        for report in reports:
            report.update({"diagnostic_outcome": "multiple single components independently sufficient",
                           "classification": "multiple single components independently sufficient",
                           "definitive_causal_classification": "multiple single components independently sufficient",
                           "minimal_sufficient_factor_sets": [["block_input"], ["attention_output"]],
                           "structural_prerequisites_valid": True,
                           "native_python_endpoint_membership_reconstruction_valid": True,
                           "causal_classification_decisive": True,
                           "classification_decisive": True})
        reports[-1]["minimal_sufficient_factor_sets"] = [
            ["block_input"], ["residual_add_contract"]]
        summary = PARITY.primary_post_attention_residual_three_factor_summary([
            {"physical_block": 10, "analysis": analysis},
            {"physical_block": 12, "analysis": control}], 10)
        row = summary["python_reference_results"]["default"][1]
        self.assertTrue(row["applicable_reference_definitive_classification_agreement"])
        self.assertFalse(row["applicable_reference_minimal_factor_set_agreement"])
        self.assertFalse(row["applicable_reference_definitive_causal_evidence_agreement"])
        self.assertFalse(row["target_token_causally_decisive"])
        self.assertEqual(row["diagnostic_outcome"], "analysis not decisive")
        self.assertIsNone(row["definitive_causal_classification"])
        self.assertEqual(row["minimal_sufficient_factor_sets"], [])
        self.assertEqual(row["non_decisive_reason"],
                         "applicable minimal sufficient factor sets disagree")
        self.assertTrue(row["descriptive_per_reference_minimal_sufficient_factor_sets"])
        self.assertFalse(summary["primary_oracle_locally_decisive"])
        self.assertEqual(summary["primary_oracle_localization_factors"], [])
        self.assertEqual(summary["primary_oracle_next_localization_targets"], [])
        self.assertIsNone(summary["global_definitive_result"])
        self.assertEqual(summary["definitive_localization_factors"], [])
        math_reports = []
        math_target = analysis["python_targets"]["math"]
        for applicability in math_target["per_token_applicability"]:
            for name in applicability["applicable_cpp_rmsnorm_references"]:
                for rows in math_target["references"][name]["router_matmul_references"].values():
                    math_reports.extend(rows[0]["softmax_references"].values())
        for report in math_reports:
            report.update({"diagnostic_outcome": "multiple minimal sufficient coalitions",
                           "classification": "multiple minimal sufficient coalitions",
                           "definitive_causal_classification": "multiple minimal sufficient coalitions",
                           "minimal_sufficient_factor_sets": [["block_input", "attention_output"],
                                                              ["block_input", "residual_add_contract"]],
                           "structural_prerequisites_valid": True,
                           "native_python_endpoint_membership_reconstruction_valid": True,
                           "causal_classification_decisive": True,
                           "classification_decisive": True})
        math_reports[-1]["minimal_sufficient_factor_sets"] = [
            ["block_input", "attention_output"], ["attention_output", "residual_add_contract"]]
        summary = PARITY.primary_post_attention_residual_three_factor_summary([
            {"physical_block": 10, "analysis": analysis},
            {"physical_block": 12, "analysis": control}], 10)
        self.assertFalse(summary["sensitivity_control_locally_decisive"])
        self.assertEqual(summary["sensitivity_control_localization_factors"], [])
        self.assertEqual(summary["sensitivity_control_next_localization_targets"], [])

    def test_attended_token_inventory_cannot_hide_missing_duplicate_or_unexpected_tokens(self):
        pristine = self._equal_contract_residual_analysis()

        def remove_applicability(target): target["per_token_applicability"] = []

        def duplicate_applicability(target):
            target["per_token_applicability"].append(copy.deepcopy(target["per_token_applicability"][0]))

        def unexpected_applicability(target):
            row = copy.deepcopy(target["per_token_applicability"][0])
            row["attended_token"] = 99
            target["per_token_applicability"].append(row)

        def remove_native(target): target["native_residual_add_reconstruction"] = []

        def duplicate_native(target):
            target["native_residual_add_reconstruction"].append(
                copy.deepcopy(target["native_residual_add_reconstruction"][0]))

        # Removing either oracle's whole row is visible and cannot disappear globally.
        for target_name in ("default", "math"):
            malformed = copy.deepcopy(pristine)
            remove_applicability(malformed["python_targets"][target_name])
            summary = PARITY.primary_post_attention_residual_three_factor_summary([
                {"physical_block": 10, "analysis": malformed},
                {"physical_block": 12, "analysis": pristine}], 10)
            self.assertIn(1, summary["python_reference_results"][target_name])
            row = summary["python_reference_results"][target_name][1]
            self.assertFalse(row["attended_token_inventory_complete"])
            self.assertEqual(row["missing_applicability_attended_tokens"], [1])
            self.assertFalse(row["target_token_causally_decisive"])
            self.assertFalse(row["target_token_structurally_valid"])
            self.assertEqual(row["non_decisive_reason"], "applicable reference evidence is incomplete")
            self.assertIsNone(summary["global_definitive_result"])

    def test_report_schema_validation_excludes_malformed_safe_reports(self):
        pristine = self._equal_contract_residual_analysis()

        def mutate_reports(analysis, operation):
            for target in analysis["python_targets"].values():
                name = target["per_token_applicability"][0]["applicable_cpp_rmsnorm_references"][0]
                rows = next(iter(target["references"][name]["router_matmul_references"].values()))
                operation(next(iter(rows[0]["softmax_references"].values())))
        cases = []
        for field in ("diagnostic_outcome", "causal_classification_decisive",
                      "definitive_causal_classification", "minimal_sufficient_factor_sets",
                      "structural_prerequisites_valid",
                      "native_python_endpoint_membership_reconstruction_valid"):
            cases.append((f"missing_{field}", lambda report, field=field: report.pop(field)))
        cases.extend((("unhashable_classification", lambda report: report.update(
            {"definitive_causal_classification": ["malformed"]})),
            ("malformed_factor_container", lambda report: report.update(
                {"minimal_sufficient_factor_sets": {"block_input"}})),
            ("malformed_factor_entry", lambda report: report.update(
                {"minimal_sufficient_factor_sets": [["block_input", 7]]}))))
        for label, operation in cases:
            malformed = copy.deepcopy(pristine)
            mutate_reports(malformed, operation)
            summary = PARITY.primary_post_attention_residual_three_factor_summary([
                {"physical_block": 10, "analysis": malformed},
                {"physical_block": 12, "analysis": malformed}], 10)
            for target_name in ("default", "math"):
                row = summary["python_reference_results"][target_name][1]
                self.assertFalse(row["applicable_reference_inventory_complete"], label)
                self.assertEqual(row["non_decisive_reason"],
                                 "applicable reference evidence is incomplete")
                self.assertTrue(row["malformed_inventory_paths"])
            self.assertFalse(summary["cross_rmsnorm_diagnostic_outcome_agreement"])
            self.assertFalse(summary["cross_router_matmul_diagnostic_outcome_agreement"])
            self.assertFalse(summary["cross_router_softmax_diagnostic_outcome_agreement"])
            self.assertFalse(summary["physical_block_12_remains_aligned_control"])
            self.assertIsNone(summary["global_definitive_result"])

    def test_native_residual_token_identities_are_safely_normalized(self):
        pristine = self._equal_contract_residual_analysis()
        mutations = (
            lambda row: row.update({"attended_token": [1]}),
            lambda row: row.update({"attended_token": "1"}),
            lambda row: row.update({"attended_token": None}),
            lambda row: row.pop("attended_token"),
            lambda row: row.update({"attended_token": 99}),
        )
        for mutate in mutations:
            malformed = copy.deepcopy(pristine)
            for target in malformed["python_targets"].values():
                mutate(target["native_residual_add_reconstruction"][0])
            summary = PARITY.primary_post_attention_residual_three_factor_summary([
                {"physical_block": 10, "analysis": malformed},
                {"physical_block": 12, "analysis": malformed}], 10)
            self.assertFalse(summary["physical_block_12_remains_aligned_control"])
            self.assertIsNone(summary["global_definitive_result"])
            self.assertTrue(all(not inventory["attended_token_inventory_complete"] for inventory in
                                summary["target_attended_token_inventories"].values()))

    def test_malformed_reference_containers_are_non_throwing_inventory_failures(self):
        pristine = self._equal_contract_residual_analysis()
        def remove_applicability(target): target["per_token_applicability"] = []

        def duplicate_applicability(target):
            target["per_token_applicability"].append(copy.deepcopy(target["per_token_applicability"][0]))

        def unexpected_applicability(target):
            row = copy.deepcopy(target["per_token_applicability"][0])
            row["attended_token"] = 99
            target["per_token_applicability"].append(row)

        def remove_native(target): target["native_residual_add_reconstruction"] = []

        def duplicate_native(target):
            target["native_residual_add_reconstruction"].append(
                copy.deepcopy(target["native_residual_add_reconstruction"][0]))

        def parts(target):
            applicability = target["per_token_applicability"][0]
            rmsnorm = applicability["applicable_cpp_rmsnorm_references"][0]
            reference = target["references"][rmsnorm]
            matmul = next(iter(reference["router_matmul_references"]))
            rows = reference["router_matmul_references"][matmul]
            row = rows[0]
            softmax = next(iter(row["softmax_references"]))
            return applicability, rmsnorm, reference, matmul, rows, row, softmax

        def applicability_container(target): target["per_token_applicability"] = "malformed"
        def applicability_row(target): target["per_token_applicability"] = ["malformed"]
        def applicable_rmsnorm_value(target): parts(target)[0]["applicable_cpp_rmsnorm_references"] = "malformed"
        def references_container(target): target["references"] = "malformed"
        def matmul_map(target): parts(target)[2]["router_matmul_references"] = "malformed"

        def row_container(target):
            _, _, reference, matmul, *_ = parts(target)
            reference["router_matmul_references"][matmul] = "malformed"

        def softmax_map(target): parts(target)[5]["softmax_references"] = "malformed"

        def report_value(target):
            *_, row, softmax = parts(target)
            row["softmax_references"][softmax] = "malformed"
        for mutate in (applicability_container, applicability_row, applicable_rmsnorm_value,
                       references_container, matmul_map, row_container, softmax_map, report_value):
            malformed = copy.deepcopy(pristine)
            for target in malformed["python_targets"].values(): mutate(target)
            summary = PARITY.primary_post_attention_residual_three_factor_summary([
                {"physical_block": 10, "analysis": malformed},
                {"physical_block": 12, "analysis": malformed}], 10)
            self.assertIsNone(summary["global_definitive_result"], mutate.__name__)
            self.assertFalse(summary["physical_block_12_remains_aligned_control"], mutate.__name__)
            self.assertTrue(summary["physical_block_12_control_failure_reasons"])
            self.assertFalse(summary["cross_rmsnorm_diagnostic_outcome_agreement"])
            self.assertFalse(summary["cross_router_matmul_diagnostic_outcome_agreement"])
            self.assertFalse(summary["cross_router_softmax_diagnostic_outcome_agreement"])

        for selected_analysis in ("malformed", {"python_targets": "malformed"},
                                  {"python_targets": {"default": "malformed", "math": "malformed"}}):
            summary = PARITY.primary_post_attention_residual_three_factor_summary([
                {"physical_block": 10, "analysis": selected_analysis},
                {"physical_block": 12, "analysis": pristine}], 10)
            self.assertIsNone(summary["global_definitive_result"])
            self.assertTrue(summary["malformed_primary_summary_paths"]
                            or summary["target_attended_token_inventories"])

        malformed_control = copy.deepcopy(pristine)
        malformed_control["python_targets"]["default"] = "malformed"
        summary = PARITY.primary_post_attention_residual_three_factor_summary([
            {"physical_block": 10, "analysis": pristine},
            {"physical_block": 12, "analysis": malformed_control}], 10)
        self.assertFalse(summary["physical_block_12_remains_aligned_control"])
        status = next(row for row in summary["physical_block_12_per_target_token_control_status"]
                      if row["python_target"] == "default")
        self.assertIsNone(status["attended_token"])
        self.assertIn("block-12 target report is malformed", status["failure_reasons"])

        for mutation in ("coalitions", "native_cpp_coalition_key", "native_python_coalition_key"):
            malformed_control = copy.deepcopy(pristine)
            target = malformed_control["python_targets"]["default"]
            if mutation == "coalitions":
                name = target["per_token_applicability"][0]["applicable_cpp_rmsnorm_references"][0]
                rows = next(iter(target["references"][name]["router_matmul_references"].values()))
                next(iter(rows[0]["softmax_references"].values())).pop("coalitions")
            else: target.pop(mutation)
            summary = PARITY.primary_post_attention_residual_three_factor_summary([
                {"physical_block": 10, "analysis": pristine},
                {"physical_block": 12, "analysis": malformed_control}], 10)
            self.assertFalse(summary["physical_block_12_remains_aligned_control"], mutation)

        malformed = copy.deepcopy(pristine)
        for target in malformed["python_targets"].values(): remove_applicability(target)
        summary = PARITY.primary_post_attention_residual_three_factor_summary([
            {"physical_block": 10, "analysis": malformed},
            {"physical_block": 12, "analysis": malformed}], 10)
        self.assertIn(1, summary["python_reference_results"]["default"])
        self.assertIn(1, summary["python_reference_results"]["math"])
        self.assertFalse(summary["primary_oracle_locally_decisive"])
        self.assertFalse(summary["sensitivity_control_locally_decisive"])
        self.assertEqual(summary["primary_oracle_localization_factors"], [])
        self.assertEqual(summary["sensitivity_control_localization_factors"], [])
        self.assertFalse(summary["physical_block_12_remains_aligned_control"])
        statuses = summary["physical_block_12_per_target_token_control_status"]
        self.assertEqual({status["attended_token"] for status in statuses}, {1})
        self.assertTrue(all("applicable token inventory is incomplete" in status["failure_reasons"]
                            and "applicable reference evidence is incomplete" in status["failure_reasons"]
                            for status in statuses))

        for mutate, field in ((duplicate_applicability, "duplicate_applicability_attended_tokens"),
                              (unexpected_applicability, "unexpected_applicability_attended_tokens"),
                              (remove_native, "missing_native_residual_attended_tokens"),
                              (duplicate_native, "duplicate_native_residual_attended_tokens")):
            malformed = copy.deepcopy(pristine)
            for target in malformed["python_targets"].values(): mutate(target)
            summary = PARITY.primary_post_attention_residual_three_factor_summary([
                {"physical_block": 10, "analysis": malformed},
                {"physical_block": 12, "analysis": malformed}], 10)
            for target_name in ("default", "math"):
                inventory = summary["target_attended_token_inventories"][target_name]
                self.assertFalse(inventory["attended_token_inventory_complete"], mutate.__name__)
                self.assertTrue(inventory[field], mutate.__name__)
            self.assertFalse(summary["primary_oracle_locally_decisive"])
            self.assertFalse(summary["sensitivity_control_locally_decisive"])
            self.assertIsNone(summary["global_definitive_result"])

    def test_residual_three_factor_stage_finiteness_failures_are_non_decisive(self):
        sides, weight, router_weight = self._residual_three_factor_sides()
        cases = []
        residual_bad = copy.deepcopy(sides)
        residual_bad["default"]["physical_block_10__attention_output"][0, 0, 0] = np.nan
        cases.append((residual_bad, "residual_surface_finite", None))
        logits_bad = copy.deepcopy(sides)
        logits_bad["cpp"]["physical_block_10__router_logits"][0, 0, 0] = np.nan
        cases.append((logits_bad, "router_logits_finite", None))
        original_norm = PARITY.diagnostic_rmsnorm_with_operational_weight
        original_softmax = dict(PARITY.DIAGNOSTIC_SOFTMAX_REFERENCES)

        def nan_norm(value, *args, **kwargs):
            result, metadata = original_norm(value, *args, **kwargs)
            result = result.copy()
            result[..., 0] = np.nan
            return result, metadata

        def nan_softmax(value): return np.full_like(np.asarray(value, np.float32), np.nan)
        try:
            for source, field, patch_stage in cases + [
                    (copy.deepcopy(sides), "rmsnorm_output_finite", "norm"),
                    (copy.deepcopy(sides), "softmax_probabilities_finite", "softmax")]:
                PARITY.diagnostic_rmsnorm_with_operational_weight = nan_norm if patch_stage == "norm" else original_norm
                PARITY.DIAGNOSTIC_SOFTMAX_REFERENCES.clear()
                PARITY.DIAGNOSTIC_SOFTMAX_REFERENCES.update(original_softmax)
                if patch_stage == "softmax":
                    PARITY.DIAGNOSTIC_SOFTMAX_REFERENCES["stable_float32"] = nan_softmax
                report = PARITY.post_attention_residual_three_factor_decomposition(
                    source, "physical_block_10__", [1], weight, weight.copy(), 1e-6, 1e-6,
                    router_weight, self._residual_profile())
                target = report["python_targets"]["default"]
                evidence_rows = [softmax for reference in target["references"].values()
                                 for rows in reference["router_matmul_references"].values()
                                 for softmax in rows[0]["softmax_references"].values()]
                self.assertTrue(any(not coalition[field] for evidence in evidence_rows
                                    for coalition in evidence["coalitions"].values()))
                self.assertTrue(any(not evidence["classification_decisive"] for evidence in evidence_rows))
        finally:
            PARITY.diagnostic_rmsnorm_with_operational_weight = original_norm
            PARITY.DIAGNOSTIC_SOFTMAX_REFERENCES.clear()
            PARITY.DIAGNOSTIC_SOFTMAX_REFERENCES.update(original_softmax)

    def test_router_tensor_names_resolve_logical_and_physical_coordinates(self):
        expected = {
            (5, 10): ("model.layers.5.mlp.router.classifier.weight",
                      "blk.10.ffn_gate_inp.weight"),
            (6, 12): ("model.layers.6.mlp.router.classifier.weight",
                      "blk.12.ffn_gate_inp.weight"),
        }
        for logical, physical in EXTRACTOR.LAYERS:
            python_name = EXTRACTOR.resolve_tensor_name(
                EXTRACTOR.DEFAULT_PYTHON_NAME_TEMPLATE, logical, physical)
            gguf_name = EXTRACTOR.resolve_tensor_name(
                EXTRACTOR.DEFAULT_GGUF_NAME_TEMPLATE, logical, physical)
            self.assertEqual((python_name, gguf_name), expected[(logical, physical)])
            self.assertEqual(EXTRACTOR.resolve_tensor_name(
                "L{logical}-P{physical}-A{layer}", logical, physical),
                f"L{logical}-P{physical}-A{logical}")
        physical_source = np.zeros((3072, 384), np.float32)
        canonical, transposed = EXTRACTOR.canonical_router_weight(physical_source)
        self.assertEqual(canonical.shape, (384, 3072))
        self.assertTrue(transposed)
        for logical, physical in EXTRACTOR.LAYERS:
            record = EXTRACTOR.make_weight_record(
                logical, physical, "gguf",
                EXTRACTOR.resolve_tensor_name(EXTRACTOR.DEFAULT_GGUF_NAME_TEMPLATE, logical, physical),
                canonical, "BF16", transposed, "test.gguf")
            self.assertEqual((record["logical_layer"], record["physical_even_block"]),
                             (logical, physical))
            self.assertEqual(record["name_template_coordinates"],
                             {"logical": logical, "physical": physical, "layer_alias": logical})
            self.assertTrue(record["source_was_transposed"])

    def test_enum_safe_gguf_router_tensor_types_and_metadata_label(self):
        gguf_path = Path(__file__).parents[1] / "gguf-py"
        sys.path.insert(0, str(gguf_path))
        try:
            from gguf.constants import GGMLQuantizationType
            for tensor_type, label in ((GGMLQuantizationType.F32, "F32"),
                                       (GGMLQuantizationType.F16, "F16"),
                                       (GGMLQuantizationType.BF16, "BF16")):
                self.assertEqual(EXTRACTOR.validate_gguf_router_tensor_type(tensor_type), label)
            with self.assertRaisesRegex(ValueError, "Q4_0"):
                EXTRACTOR.validate_gguf_router_tensor_type(GGMLQuantizationType.Q4_0)
            kind = EXTRACTOR.validate_gguf_router_tensor_type(GGMLQuantizationType.BF16)
            record = EXTRACTOR.make_weight_record(
                5, 10, "gguf", "blk.5.ffn_gate_inp.weight",
                np.zeros((384, 512), np.float32), kind, False, "test.gguf")
            self.assertEqual(record["source_dtype"], "BF16")
            self.assertNotEqual(record["source_dtype"], str(GGMLQuantizationType.BF16))
        finally:
            sys.path.remove(str(gguf_path))

    def test_bounded_safetensor_reader_supports_bf16_and_f16(self):
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError as exc:
            self.fail(f"checkpoint-free extractor dependencies unavailable: {exc}")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shard = root / "router.safetensors"
            bf16 = torch.arange(32, dtype=torch.float32).reshape(4, 8).to(torch.bfloat16)
            f16 = torch.arange(24, dtype=torch.float32).reshape(3, 8).to(torch.float16)
            untouched = torch.ones((2, 2), dtype=torch.float32)
            save_file({"wanted.bf16": bf16, "wanted.f16": f16, "unrequested": untouched}, shard)
            index = {"weight_map": {name: shard.name for name in
                                    ("wanted.bf16", "wanted.f16", "unrequested")}}
            got, dtype, location = EXTRACTOR.safetensor_weight(root, index, "wanted.bf16")
            self.assertEqual((dtype, location, got.dtype), ("torch.bfloat16", shard.name, np.float32))
            self.assertTrue(got.flags.c_contiguous)
            np.testing.assert_array_equal(got, bf16.float().numpy())
            got, dtype, _ = EXTRACTOR.safetensor_weight(root, index, "wanted.f16")
            self.assertEqual((dtype, got.dtype), ("torch.float16", np.float32))
            np.testing.assert_array_equal(got, f16.float().numpy())

    def test_router_linear_orientation_and_weight_equivalence(self):
        python = np.linspace(-1, 1, 384 * 512, dtype=np.float32).reshape(384, 512)
        gguf = PARITY.bf16_round_to_float32(python)
        canonical, transposed = EXTRACTOR.canonical_router_weight(python.T)
        self.assertTrue(transposed)
        np.testing.assert_array_equal(canonical, python)
        audit = PARITY.router_weight_equivalence_audit(python, gguf)
        self.assertTrue(audit["orientation_validated"])
        self.assertTrue(audit["gguf_equals_bf16_rounded_python_exactly"])
        mismatch = gguf.copy()
        mismatch[0, 0] += 1
        self.assertFalse(PARITY.router_weight_equivalence_audit(
            python, mismatch)["weights_equivalent_for_shared_weight_analysis"])
        with self.assertRaisesRegex(ValueError, "canonical"):
            PARITY.router_weight_equivalence_audit(python, python.T)

    def test_router_input_lineage_uses_ffn_norm_not_block_input(self):
        prefix = "physical_block_10__"
        attended = [1, 3]
        sides = {}
        for name in ("cpp", "default", "math"):
            norm = PARITY.bf16_round_to_float32(
                np.linspace(-1.3, 1.7, 16, dtype=np.float32).reshape(1, 2, 8))
            weight = np.array([[[0.37], [0.61]]], np.float32)
            sides[name] = {prefix + "ffn_norm": norm,
                           prefix + "block_input": norm + 2,
                           prefix + "identity_weight_sum": weight,
                           prefix + "identity_residual": norm * weight}
        report = PARITY.router_input_lineage_audit(sides, prefix, attended)
        self.assertEqual(report["status"], "established")
        self.assertEqual(report["router_input_canonical_surface"], "ffn_norm")
        self.assertFalse(report["proposed_block_input_alias_is_exact"])
        self.assertTrue(report["source_lineage_established"])
        self.assertTrue(report["numerical_lineage_consistent"])
        self.assertGreater(report["numerical_consistency_audit"]["cpp"][
            "division_vs_ffn_norm_raw_metrics"]["maximum_absolute_difference"], 0)
        self.assertTrue(report["numerical_consistency_audit"]["cpp"][
            "bf16_rounded_division_equals_ffn_norm_exactly"])
        sides["cpp"][prefix + "identity_residual"] += 0.1
        self.assertEqual(PARITY.router_input_lineage_audit(sides, prefix, attended)["status"],
                         "not established")

    def _linear_sides(self, input_effect=0.0, residual_effect=0.0):
        prefix = "physical_block_10__"
        hidden = 512
        weight = np.zeros((384, hidden), np.float32)
        weight[12, 0] = np.float32(input_effect)
        inputs = {"cpp": np.zeros((1, 1, hidden), np.float32),
                  "default": np.zeros((1, 1, hidden), np.float32),
                  "math": np.zeros((1, 1, hidden), np.float32)}
        inputs["default"][0, 0, 0] = inputs["math"][0, 0, 0] = 1
        base = np.full((1, 1, 384), -10, np.float32)
        base[..., :11] = 10
        base[..., 11] = np.float32(0.1)
        base[..., 12] = 0
        residuals = {"cpp": base.copy(), "default": base.copy(), "math": base.copy()}
        residuals["default"][..., 12] += np.float32(residual_effect)
        residuals["math"][..., 12] += np.float32(residual_effect)
        sides = {}
        for name in ("cpp", "default", "math"):
            logits = PARITY.diagnostic_router_linear(inputs[name].reshape(1, hidden), weight, "float32_matmul").reshape(1, 1, 384) + residuals[name]
            probabilities = PARITY.diagnostic_softmax_stable_float32(logits)
            indices = PARITY._score_topk_indices(probabilities.reshape(384)).reshape(1, 1, 12)[..., ::-1]
            identity_weight = np.array([[[0.37]]], np.float32)
            sides[name] = {prefix + "ffn_norm": inputs[name], prefix + "block_input": inputs[name] + 2,
                           prefix + "identity_weight_sum": identity_weight,
                           prefix + "identity_residual": inputs[name] * identity_weight,
                           prefix + "router_logits": logits, prefix + "router_probabilities": probabilities,
                           prefix + "router_selection_scores": probabilities.copy(),
                           prefix + "router_topk_indices": indices}
        return sides, weight

    def test_router_linear_decomposition_end_to_end_classifications(self):
        cases = ((0.25, 0, "router-input component sufficient"),
                 (0, 0.25, "diagnostic-linear residual sufficient"),
                 (0.25, 0.25, "both components independently sufficient"),
                 (0.0625, 0.0625, "requires both components"),
                 (0, 0, "native outcomes already equal"))
        for input_effect, residual_effect, expected in cases:
            sides, weight = self._linear_sides(input_effect, residual_effect)
            report = PARITY.router_linear_decomposition(sides, "physical_block_10__", [0], weight,
                                                        PARITY.bf16_round_to_float32(weight))
            self.assertEqual(report["status"], "complete")
            classifications = {softmax["classification"] for variant in report["variants"].values()
                               for softmax in variant["tokens"][0]["softmax_references"].values()}
            self.assertEqual(classifications, {expected})
            projection = report["variants"]["float32_matmul"]["tokens"][0]["direct_delta_projection"]
            if residual_effect == 0:
                self.assertLess(projection["raw"]["residual_rms"], 1e-6)
            else:
                self.assertGreater(projection["raw"]["residual_rms"], 0)
            for variant in report["variants"].values():
                selected = variant["tokens"][0]["softmax_references"]["stable_float32"]["coalitions"]
                self.assertTrue(all(len(row["selected_expert_set"]) == 12 for row in selected.values()))

    def test_weight_mismatch_fallback_contains_logits_and_membership(self):
        sides, python_weight = self._linear_sides(0.25, 0)
        gguf_weight = python_weight.copy()
        gguf_weight[13, 0] = 0.5
        report = PARITY.router_linear_decomposition(
            sides, "physical_block_10__", [0], python_weight, gguf_weight)
        self.assertEqual(report["status"], "weight representation differs")
        fallback = report["weight_sensitive_fallback"]
        self.assertFalse(fallback["shared_weight_input_residual_verdict_emitted"])
        for variant in fallback["variants"].values():
            self.assertEqual(len(variant["combinations"]), 4)
            for combination in variant["combinations"].values():
                self.assertIn("captured_logit_comparisons", combination)
                self.assertEqual(set(combination["softmax_membership_evidence"]),
                                 set(PARITY.DIAGNOSTIC_SOFTMAX_REFERENCES))
                self.assertEqual(len(combination["softmax_membership_evidence"]["stable_float32"][0][
                    "selected_expert_set"]), 12)

    def test_linear_decomposition_surfaces_reference_disagreement_and_primary_alignment(self):
        sides, weight = self._linear_sides(0.25, 0)
        original_softmax = PARITY.DIAGNOSTIC_SOFTMAX_REFERENCES["stable_float64_then_float32"]

        def shifted(logits):
            value = original_softmax(logits).copy()
            value[..., 13] += 0.5
            return value
        PARITY.DIAGNOSTIC_SOFTMAX_REFERENCES["stable_float64_then_float32"] = shifted
        try:
            report = PARITY.router_linear_decomposition(
                sides, "physical_block_10__", [0], weight, PARITY.bf16_round_to_float32(weight))
        finally:
            PARITY.DIAGNOSTIC_SOFTMAX_REFERENCES["stable_float64_then_float32"] = original_softmax
        self.assertTrue(any(not row["tokens"][0]["cross_softmax_classification_agreement"]
                            for row in report["variants"].values()))
        original_linear = PARITY.diagnostic_router_linear

        def variant_shift(router_input, router_weight, variant):
            value = original_linear(router_input, router_weight, variant)
            if variant == "float64_matmul_then_float32":
                value = value.copy()
                value[..., 13] += router_input[..., 0] * 20
            return value
        PARITY.diagnostic_router_linear = variant_shift
        try:
            matmul_report = PARITY.router_linear_decomposition(
                sides, "physical_block_10__", [0], weight, PARITY.bf16_round_to_float32(weight))
        finally:
            PARITY.diagnostic_router_linear = original_linear
        aligned_sides, aligned_weight = self._linear_sides(0, 0)
        aligned = PARITY.router_linear_decomposition(aligned_sides, "physical_block_10__", [0],
                                                     aligned_weight, PARITY.bf16_round_to_float32(aligned_weight))
        summary = PARITY.primary_router_linear_summary([
            {"physical_block": 10, "analysis": matmul_report}, {"physical_block": 12, "analysis": aligned}], 10)
        self.assertTrue(summary["block_12_remains_available_as_aligned_comparison"])
        self.assertFalse(summary["cross_matmul_agreement"])

    def _write_router_artifacts(self, root, arrays=None):
        if arrays is None:
            arrays = {key: np.zeros((384, 3072), np.float32) for key in PARITY.ROUTER_LINEAR_ARRAY_KEYS}
        npz = root / "router.npz"
        metadata_path = root / "router.json"
        EXTRACTOR.deterministic_npz(npz, arrays)
        records = []
        for block in (10, 12):
            for short, source in (("python", "python_checkpoint"), ("gguf", "gguf")):
                key = f"physical_block_{block:02d}__{short}_weight"
                value = arrays[key]
                records.append({"logical_layer": block // 2, "physical_even_block": block,
                                "source": source, "source_tensor_name": key, "canonical_tensor_orientation": "experts_by_hidden",
                                "shape": list(value.shape), "source_dtype": "torch.bfloat16" if short == "python" else "BF16",
                                "serialized_dtype": str(value.dtype), "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                                "finite_audit": {"finite": bool(np.isfinite(value).all()), "element_count": int(value.size)}})
        equivalence = []
        for block in (10, 12):
            py = arrays[f"physical_block_{block:02d}__python_weight"]
            gg = arrays[f"physical_block_{block:02d}__gguf_weight"]
            equivalence.append({"logical_layer": block // 2, "physical_even_block": block,
                                "python_array_key": f"physical_block_{block:02d}__python_weight",
                                "gguf_array_key": f"physical_block_{block:02d}__gguf_weight",
                                "weight_equivalence_audit": PARITY.router_weight_equivalence_audit(py, gg)})
        metadata = {"schema_version": 2, "model_instantiated": False, "inference_executed": False,
                    "bounded_physical_blocks": [10, 12], "weight_records": records,
                    "weight_equivalence_records": equivalence,
                    "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest()}
        metadata_path.write_text(json.dumps(metadata), encoding="ascii")
        return npz, metadata_path, metadata

    def test_router_artifact_json_npz_binding_rejections(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            npz, metadata_path, metadata = self._write_router_artifacts(root)
            arrays, _ = PARITY.load_router_linear_artifacts(npz, metadata_path)
            self.assertEqual(set(arrays), PARITY.ROUTER_LINEAR_ARRAY_KEYS)
            # Swapped artifacts are diagnosed before analysis.
            with self.assertRaises(ValueError): PARITY.load_router_linear_artifacts(metadata_path, npz)
            # Metadata-to-array hash mismatch.
            damaged = json.loads(json.dumps(metadata))
            damaged["weight_records"][0]["sha256"] = "0" * 64
            metadata_path.write_text(json.dumps(damaged), encoding="ascii")
            with self.assertRaisesRegex(ValueError, "array SHA-256"):
                PARITY.load_router_linear_artifacts(npz, metadata_path)
            # Extra/missing keys, wrong shape, and non-finite content are rejected after rebinding NPZ hash.
            for mutation in ("extra", "missing", "shape", "nonfinite"):
                values = {key: value.copy() for key, value in arrays.items()}
                if mutation == "extra": values["extra"] = np.zeros((1,), np.float32)
                elif mutation == "missing": values.pop(next(iter(values)))
                elif mutation == "shape": values[next(iter(values))] = np.zeros((384, 3071), np.float32)
                else: values[next(iter(values))][0, 0] = np.nan
                candidate = root / f"{mutation}.npz"
                EXTRACTOR.deterministic_npz(candidate, values)
                rebound = json.loads(json.dumps(metadata))
                rebound["npz_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
                candidate_json = root / f"{mutation}.json"
                candidate_json.write_text(json.dumps(rebound), encoding="ascii")
                with self.assertRaises(ValueError):
                    PARITY.load_router_linear_artifacts(candidate, candidate_json)

    def test_diagnostic_router_linear_variants_and_projection(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(2, 512)).astype(np.float32)
        weight = rng.normal(size=(384, 512)).astype(np.float32)
        for variant in PARITY.DIAGNOSTIC_LINEAR_VARIANTS:
            logits = PARITY.diagnostic_router_linear(x, weight, variant)
            self.assertEqual(logits.shape, (2, 384))
            self.assertTrue(np.isfinite(logits).all())
        delta = PARITY.diagnostic_router_linear(x[:1] - x[1:], weight, "float32_matmul")[0]
        captured = PARITY.diagnostic_router_linear(x[:1], weight, "float32_matmul")[0] - \
            PARITY.diagnostic_router_linear(x[1:], weight, "float32_matmul")[0]
        exact = PARITY._linear_projection_metrics(captured, delta)
        self.assertLess(exact["residual_rms"], 1e-5)
        inexact = PARITY._linear_projection_metrics(captured + 1, delta)
        self.assertGreater(inexact["residual_rms"], 0.9)

    def test_router_linear_membership_classifications(self):
        classify = PARITY.classify_router_linear_membership
        self.assertEqual(classify(False, True, True, False), "router-input component sufficient")
        self.assertEqual(classify(False, True, False, True), "diagnostic-linear residual sufficient")
        self.assertEqual(classify(False, True, False, False), "requires both components")
        self.assertEqual(classify(False, False, True, False), "analysis not decisive")
        self.assertEqual(classify(True, True, False, False), "native outcomes already equal")

    def test_router_linear_artifact_cli_is_replay_only(self):
        required = ["--reference-dir", "r", "--precision", "bf16", "--output-dir", "o",
                    "--block-components-window-diagnostic", "1", "--component-window-replay-only", "1",
                    "--block-components-window-default-npz", "d", "--block-components-window-math-npz", "m",
                    "--router-linear-diagnostic-npz", "w.npz", "--router-linear-diagnostic-json", "w.json",
                    "--case", "eos_window_position_2"]
        PARITY.validate_all_blocks_options(PARITY.build_parser().parse_args(required))
        for extra in (["--model", "model"], ["--capture-exe", "capture"]):
            with self.assertRaisesRegex(ValueError, "must not receive"):
                PARITY.validate_all_blocks_options(PARITY.build_parser().parse_args(required + extra))
        incomplete = required[:-4] + ["--router-linear-diagnostic-npz", "w.npz", "--case", "eos_window_position_2"]
        with self.assertRaises(ValueError):
            PARITY.validate_all_blocks_options(PARITY.build_parser().parse_args(incomplete))

    def test_component_window_inventory_validation_and_comparison(self):
        names = PARITY.component_window_names(10, 4)
        self.assertEqual(len(names), 44)
        self.assertEqual(sum(name.startswith("physical_block_10__") for name in names), 15)
        self.assertEqual(sum(name.startswith("physical_block_11__") for name in names), 7)
        self.assertEqual(sum(name.startswith("physical_block_12__") for name in names), 15)
        self.assertEqual(sum(name.startswith("physical_block_13__") for name in names), 7)
        for args in ((11, 4), (10, 3), (26, 4)):
            with self.assertRaises(ValueError): PARITY.component_window_names(*args)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = []
            default = {}
            math = {}
            reverse = {value: key for key, value in PARITY.COMPONENT_CPP_BASE.items()}
            for name in names:
                block = int(name[15:17])
                suffix = name.split("__", 1)[1]
                width = (384 if suffix in {"router_logits", "router_probabilities", "router_selection_scores"}
                         else 12 if suffix in {"router_topk_indices", "router_topk_weights"}
                         else 1 if suffix == "identity_weight_sum" else 3072)
                integer = suffix == "router_topk_indices"
                value = np.zeros((1, 2, width), np.int32 if integer else np.float32)
                if integer: value[:] = np.arange(12)
                default[name] = value.copy()
                math[name] = value.copy()
                raw = root / f"{reverse[suffix]}-{block}.raw"
                value.tofile(raw)
                dims = "1,12,2,1" if suffix == "router_topk_weights" else f"{width},2,1,1"
                rows.append(f"{reverse[suffix]}-{block}\t{'i32' if integer else 'f32'}\t{dims}\t{raw.name}")
            (root / "block-components-window-diagnostics.tsv").write_text("\n".join(rows) + "\n")
            dp = root / "d.npz"
            mp = root / "m.npz"
            np.savez(dp, **default)
            np.savez(mp, **math)
            with np.load(dp) as d, np.load(mp) as m:
                report = PARITY.compare_component_window(d, m, root, [1, 1],
                                                         {"atol": .125, "rtol": .03125})
            self.assertEqual(report["array_count"], 44)
            self.assertEqual(report["primary_oracle"], "python_default")
            self.assertEqual(len(report["even_block_shortcut_analysis"]), 2)
            self.assertEqual(len(report["router_logit_softmax_decomposition"]), 2)
            self.assertIn("primary_oracle_router_logit_softmax_decomposition", report)
            self.assertEqual(set(report["odd_block_cross_substitution"]),
                             {"block_11_vs_default", "block_11_vs_math",
                              "block_13_vs_default", "block_13_vs_math"})
            self.assertTrue({
                "first_primary_oracle_component_failure",
                "first_primary_oracle_real_expert_difference",
                "first_primary_oracle_identity_presence_difference",
                "first_primary_oracle_shortcut_failure_while_even_output_passes",
                "first_both_backend_component_failure"} <= set(report))
            for attribution in report["odd_block_cross_substitution"].values():
                self.assertEqual(attribution["coalition_count"], 8)
                self.assertAlmostEqual(attribution["shapley_additivity_error"], 0.0, places=12)
                self.assertFalse(attribution["dominant_branch_is_exclusive_causality"])
            for oracle in (default, math):
                oracle["physical_block_10__moe_shortcut"][:] = 1.0
                oracle["physical_block_11__block_output"][:] = 1.0
                oracle["physical_block_10__router_topk_indices"][..., -1] = 256
            np.savez(dp, **default)
            np.savez(mp, **math)
            with np.load(dp) as d, np.load(mp) as m:
                events = PARITY.compare_component_window(
                    d, m, root, [1, 1], {"atol": .125, "rtol": .03125})
            self.assertEqual(events["first_primary_oracle_component_failure"],
                             "physical_block_10__moe_shortcut")
            self.assertEqual(events["first_primary_oracle_real_expert_difference"], 10)
            self.assertEqual(events["first_primary_oracle_identity_presence_difference"], 10)
            self.assertEqual(events["first_primary_oracle_shortcut_failure_while_even_output_passes"],
                             "physical_block_10__moe_shortcut")
            self.assertEqual(events["first_both_backend_component_failure"],
                             "physical_block_10__moe_shortcut")
            self.assertEqual(events["primary_oracle_router_probability_bias_decomposition"][
                "physical_block"], 10)
            self.assertEqual(len(events["router_probability_bias_decomposition"]), 2)
            default[names[0]][0, 0, 0] = np.nan
            np.savez(dp, **default)
            with np.load(dp) as d, np.load(mp) as m, self.assertRaisesRegex(ValueError, "non-finite"):
                PARITY.compare_component_window(d, m, root, [1, 1], {"atol": .125, "rtol": .03125})

    def test_odd_cross_substitution_branch_attribution(self):
        def arrays(cpp_trunk, py_trunk, cpp_shortcut, py_shortcut):
            cpp = {}
            py = {}
            def p(b, s): return f"physical_block_{b:02d}__{s}"
            for side, trunk, shortcut in ((cpp, cpp_trunk, cpp_shortcut), (py, py_trunk, py_shortcut)):
                side[p(11, "post_attention_residual")] = np.array([[[trunk]]], np.float32)
                side[p(11, "dense_output")] = np.zeros((1, 1, 1), np.float32)
                side[p(10, "moe_shortcut")] = np.array([[[shortcut]]], np.float32)
            py[p(11, "block_output")] = PARITY.bf16_round_to_float32(
                PARITY.bf16_round_to_float32(py[p(11, "post_attention_residual")]) + py[p(10, "moe_shortcut")])
            return cpp, py
        criterion = {"atol": .001, "rtol": 0.0}
        self.assertEqual(PARITY.odd_cross_substitution(*arrays(1, 1, 0, 1), 11, criterion)["dominant_branch"],
                         "previous-even MoE shortcut")
        self.assertEqual(PARITY.odd_cross_substitution(*arrays(0, 1, 1, 1), 11, criterion)["dominant_branch"],
                         "odd attention/dense trunk")
        self.assertEqual(PARITY.odd_cross_substitution(*arrays(0, 1, 0, 1), 11, criterion)["dominant_branch"],
                         "mixed")
        report = PARITY.odd_cross_substitution(*arrays(1, 1, 0, 1), 11, criterion)
        self.assertTrue(report["alternatives"]["all_python_reconstruction"]["exact_reconstruction"])

    def test_three_component_coalitions_shapley_and_pass_restoration(self):
        def arrays(cpp_values, python_values):
            cpp = {}
            python = {}
            def p(b, s): return f"physical_block_{b:02d}__{s}"
            for side, values in ((cpp, cpp_values), (python, python_values)):
                side[p(11, "post_attention_residual")] = np.array([[[values[0]]]], np.float32)
                side[p(11, "dense_output")] = np.array([[[values[1]]]], np.float32)
                side[p(10, "moe_shortcut")] = np.array([[[values[2]]]], np.float32)
            python[p(11, "block_output")] = PARITY.bf16_round_to_float32(
                PARITY.bf16_round_to_float32(
                    python[p(11, "post_attention_residual")] + python[p(11, "dense_output")])
                + python[p(10, "moe_shortcut")])
            return cpp, python

        criterion = {"atol": 0.001, "rtol": 0.0}
        player_fields = (
            "shapley_rms_reduction_post_attention_residual",
            "shapley_rms_reduction_dense_output",
            "shapley_rms_reduction_previous_even_moe_shortcut")
        for changed, field in enumerate(player_fields):
            python_values = [0.0, 0.0, 0.0]
            python_values[changed] = 1.0
            report = PARITY.odd_cross_substitution(
                *arrays((0.0, 0.0, 0.0), python_values), 11, criterion)
            self.assertEqual(report["coalition_count"], 8)
            self.assertEqual(len(report["coalitions"]), 8)
            self.assertTrue(report["coalitions"]["py_P + py_D + py_S"]["exact_reconstruction"])
            self.assertAlmostEqual(report[field], report["total_all_cpp_to_all_python_rms_reduction"])
            self.assertTrue(all(abs(report[other]) < 1e-12 for other in player_fields if other != field))
            self.assertAlmostEqual(report["shapley_additivity_error"], 0.0, places=12)

        interaction = PARITY.odd_cross_substitution(
            *arrays((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), 11, criterion)
        self.assertAlmostEqual(
            interaction["shapley_rms_reduction_post_attention_residual"],
            interaction["shapley_rms_reduction_dense_output"], places=12)
        self.assertAlmostEqual(interaction["shapley_sum"],
                               interaction["total_all_cpp_to_all_python_rms_reduction"], places=12)
        self.assertLessEqual(abs(interaction["shapley_additivity_error"]), 1e-12)

        multiple = PARITY.odd_cross_substitution(
            *arrays((0.0, 0.0, 0.0), (1.0, 1.0, 0.0)), 11,
            {"atol": 1.0, "rtol": 0.0})
        self.assertEqual(multiple["threshold_crossing_attribution"],
                         "multiple single components independently sufficient")
        self.assertTrue(multiple["post_attention_residual_only_restores_pass"])
        self.assertTrue(multiple["dense_output_only_restores_pass"])
        self.assertFalse(multiple["dominant_branch_is_exclusive_causality"])

        requires_two = PARITY.odd_cross_substitution(
            *arrays((0.0, 0.0, 0.0), (1.0, 1.0, 0.0)), 11,
            {"atol": 0.5, "rtol": 0.0})
        self.assertEqual(requires_two["threshold_crossing_attribution"],
                         "requires multiple components")
        self.assertTrue(requires_two[
            "post_attention_residual_and_dense_output_restores_pass"])

    def test_odd_coalition_rounds_after_both_official_additions(self):
        original = PARITY.bf16_round_to_float32
        calls = []

        def observed(value):
            calls.append(np.asarray(value).copy())
            return original(value)
        PARITY.bf16_round_to_float32 = observed
        try:
            p = np.array([[[1.00390625]]], np.float32)
            d = np.array([[[0.00390625]]], np.float32)
            s = np.array([[[0.0078125]]], np.float32)
            got = PARITY.reconstruct_odd_coalition(p, d, s)
        finally:
            PARITY.bf16_round_to_float32 = original
        self.assertEqual(len(calls), 2)
        first = original(p + d)
        np.testing.assert_array_equal(calls[0], p + d)
        np.testing.assert_array_equal(calls[1], first + s)
        np.testing.assert_array_equal(got, original(first + s))

    def test_router_cutoff_uses_complete_scores_not_returned_order(self):
        scores = -np.arange(PARITY.ROUTED_EXPERT_COUNT, dtype=np.float32)
        returned_unsorted = np.array([11, 0, 10, 1, 9, 2, 8, 3, 7, 4, 6, 5], np.int32)
        report = PARITY.router_token_cutoff(scores, returned_unsorted, 7)
        self.assertEqual(report["selected_expert_set"], list(range(12)))
        self.assertFalse(report["returned_topk_order_used_for_ranking"])
        self.assertEqual(report["ranking_source"], "complete_384_wide_selection_scores")
        self.assertEqual(report["lowest_selected_score"], -11.0)
        self.assertEqual(report["highest_unselected_score"], -12.0)
        self.assertEqual(report["topk_cutoff_margin"], 1.0)
        self.assertEqual([row["rank"] for row in report["ranked_experts_around_cutoff"]],
                         list(range(8, 17)))

        identity_scores = scores.copy()
        identity_scores[300] = -7.5
        identity_selected = np.array(list(range(11)) + [300], np.int32)
        identity = PARITY.router_token_cutoff(identity_scores, identity_selected, 7)
        identity_row = next(row for row in identity["ranked_experts_around_cutoff"]
                            if row["expert_id"] == 300)
        self.assertEqual(identity_row["expert_class"], "identity")
        self.assertTrue(identity_row["selected"])

    def test_router_disputed_experts_sign_flip_and_order_only(self):
        left_scores = -np.arange(PARITY.ROUTED_EXPERT_COUNT, dtype=np.float32)
        right_scores = left_scores.copy()
        right_scores[11], right_scores[12] = right_scores[12], right_scores[11]
        left = PARITY.router_token_cutoff(left_scores, np.arange(12), 3)
        right = PARITY.router_token_cutoff(right_scores,
                                           np.array(list(range(11)) + [12]), 3)
        disputed = PARITY.disputed_router_token(left, right, "cpp", "default")
        self.assertEqual(disputed["experts_selected_only_by_left"], [11])
        self.assertEqual(disputed["experts_selected_only_by_right"], [12])
        self.assertTrue(disputed["ordering_inversions"][0]["gap_sign_flips"])
        self.assertTrue(disputed["has_ordering_inversion"])
        self.assertEqual(disputed["numerical_scale"]["left_cutoff_margin"], 1.0)
        self.assertEqual(disputed["numerical_scale"]["right_cutoff_margin"], 1.0)
        self.assertTrue(all("left_rank_distance_from_top12_boundary" in row
                            and "right_rank_distance_from_top12_boundary" in row
                            for row in disputed["disputed_experts"]))
        reordered = PARITY.router_token_cutoff(left_scores, np.arange(11, -1, -1), 3)
        self.assertIsNone(PARITY.disputed_router_token(left, reordered, "cpp", "default"))

    def test_router_correction_bias_reconstruction_and_pairwise_mismatch(self):
        probabilities = np.zeros((1, 2, PARITY.ROUTED_EXPERT_COUNT), np.float32)
        bias = np.linspace(-0.25, 0.25, PARITY.ROUTED_EXPERT_COUNT, dtype=np.float32)
        reconstructed, report = PARITY.reconstruct_router_correction_bias(
            probabilities, probabilities + bias[None, None, :])
        self.assertTrue(report["token_invariant_byte_exact"])
        self.assertEqual(report["maximum_per_expert_range_across_tokens"], 0.0)
        self.assertEqual(report["maximum_absolute_deviation_from_first_token"], 0.0)
        np.testing.assert_array_equal(reconstructed[0], bias)
        mismatch = PARITY._vector_metrics(bias, bias + np.float32(0.01))
        self.assertFalse(mismatch["exact_equality"])
        self.assertGreater(mismatch["maximum_absolute_difference"], 0)
        self.assertGreater(mismatch["rms_difference"], 0)

    def test_primary_router_summary_prefers_first_affected_block_and_separates_weight_order(self):
        dispute = {"attended_token": 1, "has_ordering_inversion": True,
                   "experts_selected_only_by_left": [11], "experts_selected_only_by_right": [12],
                   "disputed_experts": [{"expert_id": 11, "left_rank": 12, "right_rank": 13},
                                        {"expert_id": 12, "left_rank": 13, "right_rank": 12}]}
        bias_metrics = {"cpp_vs_python_default": {"maximum_absolute_difference": 0.01}}
        cutoffs = [
            {"physical_block": 10,
             "pairwise_disputed_experts": {"cpp_vs_python_default": [dispute]},
             "reconstructed_correction_bias": {"pairwise_metrics": bias_metrics}},
            {"physical_block": 12,
             "pairwise_disputed_experts": {"cpp_vs_python_default": []},
             "reconstructed_correction_bias": {"pairwise_metrics": bias_metrics}},
        ]
        suffixes = ("router_logits", "router_probabilities", "router_selection_scores",
                    "router_topk_weights", "identity_weight_sum", "identity_residual", "moe_shortcut")
        components = [{"physical_block": 10, "suffix": suffix,
                       "cpp_vs_python_default": {"within_diagnostic_criterion":
                                                 suffix != "router_topk_weights"}}
                      for suffix in suffixes]
        even = [{"physical_block": 10, "expert_id_aligned_topk_weights": {
            "cpp_vs_python_default": {"within_diagnostic_criterion": True}}}]
        summary = PARITY.primary_router_cutoff_summary(cutoffs, components, even)
        self.assertEqual(summary["physical_block"], 10)
        self.assertEqual(summary["affected_attended_tokens"], [1])
        self.assertTrue(summary["returned_order_topk_weight_failure_not_treated_as_weight_math_failure"])
        self.assertEqual(summary["descriptive_evidence"],
                         "continuous scores remain within criterion while discrete membership differs")

    def test_probability_bias_coalitions_membership_and_decompositions(self):
        def make_side(probabilities, bias):
            probabilities = np.asarray(probabilities, np.float32)
            bias = np.asarray(bias, np.float32)
            scores = np.asarray(probabilities + bias, np.float32)
            indices = PARITY._score_topk_indices(scores)
            return {"router_probabilities": probabilities[None, None, :],
                    "router_selection_scores": scores[None, None, :],
                    "router_topk_indices": indices[None, None, :]}

        def run(left_probability, left_bias, right_probability, right_bias):
            left = make_side(left_probability, left_bias)
            right = make_side(right_probability, right_bias)
            sides = {"cpp": {"p" + key: value for key, value in left.items()},
                     "default": {"p" + key: value for key, value in right.items()}}
            return PARITY.probability_bias_pair_decomposition(
                sides, "p", [0], "cpp", "default")["tokens"][0]

        base = np.full(PARITY.ROUTED_EXPERT_COUNT, -10.0, np.float32)
        base[:11] = np.arange(20, 9, -1, dtype=np.float32)
        left_probability = base.copy()
        left_probability[11] = 1.0
        left_probability[12] = -1.0
        zero_bias = np.zeros_like(base)

        right_probability = left_probability.copy()
        right_probability[11], right_probability[12] = -1.0, 1.0
        probability = run(left_probability, zero_bias, right_probability, zero_bias)
        self.assertEqual(probability["membership_classification"], "probability component sufficient")
        self.assertTrue(probability["right_probabilities_only_restores_right_membership"])
        self.assertFalse(probability["right_bias_only_restores_right_membership"])
        self.assertTrue(probability["ordering_inversion_decomposition"][0][
            "probability_component_alone_reverses_ordering"])
        self.assertFalse(probability["ordering_inversion_decomposition"][0][
            "bias_component_alone_reverses_ordering"])

        right_bias = zero_bias.copy()
        right_bias[11], right_bias[12] = -2.0, 2.0
        bias = run(left_probability, zero_bias, left_probability, right_bias)
        self.assertEqual(bias["membership_classification"], "bias component sufficient")
        self.assertTrue(bias["right_bias_only_restores_right_membership"])
        self.assertFalse(bias["right_probabilities_only_restores_right_membership"])
        self.assertFalse(bias["ordering_inversion_decomposition"][0][
            "probability_component_alone_reverses_ordering"])
        self.assertTrue(bias["ordering_inversion_decomposition"][0][
            "bias_component_alone_reverses_ordering"])

        interaction_probability = left_probability.copy()
        interaction_probability[11], interaction_probability[12] = 0.25, -0.25
        interaction_bias = zero_bias.copy()
        interaction_bias[11], interaction_bias[12] = -0.5, 0.5
        interaction = run(left_probability, zero_bias, interaction_probability, interaction_bias)
        self.assertEqual(interaction["membership_classification"], "requires both components")
        self.assertTrue(interaction["both_required_for_right_membership"])
        self.assertFalse(interaction["right_probabilities_only_restores_right_membership"])
        self.assertFalse(interaction["right_bias_only_restores_right_membership"])

        equal = run(left_probability, zero_bias, left_probability, zero_bias)
        self.assertEqual(equal["membership_classification"], "native memberships already equal")
        self.assertTrue(equal["native_left_membership_reconstruction_valid"])
        self.assertTrue(equal["native_right_membership_reconstruction_valid"])

        for expert in interaction["disputed_expert_delta_decomposition"]:
            self.assertTrue({"total_selection_score_delta", "probability_delta",
                             "constant_bias_delta", "float32_reconstruction_residual"} <= set(expert))
            self.assertAlmostEqual(expert["float32_reconstruction_residual"], 0.0, places=6)
        inversion = interaction["ordering_inversion_decomposition"][0]
        self.assertGreater(inversion["left_native_score_gap"], 0)
        self.assertLess(inversion["right_native_score_gap"], 0)
        self.assertAlmostEqual(inversion["decomposition_residual"], 0.0, places=6)
        self.assertFalse(inversion["probability_component_alone_reverses_ordering"])
        self.assertFalse(inversion["bias_component_alone_reverses_ordering"])
        self.assertTrue(inversion["only_combination_reproduces_native_inversion"])

    def test_constant_bias_native_reconstruction_and_dtype_grid_audit(self):
        probability = np.zeros((1, 2, PARITY.ROUTED_EXPERT_COUNT), np.float32)
        probability[..., :12] = np.arange(12, 0, -1, dtype=np.float32)
        bias = np.linspace(-0.01, 0.01, PARITY.ROUTED_EXPERT_COUNT, dtype=np.float32)
        scores = np.asarray(probability + bias[None, None, :], np.float32)
        indices = np.stack([PARITY._score_topk_indices(row) for row in scores.reshape(-1, 384)])
        constant, reconstructed, report = PARITY.constant_bias_reconstruction(
            probability, scores, indices, [0, 1])
        np.testing.assert_array_equal(reconstructed, scores.reshape(-1, 384))
        self.assertTrue(report["score_reconstruction_exact_equality"])
        self.assertTrue(report["every_token_reproduces_captured_membership"])
        self.assertTrue(all(row["selected_set_equality"]
                            for row in report["per_token_selected_set_equality"]))
        audit = PARITY.correction_bias_dtype_grid_audit(constant, bias)
        self.assertEqual(set(audit["candidates"]),
                         {"raw_float32", "bf16_rne_then_float32", "f16_rne_then_float32"})
        self.assertEqual(audit["closest_candidate_by_rms"], "raw_float32")
        self.assertTrue(audit["diagnostic_only_not_a_runtime_dtype_verdict"])
        self.assertTrue(all("exact_match_count_out_of_384" in row
                            for row in audit["candidates"].values()))

    def test_diagnostic_softmax_references_are_stable_normalized_and_shift_invariant(self):
        logits = np.linspace(-100, 100, PARITY.ROUTED_EXPERT_COUNT, dtype=np.float32)[None, :]
        for name, reference in PARITY.DIAGNOSTIC_SOFTMAX_REFERENCES.items():
            probabilities = reference(logits)
            shifted = reference(np.asarray(logits + np.float32(17.0), np.float32))
            self.assertTrue(np.isfinite(probabilities).all(), name)
            np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, rtol=0, atol=1e-6)
            np.testing.assert_allclose(probabilities, shifted, rtol=0, atol=1e-7)

    def test_logit_softmax_residual_coalitions_cover_logit_residual_and_interaction(self):
        reference = PARITY.diagnostic_softmax_stable_float32

        def side(logits, residual):
            logits = np.asarray(logits, np.float32)
            probabilities = np.asarray(reference(logits[None, :])[0] + residual, np.float32)
            bias = np.zeros(PARITY.ROUTED_EXPERT_COUNT, np.float32)
            scores = np.asarray(probabilities + bias, np.float32)
            return {"router_logits": logits[None, None, :],
                    "router_probabilities": probabilities[None, None, :],
                    "router_selection_scores": scores[None, None, :],
                    "router_topk_indices": PARITY._score_topk_indices(scores)[None, None, :]}

        def run(left_logits, left_residual, right_logits, right_residual):
            left = side(left_logits, left_residual)
            right = side(right_logits, right_residual)
            sides = {"cpp": {"p" + key: value for key, value in left.items()},
                     "default": {"p" + key: value for key, value in right.items()}}
            return PARITY.logit_softmax_pair_variant(
                sides, "p", [0], "cpp", "default", "stable_float32", reference)["tokens"][0]

        base = np.full(PARITY.ROUTED_EXPERT_COUNT, -10.0, np.float32)
        base[:11] = np.linspace(6.0, 5.0, 11, dtype=np.float32)
        base[11], base[12] = 1.0, 0.0
        swapped = base.copy()
        swapped[11], swapped[12] = 0.0, 1.0
        zero = np.zeros_like(base)
        logit = run(base, zero, swapped, zero)
        self.assertEqual(logit["membership_classification"], "logit component sufficient")
        self.assertTrue(logit["right_logits_only_restores_right_probability_outcome"])
        self.assertTrue(logit["membership_classification_decisive"])

        left_probability = reference(base[None, :])[0]
        residual = zero.copy()
        delta = float(left_probability[11] - left_probability[12])
        residual[11], residual[12] = -delta, delta
        residual_case = run(base, zero, base, residual)
        self.assertEqual(residual_case["membership_classification"],
                         "softmax-reconstruction residual sufficient")
        self.assertTrue(residual_case["right_softmax_residual_only_restores_right_probability_outcome"])

        narrowed = base.copy()
        narrowed[11], narrowed[12] = 0.6, 0.5
        narrowed_probability = reference(narrowed[None, :])[0]
        left_gap = float(left_probability[11] - left_probability[12])
        right_gap = float(narrowed_probability[11] - narrowed_probability[12])
        residual_gap = -(left_gap + right_gap) / 2
        interaction_residual = zero.copy()
        interaction_residual[11], interaction_residual[12] = residual_gap / 2, -residual_gap / 2
        interaction = run(base, zero, narrowed, interaction_residual)
        self.assertEqual(interaction["membership_classification"], "requires both components")
        self.assertTrue(interaction["both_required_for_right_probability_outcome"])
        expert = interaction["disputed_expert_probability_decomposition"][0]
        self.assertTrue({"captured_probability_delta", "diagnostic_softmax_probability_delta",
                         "softmax_reconstruction_residual_delta", "decomposition_residual",
                         "raw_logit_delta", "max_centered_logit_delta",
                         "mean_centered_logit_delta"} <= set(expert))
        self.assertAlmostEqual(expert["decomposition_residual"], 0.0, places=7)
        gap = interaction["ordering_gap_decomposition"][0]
        self.assertGreater(gap["left_captured_probability_gap"], 0)
        self.assertLess(gap["right_captured_probability_gap"], 0)
        self.assertAlmostEqual(gap["decomposition_residual"], 0.0, places=7)

        equal = run(base, zero, base, zero)
        self.assertEqual(equal["membership_classification"], "native probability outcomes already equal")

    def test_softmax_residual_reconstruction_centering_and_variant_disagreement(self):
        logits = np.linspace(-2, 2, PARITY.ROUTED_EXPERT_COUNT, dtype=np.float32)[None, :]
        captured = PARITY.diagnostic_softmax_stable_float32(logits)
        _, _, report = PARITY.diagnostic_softmax_residual(
            logits, captured, PARITY.diagnostic_softmax_stable_float32)
        self.assertTrue(report["residual_addback_exactly_reconstructs_captured_probabilities"])
        self.assertEqual(report["reconstruction_maximum_absolute_error"], 0.0)
        centered = PARITY.centered_logit_comparison(logits[0], logits[0] + np.float32(7.0))
        self.assertGreater(centered["raw_logits"]["maximum_absolute_difference"], 0)
        self.assertLess(centered["max_centered_logits"]["maximum_absolute_difference"], 1e-6)
        self.assertLess(centered["mean_centered_logits"]["maximum_absolute_difference"], 1e-6)

        def token(classification):
            return {"attended_token": 1, "membership_classification": classification,
                    "left_and_right_probability_outcomes_differ": True,
                    "left_probability_outcome_matches_captured_native_left_membership": True,
                    "right_probability_outcome_matches_captured_native_right_membership": True,
                    "membership_classification_decisive": True,
                    "right_logits_only_restores_right_probability_outcome":
                        classification == "logit component sufficient",
                    "right_softmax_residual_only_restores_right_probability_outcome": False,
                    "both_required_for_right_probability_outcome": classification == "requires both components",
                    "centered_logit_metrics": {}, "disputed_expert_probability_decomposition": [],
                    "ordering_gap_decomposition": []}
        variants = {
            "stable_float32": {"tokens": [token("logit component sufficient")],
                               "softmax_reconstruction_metrics": {}},
            "stable_float64_then_float32": {"tokens": [token("requires both components")],
                                            "softmax_reconstruction_metrics": {}}}
        aligned_variants = {name: {"tokens": [token("native probability outcomes already equal")],
                                   "softmax_reconstruction_metrics": {}}
                            for name in PARITY.DIAGNOSTIC_SOFTMAX_REFERENCES}
        for report in aligned_variants.values():
            report["tokens"][0]["left_and_right_probability_outcomes_differ"] = False
        summary = PARITY.primary_logit_softmax_summary([
            {"physical_block": 10, "pairwise": {"cpp_vs_python_default": {"variants": variants}}},
            {"physical_block": 12, "pairwise": {"cpp_vs_python_default": {"variants": aligned_variants}}}])
        self.assertEqual(summary["physical_block"], 10)
        self.assertFalse(summary["all_affected_tokens_have_reference_variant_agreement"])
        self.assertIn("diagnostic reference choice changes the categorical result",
                      summary["descriptive_evidence"])

    def _write_profile(self, root, values, legacy_indices, rounding):
        case = root / "eos_window_position_2"
        case.mkdir(parents=True)
        direct = [f"{name}\tf32\t1,1,1,1\t{name}.raw" for name in sorted(PARITY.DIRECT_NAMES)]
        (case / "captures.tsv").write_text("\n".join(direct) + "\n", encoding="ascii")
        rows = []
        reverse = {value: key for key, value in PARITY.COMPONENT_CPP_BASE.items()}
        for name in PARITY.component_names():
            block = int(name[15:17])
            suffix = name.split("__", 1)[1]
            value = values[name]
            base = reverse[suffix]
            raw = case / f"{base}-{block}.raw"
            if suffix == "router_topk_indices" and legacy_indices:
                packed = np.full(PARITY.ROUTED_EXPERT_COUNT + 12, -1, np.int32)
                packed[:12] = value[0, 0]
                packed[PARITY.ROUTED_EXPERT_COUNT:] = value[0, 1]
                packed.tofile(raw)
            else:
                value.reshape(-1).tofile(raw)
            dims = ("1,12,2,1" if suffix == "router_topk_weights" else
                    f"{value.shape[-1]},2,1,1")
            rows.append(f"{base}-{block}\t{'i32' if suffix == 'router_topk_indices' else 'f32'}\t{dims}\t{raw.name}")
        (case / "block-components-diagnostics.tsv").write_text(
            "\n".join(rows) + "\n", encoding="ascii")
        (case / "inputs.json").write_text(json.dumps({"input_ids": [1, 2], "attention_mask": [1, 1],
                                                      "position_ids": [0, 1], "cache_position": [0, 1]}), encoding="ascii")
        (case / "decoding.json").write_text(json.dumps({"argmax_id": 2909}), encoding="ascii")
        if rounding:
            (root / "capture-run-metadata.json").write_text(json.dumps({
                "longcat_bf16_boundary_rounding": True,
                "longcat_bf16_hidden_surface_rounding": True}), encoding="ascii")

    def test_profile_diff_legacy_compact_dtype_metrics_routing_and_validation(self):
        names = PARITY.component_names()
        default = {}
        math = {}
        baseline = {}
        rounded = {}
        for name in names:
            suffix = name.split("__", 1)[1]
            width = (384 if suffix in {"router_logits", "router_probabilities", "router_selection_scores"}
                     else 12 if suffix in {"router_topk_indices", "router_topk_weights"}
                     else 1 if suffix == "identity_weight_sum" else 3072)
            dtype = np.int32 if suffix == "router_topk_indices" else np.float32
            value = np.zeros((1, 2, width), dtype)
            if suffix == "router_topk_indices": value[:] = np.arange(12)
            default[name] = value.copy()
            math[name] = value.copy()
            baseline[name] = value.copy()
            rounded[name] = value.copy()
        baseline["physical_block_00__attention_output"][..., 0] = .25
        rounded["physical_block_00__attention_output"][..., 0] = .125
        baseline["physical_block_01__dense_output"][..., 0] = .125
        rounded["physical_block_01__dense_output"][..., 0] = .25
        rounded["physical_block_00__router_topk_indices"][..., 0] = 7
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            b = root / "baseline"
            r = root / "rounded"
            b.mkdir()
            r.mkdir()
            self._write_profile(b, baseline, True, False)
            self._write_profile(r, rounded, False, True)
            default_path = root / "default.npz"
            math_path = root / "math.npz"
            np.savez(default_path, **default)
            np.savez(math_path, **math)
            dtype_path = root / "dtypes.json"
            dtype_path.write_text(json.dumps({"components": [
                {"name": name, "source_torch_dtype":
                    "torch.float32" if name in PARITY.observed_float32_policy_names() else "torch.bfloat16"}
                for name in PARITY.observed_float32_policy_names() | PARITY.observed_bfloat16_policy_names()]}))
            with np.load(default_path) as d, np.load(math_path) as m:
                report = PARITY.compare_component_profiles(
                    d, m, b, r, "eos_window_position_2", [1, 1], {"atol": .125, "rtol": .03125},
                    "legacy-default-off", "cpu-flash-disabled-threads-0-bf16", dtype_path, 8228)
            attention = next(row for row in report["components"]
                             if row["name"] == "physical_block_00__attention_output")
            self.assertEqual(attention["classification_vs_python_default"], "improved")
            self.assertEqual(attention["pass_state_transition_vs_python_default"], "fail -> pass")
            dense = next(row for row in report["components"]
                         if row["name"] == "physical_block_01__dense_output")
            self.assertEqual(dense["pass_state_transition_vs_python_default"], "pass -> fail")
            self.assertEqual(
                report["aggregate_totals_by_backend_and_suffix"]["python_default"]["attention_output"]["fail -> pass"], 1)
            self.assertIn("physical_block_00__router_logits", report["python_components_observed_float32"])
            self.assertIn("physical_block_00__block_output", report["python_components_observed_bfloat16"])
            self.assertTrue(report["representability_is_not_execution_dtype"])
            self.assertEqual(report["observed_bfloat16_policy_surface_count"], 75)
            self.assertEqual(len(report["observed_bfloat16_policy_surfaces_on_bf16_grid"]), 75)
            self.assertTrue(report["observed_bfloat16_policy_coverage_complete"])
            self.assertEqual(report["observed_float32_policy_surfaces_rounded"], [])
            route = next(row for row in report["routing_profile_comparison"]
                         if row["physical_block"] == 0 and row["token"] == 0)
            self.assertTrue(route["boundary_rounding_changed_real_expert"])
            self.assertTrue(route["shortcut_remains_within_criterion_vs_both"])
            self.assertIn("does not restore", report["boundary_rounding_overall_verdict"])
            with self.assertRaisesRegex(ValueError, "legacy-default-off"):
                PARITY.validate_profile_capture(b, "eos_window_position_2", False, None)

    def test_bf16_grid_and_material_metric_helpers(self):
        self.assertEqual(len(PARITY.observed_bfloat16_policy_names()), 75)
        self.assertEqual(len(PARITY.observed_float32_policy_names()), 30)
        self.assertEqual(len(PARITY.boundary_rounding_policy_names()), 34)
        additions = PARITY.hidden_surface_rounding_additions()
        self.assertEqual(len(additions), 41)
        expected = {"physical_block_00__block_input"}
        expected.update(f"physical_block_{block:02d}__{suffix}" for block in range(10)
                        for suffix in ("attention_norm", "attention_output", "ffn_norm", "dense_output"))
        self.assertEqual(additions, expected)
        self.assertTrue(PARITY.exactly_bf16_representable(np.array([1.0, -2.0], np.float32)))
        self.assertFalse(PARITY.exactly_bf16_representable(np.array([1.001], np.float32)))
        self.assertEqual(PARITY.material_metric_change(1.0, .5), "improved")
        self.assertEqual(PARITY.material_metric_change(1.0, 2.0), "worsened")
        self.assertEqual(PARITY.material_metric_change(1.0, 1.005), "unchanged")

    def test_component_manifest_and_three_way_router_diagnostics(self):
        names = PARITY.component_names()
        self.assertEqual(len(names), 110)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = []
            default = {}
            math = {}
            cpp_values = {}
            for name in names:
                block = int(name[15:17])
                suffix = name.split("__", 1)[1]
                width = (384 if suffix in {"router_logits", "router_probabilities", "router_selection_scores"}
                         else 12 if suffix in {"router_topk_indices", "router_topk_weights"}
                         else 1 if suffix == "identity_weight_sum" else 3072)
                is_indices = suffix == "router_topk_indices"
                value = np.zeros((1, 2, width), np.int32 if is_indices else np.float32)
                if is_indices:
                    value[:] = np.arange(12)
                default[name] = value.copy()
                math[name] = value.copy()
                cpp = value.copy()
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
            default_path = root / "default.npz"
            math_path = root / "math.npz"
            np.savez(default_path, **default)
            np.savez(math_path, **math)
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
        with self.assertRaisesRegex(ValueError, "requires boundary rounding"):
            PARITY.validate_all_blocks_options(parser.parse_args(
                required + ["--longcat-bf16-hidden-surface-rounding", "1"]))
        both = parser.parse_args(required + ["--longcat-bf16-boundary-rounding", "1",
                                             "--longcat-bf16-hidden-surface-rounding", "1"])
        PARITY.validate_all_blocks_options(both)

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
        profile = replay.copy()
        profile[profile.index("--component-replay-only")] = "--component-profile-diff-replay-only"
        profile += ["--baseline-capture-dir", "baseline", "--rounded-capture-dir", "rounded",
                    "--baseline-profile-identity", "legacy-default-off",
                    "--profile-execution-context", "cpu-flash-disabled-threads-0-bf16"]
        PARITY.validate_all_blocks_options(parser.parse_args(profile))
        for extra in (["--model", "model.gguf"], ["--capture-exe", "capture"]):
            with self.assertRaises(ValueError):
                PARITY.validate_all_blocks_options(parser.parse_args(profile + extra))
        window = ["--reference-dir", "r", "--precision", "bf16", "--output-dir", "existing",
                  "--component-window-replay-only", "1", "--block-components-window-diagnostic", "1",
                  "--block-components-window-start", "10", "--block-components-window-count", "4",
                  "--block-components-window-default-npz", "default.npz",
                  "--block-components-window-math-npz", "math.npz", "--case", "eos_window_position_2"]
        PARITY.validate_all_blocks_options(parser.parse_args(window))
        for extra in (["--model", "model.gguf"], ["--capture-exe", "capture"],
                      ["--block-components-diagnostic", "1"]):
            with self.assertRaises(ValueError):
                PARITY.validate_all_blocks_options(parser.parse_args(window + extra))

        def without_pair(argv, option):
            index = argv.index(option)
            return argv[:index] + argv[index + 2:]
        crossed = [
            replay + ["--block-components-window-diagnostic", "1"],
            window + ["--block-components-diagnostic", "1"],
            without_pair(window, "--block-components-window-diagnostic"),
            without_pair(replay, "--block-components-diagnostic"),
            replay + ["--component-attribution-replay-only", "1"],
        ]
        for argv in crossed:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                PARITY.validate_all_blocks_options(parser.parse_args(argv))

    def test_component_window_replay_metadata_binds_exact_window(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "capture-run-metadata.json"
            payload = {"block_components_diagnostic": False,
                       "block_components_window_diagnostic": True,
                       "block_components_window_start": 10,
                       "block_components_window_count": 4,
                       "longcat_bf16_boundary_rounding": True,
                       "longcat_bf16_hidden_surface_rounding": True}
            path.write_text(json.dumps(payload), encoding="ascii")
            self.assertEqual(PARITY.validate_component_window_capture_metadata(root, 10, 4), payload)
            for key, value in (("block_components_window_start", 12),
                               ("block_components_window_count", 2),
                               ("block_components_window_diagnostic", False),
                               ("block_components_diagnostic", True)):
                invalid = dict(payload)
                invalid[key] = value
                path.write_text(json.dumps(invalid), encoding="ascii")
                with self.subTest(key=key), self.assertRaises(ValueError):
                    PARITY.validate_component_window_capture_metadata(root, 10, 4)
            for invalid_profile in ({}, {"longcat_bf16_boundary_rounding": "true",
                                         "longcat_bf16_hidden_surface_rounding": True},
                                    {"longcat_bf16_boundary_rounding": False,
                                     "longcat_bf16_hidden_surface_rounding": True}):
                invalid = dict(payload)
                invalid.update(invalid_profile)
                if not invalid_profile: invalid.pop("longcat_bf16_boundary_rounding")
                path.write_text(json.dumps(invalid), encoding="ascii")
                with self.assertRaises(ValueError):
                    PARITY.validate_component_window_capture_metadata(root, 10, 4)

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
                            "has_cpp_vs_python_default_real_expert_set_difference": False,
                            "has_cpp_vs_python_math_real_expert_set_difference": False,
                            "shortcut_within_criterion_vs_both": True}]
        local_passes = [{"suffix": suffix, "outside_vs_both": False,
                         "python_backends_outside_criterion": False}
                        for suffix in ("attention_output", "dense_output", "moe_shortcut", "block_output")]
        self.assertEqual(PARITY.classify_numerical_attribution(identity_router, local_passes, 7),
                         "cumulative numerical drift with no discrete local operator failure")
        real_router = [{"physical_block": 0, "has_real_expert_set_difference": True,
                        "has_identity_presence_difference": False,
                        "has_cpp_vs_python_default_real_expert_set_difference": True,
                        "has_cpp_vs_python_math_real_expert_set_difference": True,
                        "shortcut_within_criterion_vs_both": False}]
        self.assertEqual(PARITY.classify_numerical_attribution(real_router, local_passes, 7),
                         "real router-selection divergence")

    def test_bf16_boundary_contract_attribution_and_pairwise_router_fields(self):
        def metric(exact):
            return {"byte_exact_alternatives": [exact]}
        reconstruction = {}
        expected = {
            "cpp": ("pure_f32_addition", "pure_f32_addition", "all_f32_official_association"),
            "python_default": ("bf16_round_after_addition", "bf16_round_after_addition",
                               "bf16_after_each_official_addition"),
            "python_math": ("bf16_round_after_addition", "bf16_round_after_addition",
                            "bf16_after_each_official_addition"),
        }
        for side, names in expected.items():
            reconstruction[side] = [
                {"physical_block": block, "attention_residual": metric(names[0]),
                 "block_output": metric(names[1] if block % 2 == 0 else names[2])}
                for block in range(10)]
        self.assertTrue(PARITY.reconstruction_proves_bf16_boundary_contract(reconstruction))
        self.assertEqual(PARITY.classify_numerical_attribution([], [], None, reconstruction),
                         "BF16 residual-boundary precision contract mismatch")

        sums = np.ones((1, 1, 1), np.float32)
        report = PARITY.semantic_router_report(
            np.array([[[1, 256]]]), np.array([[[1, 256]]]), np.array([[[2, 256]]]),
            sums, sums, sums)
        self.assertFalse(report["has_cpp_vs_python_default_real_expert_set_difference"])
        self.assertTrue(report["has_cpp_vs_python_math_real_expert_set_difference"])

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
                        key = f"{case}/{suffix}"
                        self.files.append(key)
                        self.data[key] = np.asarray(value)

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
