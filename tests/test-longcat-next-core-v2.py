#!/usr/bin/env python3
"""Checkpoint-free schema-v2 generator safety tests."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
import hashlib
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "core_reference_v2", ROOT / "scripts/longcat-next/core_reference_v2.py")
v2 = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(v2)


class Handle:
    def remove(self): pass


class Module:
    def __init__(self): self.pre = []; self.post = []
    def register_forward_pre_hook(self, hook): self.pre.append(hook); return Handle()
    def register_forward_hook(self, hook): self.post.append(hook); return Handle()


class Layer(Module):
    def __init__(self): super().__init__(); self.input_layernorm = [Module(), Module()]


class Model:
    def __init__(self):
        trunk = type("Trunk", (), {})()
        trunk.layers = [Layer() for _ in range(14)]
        trunk.named_modules = lambda: []
        self.model = trunk


class Recorder:
    def __init__(self): self.blocks = []
    def check(self, value, **context): self.blocks.append(context["physical_block"])


class FakeTensor:
    softmax_override = None
    def __init__(self, value, dtype=None):
        self.value = np.asarray(value); self.dtype = dtype or self.value.dtype; self.device = "cpu"
    @property
    def shape(self): return self.value.shape
    def is_floating_point(self): return self.value.dtype.kind == "f"
    def numel(self): return self.value.size
    def sum(self): return FakeTensor(self.value.sum())
    def item(self): return self.value.item()
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return self.value.tolist()
    def min(self): return FakeTensor(self.value.min())
    def max(self): return FakeTensor(self.value.max())
    def abs(self): return FakeTensor(np.abs(self.value))
    def view(self, *shape):
        if len(shape) == 1 and shape[0] == FakeTorch.uint8:
            return FakeTensor(np.ascontiguousarray(self.value).view(np.uint8), FakeTorch.uint8)
        return FakeTensor(self.value.reshape(shape), self.dtype)
    def unsqueeze(self, dim): return FakeTensor(np.expand_dims(self.value, dim), self.dtype)
    def contiguous(self): return self
    def all(self): return FakeTensor(self.value.all())
    def float(self): return FakeTensor(self.value.astype(np.float32), "torch.float32")
    def type(self, dtype):
        if dtype == FakeTorch.float32:
            return FakeTensor(self.value.astype(np.float32), "torch.float32")
        raise TypeError(f"unsupported fake dtype {dtype}")
    def softmax(self, dim=-1):
        if FakeTensor.softmax_override is not None:
            return FakeTensor(FakeTensor.softmax_override)
        shifted = self.value - np.max(self.value, axis=dim, keepdims=True)
        values = np.exp(shifted); return FakeTensor(values / values.sum(axis=dim, keepdims=True))
    def gather(self, dim, indices):
        return FakeTensor(np.take_along_axis(self.value, indices.value.astype(np.int64), axis=dim))
    def sum(self, dim=None, keepdim=False):
        if dim is None: return FakeTensor(self.value.sum())
        return FakeTensor(self.value.sum(axis=dim, keepdims=keepdim))
    def __add__(self, other): return FakeTensor(self.value + getattr(other, "value", other))
    def __truediv__(self, other): return FakeTensor(self.value / getattr(other, "value", other))
    def __mul__(self, other): return FakeTensor(self.value * getattr(other, "value", other))
    def __invert__(self): return FakeTensor(~self.value)
    def __getitem__(self, key):
        if isinstance(key, FakeTensor): key = key.value
        return FakeTensor(self.value[key])


class FakeTorch:
    linear_override = None
    float32 = "torch.float32"
    uint8 = "torch.uint8"
    topk_sorted_calls = []
    @staticmethod
    def is_tensor(value): return isinstance(value, FakeTensor)
    @staticmethod
    def isfinite(value): return FakeTensor(np.isfinite(value.value))
    @staticmethod
    def isnan(value): return FakeTensor(np.isnan(value.value))
    @staticmethod
    def isposinf(value): return FakeTensor(np.isposinf(value.value))
    @staticmethod
    def isneginf(value): return FakeTensor(np.isneginf(value.value))
    @staticmethod
    def nonzero(value, as_tuple=False): return FakeTensor(np.argwhere(value.value))
    @staticmethod
    def topk(value, k, dim=-1, sorted=True):
        FakeTorch.topk_sorted_calls.append(sorted)
        indices = np.argsort(value.value, axis=dim)[..., ::-1][..., :k]
        return FakeTensor(np.take_along_axis(value.value, indices, axis=dim)), FakeTensor(indices)
    @staticmethod
    def equal(left, right): return np.array_equal(left.value, right.value)


def fake_linear(hidden, weight):
    if FakeTorch.linear_override is not None:
        return FakeTensor(FakeTorch.linear_override)
    return FakeTensor(hidden.value @ weight.value.T)


FakeTorch.nn = type("NN", (), {"functional": type("Functional", (), {
    "linear": staticmethod(fake_linear)})()})()


CONTRACT_PATH = ROOT / "scripts/longcat-next/core-accepted-contract-v2.json"
IMPLEMENTATIONS = [ROOT / "scripts/longcat-next" / name for name in (
    "make-reference-fixtures.py", "core_reference.py", "core_reference_v2.py",
    "checkpoint-shards-v2.json", "core-accepted-contract-v2.json")]


def contract_arrays(precision="bf16"):
    contract = v2.load_accepted_contract(CONTRACT_PATH)
    expected = v2.expected_array_contract(precision, contract)
    activation = np.float32 if precision == "bf16" else np.float16
    arrays = {}
    for name, rule in expected.items():
        case = name.split("/", 1)[0]
        shape = tuple(1 if dim == "tokens" else 9 if dim == "prompt_plus_generated" else dim
                      for dim in rule["shape"])
        dtype = np.int64 if rule["dtype"] == "int64" else (
            np.float32 if rule["dtype"] == "float32" else activation)
        arrays[name] = np.zeros(shape, dtype=dtype)
    return arrays


def write_valid_worker(run_dir, run_index, precision="bf16", backend="default"):
    arrays = contract_arrays(precision)
    np.savez(run_dir / "arrays.npz", **arrays)
    identities = {path.name: v2.sha256_file(path) for path in IMPLEMENTATIONS}
    provenance = {name: None for name in v2.PROVENANCE_REQUIRED_FIELDS}
    provenance.update({"run_index": run_index, "process_id": 1000 + run_index,
                       "requested_precision": precision,
                       "effective_precision": ("torch.bfloat16" if precision == "bf16" else "torch.float16"),
                       "requested_attention_backend": backend,
                       "effective_attention_backend": {"requested_backend": backend,
                                                       "effective_implementation": "sdpa"},
                       "script_sha256": identities,
                       "shard_manifest_sha256": identities["checkpoint-shards-v2.json"],
                       "start_utc": "2026-01-01T00:00:00Z", "end_utc": "2026-01-01T00:01:00Z"})
    provenance["sdpa_backend_controls"] = {"effective_kernel_identity": "synthetic"}
    metadata = {"schema_version": 2, "kind": "longcat-next-core-reference",
                "precision": precision, "checkpoint": {"revision": v2.PINNED_MODEL_REVISION},
                "source_weight_validation": {"all_finite": True},
                "whole_candidate_validation": {"all_finite": True}, "provenance": provenance,
                "arrays": {name: v2.array_metadata_for(value) for name, value in arrays.items()}}
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="ascii")
    (run_dir / "worker-reproducibility.json").write_text(
        json.dumps({"schema_version": 2, "repeat_count": 1}), encoding="ascii")
    (run_dir / "validation.json").write_text(
        json.dumps({"all_finite": True, "contract_valid": True}), encoding="ascii")
    return arrays


def synthetic_router_model(scaling=1.0):
    model = Model()
    router = Module()
    router.classifier = type("Classifier", (), {
        "weight": FakeTensor(np.eye(3)), "bias": None})()
    router.config = type("Config", (), {"hidden_size": 3})()
    router.e_score_correction_bias = FakeTensor(np.zeros(3, dtype=np.float32))
    router.top_k = 2
    router.n_routed_experts = 3
    router.routed_scaling_factor = scaling
    router.router_bias = False
    model.model.named_modules = lambda: [("layers.4.mlp.router", router)]
    return model, router


class CoreV2Tests(unittest.TestCase):
    def setUp(self):
        self.original_router_runtime_identity = v2.router_runtime_identity
        v2.router_runtime_identity = lambda module: {
            "runtime_router_class": "LongcatFlashTopkRouter",
            "runtime_router_module": "transformers.models.longcat_flash.modeling_longcat_flash",
            "transformers_version": "4.57.6",
            "runtime_router_source_path": "/synthetic/modeling_longcat_flash.py",
            "runtime_router_source_sha256": "0" * 64,
            "replay_variant": "transformers-4.57.6-unconditional-scale",
        }

    def tearDown(self):
        v2.router_runtime_identity = self.original_router_runtime_identity
        FakeTorch.linear_override = None
        FakeTensor.softmax_override = None

    def test_numpy_nonfinite_is_never_reproducible(self):
        for bad in (np.array([np.nan], np.float32), np.array([np.inf], np.float32),
                    np.array([-np.inf], np.float32)):
            with self.assertRaises(v2.V2Error): v2.validate_numpy_array("bad", bad)
            with self.assertRaises(v2.V2Error):
                v2.compare_independent_runs([{"x": bad}] * 2)
        with self.assertRaises(v2.V2Error):
                v2.compare_independent_runs([{"x": np.zeros(1, np.float32)},
                                         {"x": np.array([np.nan], np.float32)}])

    def test_recursive_first_nonfinite_report_identifies_exact_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            checker = v2.TorchFiniteChecker(temporary, "case", "eager", 4, FakeTorch)
            with self.assertRaisesRegex(v2.V2Error, "physical_block=7"):
                checker.check({"nested": [FakeTensor([1.0, np.nan])]},
                              module_name="model.model.layers.3.mlp",
                              logical_layer=3, physical_block=7,
                              operation="forward", role="output")
            report = json.loads((Path(temporary) / "first-nonfinite.json").read_text())
            self.assertEqual(report["module_name"], "model.model.layers.3.mlp")
            self.assertEqual(report["physical_block"], 7)
            self.assertEqual(report["first_affected_indices"], [[1]])
            trace = json.loads((Path(temporary) / "finite-trace.json").read_text())
            self.assertEqual(trace["first_nonfinite"]["module_name"], report["module_name"])
            self.assertGreaterEqual(len(trace["checks"]), 1)

    def test_all_28_physical_boundaries_are_checked_in_order(self):
        model, recorder = Model(), Recorder()
        v2.install_trunk_finite_hooks(model, recorder, serialize_blocks=())
        for layer in model.model.layers:
            layer.input_layernorm[1].pre[0](layer.input_layernorm[1], (object(),))
            layer.post[0](layer, (), object())
        self.assertEqual(recorder.blocks, list(range(28)))

    def _run_invalid_router(self, temporary, original_weights, scaling=1.0,
                            original_indices=None):
        model, router = synthetic_router_model(scaling)
        checker = v2.TorchFiniteChecker(temporary, "case", "default", 0, FakeTorch)
        v2.install_trunk_finite_hooks(model, checker, serialize_blocks=())
        hidden = FakeTensor([[1.0, 2.0, 3.0]])
        router.pre[0](router, (hidden,))
        indices = (FakeTorch.topk(FakeTensor(np.zeros((1, 3))), 2,
                                  dim=-1, sorted=False)[1]
                   if original_indices is None else FakeTensor(original_indices))
        with self.assertRaises(v2.V2Error):
            router.post[0](router, (hidden,), (indices, FakeTensor(original_weights)))
        return json.loads((Path(temporary) / "first-nonfinite.json").read_text()), \
            json.loads((Path(temporary) / "finite-trace.json").read_text())

    def test_shortcut_router_maps_layer_four_to_physical_block_eight(self):
        with tempfile.TemporaryDirectory() as temporary:
            FakeTorch.topk_sorted_calls = []
            FakeTorch.linear_override = [[np.nan, np.nan, np.nan]]
            FakeTensor.softmax_override = None
            failure, trace = self._run_invalid_router(temporary, [[np.nan, np.nan]])
            self.assertEqual((failure["logical_layer"], failure["physical_block"]), (4, 8))
            self.assertEqual(failure["operation"], "router_logits")
            roles = [row.get("role") for row in trace["checks"]]
            self.assertIn("topk_weights", roles)
            self.assertTrue(any(row.get("operation") == "router_input" for row in trace["checks"]))
            self.assertEqual(FakeTorch.topk_sorted_calls[-1], False)
            configuration = next(row for row in trace["checks"]
                                 if row.get("operation") == "router_configuration")
            self.assertEqual(configuration["top_k"], 2)
            self.assertEqual(configuration["n_routed_experts"], 3)
            self.assertEqual(configuration["routed_scaling_factor"], 1.0)
            self.assertFalse(configuration["router_bias"])
            self.assertFalse(configuration["classifier_bias_present"])
            self.assertEqual(configuration["config_hidden_size"], 3)
            self.assertFalse(configuration["norm_topk_prob_attribute_present"])
            self.assertEqual(configuration["transformers_version"], "4.57.6")
            self.assertEqual(configuration["replay_variant"],
                             "transformers-4.57.6-unconditional-scale")
            self.assertEqual(configuration["runtime_router_source_path"],
                             "/synthetic/modeling_longcat_flash.py")
            self.assertEqual(configuration["runtime_router_source_sha256"], "0" * 64)
            FakeTorch.linear_override = None

    def test_router_softmax_is_exact_first_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            FakeTensor.softmax_override = [[np.nan, np.nan, np.nan]]
            failure, _ = self._run_invalid_router(temporary, [[np.nan, np.nan]])
            self.assertEqual(failure["operation"], "softmax_scores")

    def test_pinned_router_sequence_scales_without_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            FakeTorch.linear_override = None
            FakeTensor.softmax_override = None
            failure, trace = self._run_invalid_router(temporary, [[np.inf, np.inf]], scaling=np.inf)
            self.assertEqual(failure["operation"], "topk_weights_scaled")
            operations = [row.get("operation") for row in trace["checks"]]
            for operation in ("router_input", "classifier_weight", "e_score_correction_bias",
                              "router_logits", "softmax_scores", "scores_for_choice",
                              "topk_indices", "topk_weights_gathered", "topk_weights_scaled"):
                self.assertIn(operation, operations)
            self.assertNotIn("topk_denominator", operations)
            self.assertNotIn("topk_weights_normalized", operations)

    def test_router_logits_is_exact_first_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            FakeTorch.linear_override = [[np.nan, np.nan, np.nan]]
            failure, _ = self._run_invalid_router(temporary, [[np.nan, np.nan]])
            self.assertEqual(failure["operation"], "router_logits")

    def test_router_replay_mismatch_refuses_attribution(self):
        with tempfile.TemporaryDirectory() as temporary:
            FakeTorch.linear_override = [[np.nan, np.nan, np.nan]]
            model, router = synthetic_router_model()
            checker = v2.TorchFiniteChecker(temporary, "case", "default", 0, FakeTorch)
            v2.install_trunk_finite_hooks(model, checker, serialize_blocks=())
            hidden = FakeTensor([[1.0, 2.0, 3.0]])
            router.pre[0](router, (hidden,))
            # Replay order is [2, 1]; preserve the same IDs and count but
            # reverse them to prove ordering is part of exact reproduction.
            wrong_indices = FakeTensor([[1, 2]])
            with self.assertRaisesRegex(v2.V2Error, "replay mismatch"):
                router.post[0](router, (hidden,), (wrong_indices, FakeTensor([[np.nan, np.nan]])))
            trace = json.loads((Path(temporary) / "finite-trace.json").read_text())
            mismatch = next(row for row in trace["checks"]
                            if row.get("operation") == "router_replay_mismatch")
            self.assertIn("indices_order", mismatch["reasons"])
            FakeTorch.linear_override = None

    def test_router_replay_rejects_different_mixed_finite_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            FakeTorch.linear_override = None
            FakeTensor.softmax_override = [[np.nan, 0.2, 0.3]]
            # NumPy's test top-k selects the NaN and 0.3 entries. The original
            # has the same one-NaN aggregate, but at the opposite output index.
            self._run_invalid_router(
                temporary, [[1.0, np.nan]],
                original_indices=[[0, 2]])
            trace = json.loads((Path(temporary) / "finite-trace.json").read_text())
            mismatch = next(row for row in trace["checks"]
                            if row.get("operation") == "router_replay_mismatch")
            self.assertIn("finite_mask", mismatch["reasons"])
            FakeTensor.softmax_override = None

    def test_router_instrumentation_error_persists_both_failure_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, router = synthetic_router_model()
            del router.config
            checker = v2.TorchFiniteChecker(temporary, "case", "default", 0, FakeTorch)
            v2.install_trunk_finite_hooks(model, checker, serialize_blocks=())
            hidden = FakeTensor([[1.0, 2.0, 3.0]])
            router.pre[0](router, (hidden,))
            indices = FakeTensor([[2, 1]])
            with self.assertRaisesRegex(v2.V2Error, "internal attribution failed"):
                router.post[0](router, (hidden,),
                               (indices, FakeTensor([[np.nan, np.nan]])))
            first = Path(temporary) / "first-nonfinite.json"
            trace_path = Path(temporary) / "finite-trace.json"
            self.assertTrue(first.is_file())
            self.assertTrue(trace_path.is_file())
            first_report = json.loads(first.read_text())
            self.assertEqual(first_report["role"], "topk_weights")
            trace = json.loads(trace_path.read_text())
            error = next(row for row in trace["checks"]
                         if row.get("operation") == "router_instrumentation_error")
            self.assertEqual(error["attribution_status"], "replay_failed")
            self.assertEqual(error["exception_type"], "AttributeError")

    def test_sdpa_numpy_causal_mask_scale_and_gqa(self):
        q = np.array([[[[1., 0.], [0., 1.]], [[1., 0.], [0., 1.]]]])
        k = np.array([[[[1., 0.], [0., 1.]]]])
        v = np.array([[[[2., 3.], [5., 7.]]]])
        out = v2.manual_sdpa_f32_numpy(q, k, v, is_causal=True,
                                       scale=0.5, enable_gqa=True)
        self.assertEqual(out.shape, q.shape)
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_array_equal(out[..., 0, :], np.array([[[2., 3.], [2., 3.]]]))
        masked = v2.manual_sdpa_f32_numpy(q[:, :1], k, v,
                                          attn_mask=np.array([[True, False]]), scale=1.0)
        np.testing.assert_allclose(masked, np.array([[[[2., 3.], [2., 3.]]]]))

    def test_attention_mode_configuration(self):
        for mode in v2.ATTENTION_BACKENDS:
            config = type("Config", (), {"_attn_implementation": "sdpa"})()
            result = v2.configure_attention_backend(config, mode)
            self.assertEqual(result["requested_backend"], mode)
            if mode == "eager": self.assertEqual(config._attn_implementation, "eager")

    def test_sdpa_wrapper_restores_after_success_and_exception(self):
        class Checker:
            def check(self, value, **context): pass
        def original(q, k, value, **kwargs): return value
        functional = type("Functional", (), {"scaled_dot_product_attention": staticmethod(original)})()
        torch = type("Torch", (), {"nn": type("NN", (), {"functional": functional})()})()
        with v2.instrument_sdpa(torch, Checker(), "default"):
            functional.scaled_dot_product_attention(1, 2, 3)
        self.assertIs(functional.scaled_dot_product_attention, original)
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with v2.instrument_sdpa(torch, Checker(), "default"):
                raise RuntimeError("boom")
        self.assertIs(functional.scaled_dot_product_attention, original)

    def test_two_attention_context_assigns_sdpa_to_physical_block(self):
        model = Model()
        attention = [Module(), Module()]
        model.model.named_modules = lambda: [
            ("layers.7.self_attn.0", attention[0]),
            ("layers.7.self_attn.1", attention[1])]
        class Checker:
            def __init__(self): self.rows = []
            def check(self, value, **context): self.rows.append(context)
        checker = Checker()
        v2.install_trunk_finite_hooks(model, checker, serialize_blocks=())
        def original(q, k, value, **kwargs): return value
        functional = type("Functional", (), {"scaled_dot_product_attention": staticmethod(original)})()
        torch = type("Torch", (), {"nn": type("NN", (), {"functional": functional})()})()
        with v2.instrument_sdpa(torch, checker, "default"):
            for module in attention:
                module.pre[0](module, (1,))
                functional.scaled_dot_product_attention(1, 2, 3)
                module.post[0](module, (), 3)
        sdpa_outputs = [row for row in checker.rows
                        if row.get("operation") == "scaled_dot_product_attention" and
                        row.get("role") == "output"]
        self.assertEqual([row["physical_block"] for row in sdpa_outputs], [14, 15])
        self.assertEqual(sdpa_outputs[1]["module_name"], "model.model.layers.7.self_attn.1")

    def test_finite_trunk_nonfinite_lm_head_reports_lm_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Module(); model.model = type("Trunk", (), {"norm": Module()})(); model.lm_head = Module()
            checker = v2.TorchFiniteChecker(temporary, "case", "default", 0, FakeTorch)
            v2.install_output_finite_hooks(model, checker)
            model.model.norm.post[0](model.model.norm, (), FakeTensor([1.0]))
            with self.assertRaisesRegex(v2.V2Error, "model.lm_head"):
                model.lm_head.post[0](model.lm_head, (), FakeTensor([np.nan]))
            report = json.loads((Path(temporary) / "first-nonfinite.json").read_text())
            self.assertEqual(report["module_name"], "model.lm_head")

    def test_generation_step_two_nonfinite_score_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            checker = v2.TorchFiniteChecker(temporary, "generation_prompt_1", "default", 0, FakeTorch)
            checker.check(FakeTensor([1.0]), module_name="generation.scores",
                          operation="generation", role="score", prompt="prompt_1", generation_step=0)
            with self.assertRaises(v2.V2Error):
                checker.check(FakeTensor([np.nan]), module_name="generation.scores",
                              operation="generation", role="score", prompt="prompt_1",
                              generation_step=2)
            report = json.loads((Path(temporary) / "first-nonfinite.json").read_text())
            self.assertEqual((report["prompt"], report["generation_step"]), ("prompt_1", 2))

    def test_manifest_is_pinned_official_lfs_metadata(self):
        manifest = v2.load_shard_manifest(
            ROOT / "scripts/longcat-next/checkpoint-shards-v2.json")
        self.assertEqual(len(manifest["shards"]), 15)
        self.assertEqual(sum(row["bytes"] for row in manifest["shards"]), 150827115056)

    def test_bf16_safetensors_source_scan_preserves_dtype_and_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); shards = []; tensors = {}
            values = ([1.0], [np.nan], [np.inf], [-np.inf])
            bf16_bits = (0x3f80, 0x7fc0, 0x7f80, 0xff80)
            for index in range(15):
                path = root / f"model-{index + 1:05d}-of-00015.safetensors"
                value = values[index] if index < len(values) else [float(index)]
                # A real Safetensors envelope declaring BF16; the injected opener
                # avoids requiring a heavyweight Torch install in this unit test.
                header = json.dumps({"weight": {"dtype": "BF16", "shape": [1],
                                                  "data_offsets": [0, 2]}}).encode("utf-8")
                header += b" " * ((8 - len(header) % 8) % 8)
                raw = bf16_bits[index] if index < len(bf16_bits) else 0x4000
                path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<H", raw))
                tensors[path.name] = FakeTensor(value, "torch.bfloat16")
                shards.append({"filename": path.name, "bytes": path.stat().st_size,
                               "lfs_sha256": v2.sha256_file(path)})
            manifest = {"schema_version": 1, "repository": v2.PINNED_MODEL_REPOSITORY,
                        "revision": v2.PINNED_MODEL_REVISION, "shards": shards}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="ascii")
            class Opened:
                def __init__(self, tensor): self.tensor = tensor
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def keys(self): return ["weight"]
                def get_tensor(self, name): return self.tensor
            def opener(path, framework, device):
                self.assertEqual((framework, device), ("pt", "cpu"))
                return Opened(tensors[Path(path).name])
            original = dict(tensors)
            tensors.update({name: FakeTensor([1.0], "torch.bfloat16") for name in tensors})
            finite = v2.verify_and_scan_source(
                root, manifest_path, root / "finite-reports", opener, FakeTorch)
            self.assertEqual(finite["dtype_counts"], {"torch.bfloat16": 15})
            summary_path = root / "finite-reports/source-scan-summary.json"
            self.assertTrue(summary_path.is_file())
            summary = json.loads(summary_path.read_text())
            self.assertEqual(len(summary["verified_shards"]), 15)
            tensors[shards[1]["filename"]] = original[shards[1]["filename"]]
            with self.assertRaises(v2.V2Error):
                v2.verify_and_scan_source(root, manifest_path, root / "reports", opener, FakeTorch)
            self.assertFalse((root / "reports/source-scan-summary.json").exists())
            failure = json.loads((root / "reports/source-scan/first-nonfinite-source.json").read_text())
            self.assertEqual(failure["dtype"], "torch.bfloat16")
            self.assertEqual(failure["nan_count"], 1)
            for filename, count_key in ((shards[2]["filename"], "positive_infinity_count"),
                                        (shards[3]["filename"], "negative_infinity_count")):
                tensors.update({name: FakeTensor([1.0], "torch.bfloat16") for name in tensors})
                tensors[filename] = original[filename]
                with self.assertRaises(v2.V2Error):
                    v2.verify_and_scan_source(root, manifest_path,
                                              root / ("reports-" + filename), opener, FakeTorch)
                report = json.loads((root / ("reports-" + filename) / "source-scan" /
                                     "first-nonfinite-source.json").read_text())
                self.assertEqual(report[count_key], 1)

    def test_independent_workers_are_retained_and_promoted_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "candidate"
            commands = [["fake", "--output-dir", "{run_dir}"] for _ in range(2)]
            def runner(command, **kwargs):
                run_dir = Path(command[command.index("--output-dir") + 1])
                write_valid_worker(run_dir, int(run_dir.name[-2:]))
                return subprocess.CompletedProcess(command, 0)
            v2.orchestrate_workers(final, commands, "bf16", "default",
                                   CONTRACT_PATH, IMPLEMENTATIONS, runner=runner)
            self.assertTrue((final / "candidate-validation.json").is_file())
            self.assertTrue((final / "runs/run-00/arrays.npz").is_file())
            self.assertTrue((final / "runs/run-01/arrays.npz").is_file())
            report = json.loads((final / "candidate-validation.json").read_text())
            self.assertTrue(report["whole_candidate_finite"])
            self.assertTrue((final / "longcat-next-core-bf16.npz").is_file())
            self.assertTrue((final / "longcat-next-core-bf16.json").is_file())
            self.assertTrue((final / "longcat-next-core-reproducibility.json").is_file())
            with np.load(final / "longcat-next-core-bf16.npz", allow_pickle=False) as root_npz:
                self.assertEqual(len(root_npz.files), 433)
            root_metadata = json.loads((final / "longcat-next-core-bf16.json").read_text())
            self.assertEqual(root_metadata["kind"], "longcat-next-core-reference")
            validated = v2.validate_candidate(final, CONTRACT_PATH, IMPLEMENTATIONS)
            self.assertTrue(validated["valid"])
            cli = subprocess.run([
                __import__("sys").executable,
                str(ROOT / "scripts/longcat-next/make-reference-fixtures.py"),
                "--mode", "core-validate", "--candidate-dir", str(final)],
                text=True, capture_output=True)
            self.assertEqual(cli.returncode, 0, cli.stderr)
            self.assertIs(json.loads(cli.stdout)["valid"], True)
            (final / "longcat-next-core-bf16.npz").write_bytes(b"corrupt")
            invalid = subprocess.run(cli.args, text=True, capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIs(json.loads(invalid.stdout)["valid"], False)

    def test_empty_worker_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "candidate"
            def runner(command, **kwargs):
                run_dir = Path(command[command.index("--output-dir") + 1])
                write_valid_worker(run_dir, int(run_dir.name[-2:]))
                (run_dir / "metadata.json").write_text("{}", encoding="ascii")
                return subprocess.CompletedProcess(command, 0)
            with self.assertRaises(v2.V2Error):
                v2.orchestrate_workers(final, [["x", "--output-dir", "{run_dir}"]] * 2,
                                       "bf16", "default", CONTRACT_PATH,
                                       IMPLEMENTATIONS, runner=runner)
            self.assertFalse(final.exists())

    def test_count_correct_wrong_contract_name_is_rejected(self):
        arrays = contract_arrays()
        arrays["arbitrary/replacement"] = arrays.pop(next(iter(arrays)))
        with self.assertRaisesRegex(v2.V2Error, "accepted array names differ"):
            v2.validate_accepted_arrays(
                arrays, "bf16", v2.load_accepted_contract(CONTRACT_PATH))

    def test_failed_worker_never_creates_final_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "candidate"
            def failed(command, **kwargs): return subprocess.CompletedProcess(command, 9)
            with self.assertRaises(v2.V2Error):
                v2.orchestrate_workers(final, [["bad"], ["bad"]], "bf16", "default",
                                       CONTRACT_PATH, IMPLEMENTATIONS, runner=failed)
            self.assertFalse(final.exists())

    def test_accepted_contract_is_433(self):
        self.assertEqual(v2.EXPECTED_ACCEPTED_ARRAYS, 433)
        self.assertIn("shard_manifest_sha256", v2.PROVENANCE_REQUIRED_FIELDS)
        self.assertIn("effective_attention_backend", v2.PROVENANCE_REQUIRED_FIELDS)


if __name__ == "__main__": unittest.main()
