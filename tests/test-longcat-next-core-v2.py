#!/usr/bin/env python3
"""Checkpoint-free schema-v2 generator safety tests."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
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
    def __init__(self, value): self.value = np.asarray(value); self.dtype = self.value.dtype; self.device = "cpu"
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
    def __invert__(self): return FakeTensor(~self.value)
    def __getitem__(self, key):
        if isinstance(key, FakeTensor): key = key.value
        return FakeTensor(self.value[key])


class FakeTorch:
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


class CoreV2Tests(unittest.TestCase):
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

    def test_all_28_physical_boundaries_are_checked_in_order(self):
        model, recorder = Model(), Recorder()
        v2.install_trunk_finite_hooks(model, recorder, serialize_blocks=())
        for layer in model.model.layers:
            layer.input_layernorm[1].pre[0](layer.input_layernorm[1], (object(),))
            layer.post[0](layer, (), object())
        self.assertEqual(recorder.blocks, list(range(28)))

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

    def test_manifest_is_pinned_official_lfs_metadata(self):
        manifest = v2.load_shard_manifest(
            ROOT / "scripts/longcat-next/checkpoint-shards-v2.json")
        self.assertEqual(len(manifest["shards"]), 15)
        self.assertEqual(sum(row["bytes"] for row in manifest["shards"]), 150827115056)

    def test_tiny_safetensors_source_scan_rejects_nonfinite(self):
        from safetensors.numpy import save_file
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); shards = []
            for index in range(15):
                path = root / f"model-{index + 1:05d}-of-00015.safetensors"
                save_file({"weight": np.array([index], np.float32)}, path)
                shards.append({"filename": path.name, "bytes": path.stat().st_size,
                               "lfs_sha256": v2.sha256_file(path)})
            manifest = {"schema_version": 1, "repository": v2.PINNED_MODEL_REPOSITORY,
                        "revision": v2.PINNED_MODEL_REVISION, "shards": shards}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="ascii")
            report = v2.verify_and_scan_source(root, manifest_path, root / "reports")
            self.assertEqual(report["tensor_count"], 15)
            bad = root / shards[0]["filename"]
            save_file({"weight": np.array([np.nan], np.float32)}, bad)
            shards[0].update(bytes=bad.stat().st_size, lfs_sha256=v2.sha256_file(bad))
            manifest_path.write_text(json.dumps(manifest), encoding="ascii")
            with self.assertRaises(v2.V2Error):
                v2.verify_and_scan_source(root, manifest_path, root / "bad-reports")

    def test_independent_workers_are_retained_and_promoted_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "candidate"
            commands = [["fake", "--output-dir", "{run_dir}"] for _ in range(2)]
            def runner(command, **kwargs):
                run_dir = Path(command[command.index("--output-dir") + 1])
                arrays = {f"a{i:03d}": np.array([i], np.float32) for i in range(433)}
                np.savez(run_dir / "arrays.npz", **arrays)
                (run_dir / "metadata.json").write_text("{}", encoding="ascii")
                return subprocess.CompletedProcess(command, 0)
            v2.orchestrate_workers(final, commands, runner=runner)
            self.assertTrue((final / "candidate-validation.json").is_file())
            self.assertTrue((final / "runs/run-00/arrays.npz").is_file())
            self.assertTrue((final / "runs/run-01/arrays.npz").is_file())
            report = json.loads((final / "candidate-validation.json").read_text())
            self.assertTrue(report["whole_candidate_finite"])

    def test_failed_worker_never_creates_final_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "candidate"
            def failed(command, **kwargs): return subprocess.CompletedProcess(command, 9)
            with self.assertRaises(v2.V2Error):
                v2.orchestrate_workers(final, [["bad"], ["bad"]], runner=failed)
            self.assertFalse(final.exists())

    def test_accepted_contract_is_433(self):
        self.assertEqual(v2.EXPECTED_ACCEPTED_ARRAYS, 433)
        self.assertIn("shard_manifest_sha256", v2.PROVENANCE_REQUIRED_FIELDS)
        self.assertIn("effective_attention_backend", v2.PROVENANCE_REQUIRED_FIELDS)


if __name__ == "__main__": unittest.main()
