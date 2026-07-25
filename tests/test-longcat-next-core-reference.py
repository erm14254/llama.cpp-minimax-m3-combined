#!/usr/bin/env python3
"""Checkpoint-free tests for the LongCat-Next core reference fixture tooling."""

import hashlib
import importlib.util
import json
import os
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "longcat_next_core_reference", ROOT / "scripts/longcat-next/core_reference.py")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


class CheckpointValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in core.REQUIRED_FILES:
            (self.root / name).write_text("{}" if name.endswith(".json") else "# test\n", encoding="ascii")
        config = {"text_vocab_size": 131072,
                  "text_vocab_plus_multimodal_special_token_size": 131125,
                  "vocab_size": 282624}
        (self.root / "config.json").write_text(json.dumps(config), encoding="ascii")
        generation = {"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 3,
                      "transformers_version": "4.57.6",
                      "visual_generation_config": {"custom_params": {
                          "token_h": 37, "token_w": 37,
                          "anyres_prefix": "<longcat_img_token_size>{h} {w}</longcat_img_token_size>"}},
                      "audio_generation_config": {"audio_parallel_decoding": False}}
        (self.root / "generation_config.json").write_text(json.dumps(generation), encoding="ascii")
        self.shards = [f"model-{i:05d}-of-00015.safetensors" for i in range(1, 16)]
        base = core.EXPECTED_SHARD_BYTES // 15
        for index, name in enumerate(self.shards):
            size = base if index < 14 else core.EXPECTED_SHARD_BYTES - base * 14
            with (self.root / name).open("wb") as stream:
                stream.truncate(size)
        names = {}
        prefixes = ["model.visual_tokenizer.visual_model.", "visual_head.",
                    "model.audio_tokenizer.audio_model.", "audio_head.", "model.layers."]
        for index in range(core.EXPECTED_TENSORS):
            name = f"{prefixes[index % len(prefixes)]}synthetic_{index}"
            names[name] = self.shards[index % len(self.shards)]
        self.index = {"metadata": {"total_size": core.EXPECTED_PAYLOAD}, "weight_map": names}
        self.write_index()
        self.old_identities = core.EXPECTED_IDENTITIES
        core.EXPECTED_IDENTITIES = {
            name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
            for name in core.EXPECTED_IDENTITIES}

    def tearDown(self):
        core.EXPECTED_IDENTITIES = self.old_identities
        self.temp.cleanup()

    def write_index(self):
        (self.root / "model.safetensors.index.json").write_text(json.dumps(self.index), encoding="ascii")

    def test_complete_checkpoint_validation(self):
        report = core.validate_checkpoint(self.root)
        self.assertEqual(report["tensor_count"], 13450)
        self.assertEqual(report["shard_count"], 15)
        self.assertEqual(report["total_shard_file_bytes"], 150827115056)

    def test_missing_shard(self):
        (self.root / self.shards[0]).unlink()
        with self.assertRaisesRegex(core.CoreFixtureError, "missing referenced shards"):
            core.validate_checkpoint(self.root)

    def test_wrong_shard_count(self):
        replacement = self.shards[1]
        self.index["weight_map"] = {name: replacement if shard == self.shards[0] else shard
                                    for name, shard in self.index["weight_map"].items()}
        self.write_index()
        with self.assertRaisesRegex(core.CoreFixtureError, "exactly 15"):
            core.validate_checkpoint(self.root)

    def test_wrong_aggregate_shard_size(self):
        with (self.root / self.shards[-1]).open("r+b") as stream:
            stream.truncate((self.root / self.shards[-1]).stat().st_size - 1)
        with self.assertRaisesRegex(core.CoreFixtureError, "total shard-file bytes"):
            core.validate_checkpoint(self.root)

    def test_wrong_index_payload_size(self):
        self.index["metadata"]["total_size"] -= 1
        self.write_index()
        with self.assertRaisesRegex(core.CoreFixtureError, "payload bytes"):
            core.validate_checkpoint(self.root)

    def test_unexpected_mtp(self):
        old = next(iter(self.index["weight_map"]))
        shard = self.index["weight_map"].pop(old)
        self.index["weight_map"]["model.mtp.unexpected.weight"] = shard
        self.write_index()
        with self.assertRaisesRegex(core.CoreFixtureError, "model.mtp"):
            core.validate_checkpoint(self.root)

    def test_wrong_vocabulary_extent(self):
        config = json.loads((self.root / "config.json").read_text())
        config["vocab_size"] = 1
        (self.root / "config.json").write_text(json.dumps(config), encoding="ascii")
        core.EXPECTED_IDENTITIES["config.json"] = hashlib.sha256(
            (self.root / "config.json").read_bytes()).hexdigest()
        with self.assertRaisesRegex(core.CoreFixtureError, "vocab_size"):
            core.validate_checkpoint(self.root)

    def test_missing_generation_config(self):
        (self.root / "generation_config.json").unlink()
        with self.assertRaisesRegex(core.CoreFixtureError, "generation_config.json"):
            core.validate_checkpoint(self.root)

    def test_wrong_generation_config(self):
        generation = json.loads((self.root / "generation_config.json").read_text())
        generation["visual_generation_config"]["custom_params"]["token_h"] = 36
        (self.root / "generation_config.json").write_text(json.dumps(generation), encoding="ascii")
        core.EXPECTED_IDENTITIES["generation_config.json"] = hashlib.sha256(
            (self.root / "generation_config.json").read_bytes()).hexdigest()
        with self.assertRaisesRegex(core.CoreFixtureError, "token_h"):
            core.validate_checkpoint(self.root)


class CoreHelperTests(unittest.TestCase):
    def test_dependency_preflight_reports_versions_and_missing_imports(self):
        expected = core.EXPECTED_DEPENDENCIES
        core.EXPECTED_DEPENDENCIES = {
            "available": ("available-dist", "1.0"),
            "missing": ("missing-dist", None),
        }
        def version(name):
            if name == "available-dist":
                return "1.0"
            raise importlib.metadata.PackageNotFoundError(name)
        def importer(name):
            if name == "available":
                return object()
            raise ImportError("not installed")
        try:
            with mock.patch.object(core.importlib.metadata, "version", side_effect=version), \
                 mock.patch.object(core.importlib, "import_module", side_effect=importer):
                report = core.dependency_preflight()
        finally:
            core.EXPECTED_DEPENDENCIES = expected
        self.assertFalse(report["ok"])
        self.assertTrue(report["packages"]["available"]["import_ok"])
        self.assertIn("import failed", report["packages"]["missing"]["error"])

    def test_network_disabled_loading_arguments(self):
        fake_torch = types.SimpleNamespace(bfloat16="bf16", float16="f16")
        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            kwargs = core.loading_kwargs("bf16", "auto", "offload", "220GiB", "88GiB")
        self.assertTrue(kwargs["local_files_only"])
        self.assertTrue(kwargs["trust_remote_code"])
        self.assertTrue(kwargs["use_safetensors"])
        self.assertTrue(kwargs["low_cpu_mem_usage"])
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(kwargs["device_map"], "auto")
        self.assertEqual(kwargs["max_memory"], {"cpu": "220GiB", 0: "88GiB"})

    def test_precision_option_validation(self):
        for precision in ("bf16", "f16"):
            core.validate_core_options(precision, "auto", 2, 1024)
        with self.assertRaisesRegex(core.CoreFixtureError, "precision"):
            core.validate_core_options("f32", "auto", 2, 1024)

    def test_deterministic_npz_and_json_serialization(self):
        arrays = {"z": np.array([3, 4], dtype=np.int64),
                  "a": np.array([1.5], dtype=np.float32)}
        first = core.deterministic_npz_bytes(arrays)
        second = core.deterministic_npz_bytes(arrays)
        self.assertEqual(first, second)
        with zipfile.ZipFile(__import__("io").BytesIO(first)) as archive:
            self.assertEqual(archive.namelist(), ["a.npy", "z.npy"])
        with tempfile.TemporaryDirectory() as temp:
            one = core.write_core_outputs(Path(temp) / "one", "fixture", arrays,
                                          {"schema_version": 1}, {"repeat_count": 1}, 100000)
            two = core.write_core_outputs(Path(temp) / "two", "fixture", arrays,
                                          {"schema_version": 1}, {"repeat_count": 1}, 100000)
            for kind in one:
                self.assertEqual(one[kind].read_bytes(), two[kind].read_bytes())

    def test_output_size_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            arrays = {"activation": np.ones(100, dtype=np.float32)}
            with self.assertRaisesRegex(core.CoreFixtureError, "above limit"):
                core.write_core_outputs(temp, "fixture", arrays, {}, {}, 10)

    def test_model_weight_filename_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "accidental.safetensors").write_bytes(b"")
            with self.assertRaisesRegex(core.CoreFixtureError, "model-weight"):
                core.write_core_outputs(temp, "fixture", {"x": np.ones(1)}, {}, {}, 10000)

    def test_reproducibility_exact_and_numeric_reporting(self):
        first = {"input_ids/case": np.array([1, 2], dtype=np.int64),
                 "activation": np.array([1.0, 2.0], dtype=np.float32)}
        second = {"input_ids/case": np.array([1, 2], dtype=np.int64),
                  "activation": np.array([1.0, 2.25], dtype=np.float32)}
        report = core.compare_runs([first, second])
        self.assertFalse(report["activation"]["byte_identical"])
        self.assertEqual(report["activation"]["max_absolute_difference"], 0.25)
        self.assertGreater(report["activation"]["max_relative_difference"], 0)
        self.assertIsNone(report["activation"]["comparison_tolerance"])

    def test_reproducibility_rejects_exact_mismatch(self):
        with self.assertRaisesRegex(core.CoreFixtureError, "exact output differs"):
            core.compare_runs([{"greedy_ids/a": np.array([1])},
                               {"greedy_ids/a": np.array([2])}])

    def test_hook_resolution_failure(self):
        with self.assertRaisesRegex(core.CoreFixtureError, "module hook"):
            core.resolve_capture_modules(types.SimpleNamespace())

    def test_greedy_result_contracts(self):
        class Tensor:
            ndim = 2
            def detach(self):
                return self
        sequences = Tensor()
        self.assertIs(core.extract_greedy_sequences(types.SimpleNamespace(sequences=sequences)), sequences)
        self.assertIs(core.extract_greedy_sequences((sequences, None, None, None)), sequences)
        with self.assertRaisesRegex(core.CoreFixtureError, "neither"):
            core.extract_greedy_sequences(sequences)
        rank_one = Tensor()
        rank_one.ndim = 1
        with self.assertRaisesRegex(core.CoreFixtureError, "shape"):
            core.extract_greedy_sequences(types.SimpleNamespace(sequences=rank_one))

    def test_no_weight_end_to_end_contract(self):
        class Status:
            mode = "text"
            def __init__(self, visual, audio):
                self.visual_generation_config = visual
                self.audio_generation_config = audio
            def switch_to(self, mode):
                self.mode = mode
        class GenerationConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                return cls(visual_generation_config={"custom_params": {
                               "token_h": 37, "token_w": 37, "anyres_prefix": "x"}},
                           audio_generation_config={"audio_parallel_decoding": False})
        dynamic = types.ModuleType("mock_longcat_official")
        dynamic.LongcatNextForCausalLMGenerationStatus = Status
        transformers = types.ModuleType("transformers")
        transformers.GenerationConfig = GenerationConfig

        class Model:
            def __call__(self, input_ids, multimodal_generation_status=None, **kwargs):
                if multimodal_generation_status is None or multimodal_generation_status.mode != "text":
                    raise RuntimeError("official text status required")
                logits = np.arange(131125, dtype=np.float32).reshape(1, 1, -1)
                self.last_logits_shape = tuple(logits.shape)
                return types.SimpleNamespace(logits=logits)
        Model.__module__ = dynamic.__name__
        model = Model()
        with self.assertRaisesRegex(RuntimeError, "status required"):
            model(input_ids=np.array([[1]]))
        with mock.patch.dict("sys.modules", {dynamic.__name__: dynamic, "transformers": transformers}):
            context = core.build_text_generation_context(model, "unused")
            output = core.call_text_forward(model, {"input_ids": np.array([[1, 5, 7]])}, context)
            first, _ = core.summarize_forward_logits(output, [0, 2, 131124], True)
            second, _ = core.summarize_forward_logits(output, [0, 2, 131124], True)
        self.assertEqual(model.last_logits_shape, (1, 1, 131125))
        self.assertEqual(first["complete_final_position_logits"].shape, (1, 131125))
        self.assertEqual(first["selected_logits"].shape, (1, 3))
        self.assertEqual(first["argmax_token_id"].item(), 131124)
        class GreedyTensor:
            ndim = 2
            def __init__(self):
                self.value = np.array([[1, 5, 7, 9]], dtype=np.int64)
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.value
        greedy = core.extract_greedy_sequences(
            types.SimpleNamespace(sequences=GreedyTensor())).cpu().numpy()
        first["greedy_ids/case"] = greedy
        second["greedy_ids/case"] = greedy.copy()
        report = core.compare_runs([first, second])
        self.assertTrue(report["greedy_ids/case"]["byte_identical"])
        with tempfile.TemporaryDirectory() as temp:
            outputs = core.write_core_outputs(temp, "mini", first, {}, report, 10 * 1024 * 1024)
            self.assertTrue(outputs["npz"].is_file())


if __name__ == "__main__":
    unittest.main()
