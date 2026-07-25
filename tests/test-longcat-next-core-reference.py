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

    def test_runtime_profile_records_official_departure(self):
        expected = core.EXPECTED_DEPENDENCIES
        core.EXPECTED_DEPENDENCIES = {"torch": ("torch", "2.6.0")}
        fake_torch = types.SimpleNamespace(__version__="2.7.1", version=types.SimpleNamespace(cuda=None))
        try:
            with mock.patch.object(core.importlib.metadata, "version", return_value="2.7.1"), \
                 mock.patch.object(core.importlib, "import_module", return_value=fake_torch):
                report = core.dependency_preflight("blackwell-compatible", "cpu")
        finally:
            core.EXPECTED_DEPENDENCIES = expected
        self.assertTrue(report["packages"]["torch"]["runtime_profile_allows_departure"])
        self.assertEqual(report["packages"]["torch"]["official_pinned_version"], "2.6.0")

    def test_sm120_rejects_official_pinned_torch(self):
        cuda = types.SimpleNamespace(
            is_available=lambda: True, get_device_capability=lambda index: (12, 0),
            get_arch_list=lambda: ["sm_120"], get_device_name=lambda index: "Blackwell",
            is_bf16_supported=lambda: True)
        fake_torch = types.SimpleNamespace(
            __version__="2.6.0", version=types.SimpleNamespace(cuda="12.6"), cuda=cuda)
        packages = {"torch": {"installed_version": "2.6.0", "version_ok": True},
                    "torchaudio": {"installed_version": "2.6.0"},
                    "torchvision": {"installed_version": "0.21.0"}}
        with mock.patch.object(core.importlib, "import_module", return_value=fake_torch):
            report = core.runtime_probe(packages, "official-pinned", "cuda")
        self.assertFalse(report["ok"])
        self.assertIn("Blackwell", report["error"])

    def test_blackwell_profile_rejects_non_sm120_architecture(self):
        cuda = types.SimpleNamespace(
            is_available=lambda: True, get_device_capability=lambda index: (12, 0),
            get_arch_list=lambda: ["sm_90"], get_device_name=lambda index: "Blackwell",
            is_bf16_supported=lambda: True)
        fake_torch = types.SimpleNamespace(
            __version__="2.7.1", version=types.SimpleNamespace(cuda="12.8"), cuda=cuda)
        packages = {"torch": {"installed_version": "2.7.1"},
                    "torchaudio": {"installed_version": "2.7.1"},
                    "torchvision": {"installed_version": "0.22.1"}}
        with mock.patch.object(core.importlib, "import_module", return_value=fake_torch):
            report = core.runtime_probe(packages, "blackwell-compatible", "cuda")
        self.assertFalse(report["ok"])
        self.assertIn("sm_120", report["error"])

    def test_local_custom_code_preflight_uses_dynamic_loader(self):
        dynamic_utils = types.ModuleType("transformers.dynamic_module_utils")
        calls = []
        def loader(reference, path, **kwargs):
            calls.append((reference, path, kwargs))
            return type(reference.rsplit(".", 1)[-1], (), {})
        dynamic_utils.get_class_from_dynamic_module = loader
        transformers = types.ModuleType("transformers")
        with mock.patch.dict("sys.modules", {
                "transformers": transformers,
                "transformers.dynamic_module_utils": dynamic_utils}):
            report = core.import_local_custom_classes("checkpoint")
        self.assertTrue(report["ok"])
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(row[2]["local_files_only"] for row in calls))

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
        self.assertEqual(kwargs["dtype"], "bf16")
        self.assertNotIn("torch_dtype", kwargs)

    def test_loading_precision_maps_to_non_deprecated_dtype(self):
        fake_torch = types.SimpleNamespace(bfloat16="bf16", float16="f16")
        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            self.assertEqual(core.loading_kwargs("bf16", "cpu")["dtype"], "bf16")
            self.assertEqual(core.loading_kwargs("f16", "cpu")["dtype"], "f16")

    def test_effective_embedding_dtype_provenance_and_mismatch(self):
        fake_torch = types.SimpleNamespace(bfloat16="bf16", float16="f16")
        embedding = types.SimpleNamespace(weight=types.SimpleNamespace(dtype="bf16"))
        model = types.SimpleNamespace(dtype="bf16")
        with mock.patch.object(core, "resolve_capture_modules",
                               return_value={"base_embedding": embedding}):
            report = core.effective_dtype_provenance(model, "bf16", fake_torch)
            self.assertEqual(report, {
                "requested_precision": "bf16", "requested_torch_dtype": "bf16",
                "effective_model_dtype": "bf16", "base_embedding_weight_dtype": "bf16"})
            embedding.weight.dtype = "f16"
            with self.assertRaisesRegex(core.CoreFixtureError, "base embedding"):
                core.effective_dtype_provenance(model, "bf16", fake_torch)

    def test_tokenizer_loading_preserves_backend_and_rejects_regex_patch(self):
        kwargs = core.tokenizer_loading_kwargs(False)
        self.assertIs(kwargs["fix_mistral_regex"], False)
        with self.assertRaisesRegex(core.CoreFixtureError, "forbids fix_mistral_regex=True"):
            core.tokenizer_loading_kwargs(True)
        state = b'{"type":"ByteLevel","add_prefix_space":false}'
        pretokenizer = types.SimpleNamespace(__getstate__=lambda: state)
        backend = types.SimpleNamespace(pre_tokenizer=pretokenizer)
        BloomTokenizer = type("BloomTokenizer", (), {})
        BloomTokenizerFast = type("BloomTokenizerFast", (), {})
        tokenizer = BloomTokenizerFast()
        tokenizer.is_fast = True
        tokenizer.slow_tokenizer_class = BloomTokenizer
        tokenizer.backend_tokenizer = backend
        before = core.tokenizer_backend_pretokenizer_sha256(tokenizer)
        after = core.tokenizer_backend_pretokenizer_sha256(tokenizer)
        self.assertEqual(before, after)
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "tokenizer.json").write_text("{}", encoding="ascii")
            Path(temp, "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "BloomTokenizer"}), encoding="ascii")
            provenance = core.tokenizer_provenance(tokenizer, temp)
        self.assertEqual(provenance["declared_tokenizer_class"], "BloomTokenizer")
        self.assertEqual(provenance["runtime_tokenizer_class"], "BloomTokenizerFast")
        self.assertTrue(provenance["runtime_tokenizer_is_fast"])
        self.assertEqual(provenance["runtime_slow_tokenizer_class"], "BloomTokenizer")
        self.assertIs(provenance["fix_mistral_regex"], False)
        self.assertEqual(provenance["backend_pre_tokenizer_state_sha256"], before)
        self.assertNotIn("use_fast", kwargs)

    def test_tokenizer_provenance_rejects_unrelated_runtime_and_wrong_declaration(self):
        pretokenizer = types.SimpleNamespace(__getstate__=lambda: b"state")
        backend = types.SimpleNamespace(pre_tokenizer=pretokenizer)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            root.joinpath("tokenizer.json").write_text("{}", encoding="ascii")
            root.joinpath("tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "BloomTokenizer"}), encoding="ascii")
            unrelated = type("UnrelatedTokenizerFast", (), {})()
            unrelated.is_fast = True
            unrelated.backend_tokenizer = backend
            with self.assertRaisesRegex(core.CoreFixtureError, "BloomTokenizerFast"):
                core.tokenizer_provenance(unrelated, root)
            root.joinpath("tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "MistralTokenizer"}), encoding="ascii")
            bloom = type("BloomTokenizerFast", (), {})()
            bloom.is_fast = True
            bloom.backend_tokenizer = backend
            with self.assertRaisesRegex(core.CoreFixtureError, "declared tokenizer class"):
                core.tokenizer_provenance(bloom, root)

    def test_direct_and_greedy_prompt_ids_must_match(self):
        for name, ids in (("prompt_0", [1, 7, 9]), ("prompt_1", [1, 8, 10])):
            self.assertEqual(core.require_prompt_ids_match(name, ids, ids), ids)
        with self.assertRaisesRegex(core.CoreFixtureError, "tokenization differ"):
            core.require_prompt_ids_match("prompt_0", [1, 7], [1, 8])

    def test_explicit_greedy_policy_is_copied_and_clears_sampling(self):
        original = types.SimpleNamespace(do_sample=True, temperature=0.7,
                                         top_p=0.8, top_k=40, use_cache=False,
                                         return_dict_in_generate=False)
        config, settings = core.fixture_greedy_generation_config(original, 3)
        self.assertIsNot(config, original)
        self.assertTrue(original.do_sample)
        self.assertEqual(original.temperature, 0.7)
        self.assertEqual(settings, {
            "do_sample": False, "temperature": None, "top_p": None, "top_k": None,
            "max_new_tokens": 3, "use_cache": True,
            "return_dict_in_generate": True})
        for name, value in settings.items():
            self.assertEqual(getattr(config, name), value)
        prompt_configs = [config, config]
        self.assertIs(prompt_configs[0], prompt_configs[1])

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

    def test_case_specific_padding_and_official_positions(self):
        padded = core.prepare_case_inputs("bos_left_zero", [0, 0, 1, 17])
        literal = core.prepare_case_inputs("literal_zero", [19, 0, 29])
        ordinary = core.prepare_case_inputs("maximum_text_token", [7, 11, 131071])
        self.assertEqual(padded["attention_mask"].tolist(), [[0, 0, 1, 1]])
        self.assertEqual(padded["position_ids"].tolist(), [[1, 1, 0, 1]])
        self.assertEqual(padded["cache_position"].tolist(), [0, 1, 2, 3])
        self.assertEqual(literal["attention_mask"].tolist(), [[1, 1, 1]])
        self.assertEqual(literal["position_ids"].tolist(), [[0, 1, 2]])
        self.assertEqual(ordinary["attention_mask"].tolist(), [[1, 1, 1]])

    def test_prompt_and_incremental_history_inputs_remain_visible(self):
        prompt = core.prepare_case_inputs("prompt_at_once_vs_token_at_a_time", [1, 101, 103, 107])
        for end in range(1, 5):
            incremental = core.prepare_case_inputs("prompt_incremental", [1, 101, 103, 107][:end])
            self.assertTrue((incremental["attention_mask"] == 1).all())
            self.assertEqual(incremental["input_ids"].tolist()[0], [1, 101, 103, 107][:end])
        self.assertTrue((prompt["attention_mask"] == 1).all())

    def test_analytical_ngram_report_exposes_rounding_and_ignored_tokens(self):
        base = np.array([[[1.0], [2.0]]], dtype=np.float32)
        raw = [np.array([[[0.1], [0.25]]], dtype=np.float32) for _ in range(12)]
        ignored = np.array([False, True])
        analytical_expected = (1.0 + 1.2) / 13.0
        official = np.array([[[np.float32(0.169921875)], [5.0]]], dtype=np.float32)
        contributions, reconstructed, error, report = core.analytical_ngram_decomposition(
            base, raw, ignored, official)
        self.assertAlmostEqual(float(reconstructed[0, 0, 0]), analytical_expected, places=6)
        self.assertAlmostEqual(float(reconstructed[0, 1, 0]), 5.0, places=6)
        self.assertAlmostEqual(float(contributions[0][0, 1, 0]), 0.25, places=6)
        self.assertGreater(report["max_absolute_error"], 0)
        self.assertFalse(report["is_official_captured_intermediate"])
        self.assertAlmostEqual(float(error[0, 1, 0]), 0.0, places=6)

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


class NativeWindowsFlashAttentionTests(unittest.TestCase):
    class Distribution:
        version = "2.9.2"
        metadata = {"Name": "flash-attn"}
        def __init__(self, direct_url=None):
            self.direct_url = direct_url
        def read_text(self, name):
            if name == "WHEEL":
                return "Wheel-Version: 1.0\nTag: cp312-cp312-win_amd64\n"
            if name == "direct_url.json" and self.direct_url is not None:
                return json.dumps(self.direct_url)
            return None
        def locate_file(self, name):
            return Path(r"C:\fixture\site-packages")

    @staticmethod
    def observed_runtime():
        return {"torch_version": "2.13.0+cu132", "torch_cuda_build": "13.2",
                "gpu_compute_capability": [12, 0], "torch_cxx11_abi": True}

    def test_pep610_origin_recovers_complete_native_wheel_identity(self):
        filename = ("flash_attn-2.9.2+cu132torch2.13.0cxx11abiTRUE.blackwell-"
                    "cp312-cp312-win_amd64.whl")
        direct = {"url": "file:///C:/Downloads/" + filename.replace("+", "%2B"),
                  "archive_info": {"hash": "sha256=abc123"}}
        from packaging.tags import Tag
        with mock.patch("packaging.tags.sys_tags",
                        return_value=iter([Tag("cp312", "cp312", "win_amd64")])):
            report = core.windows_flash_distribution_report(
                self.Distribution(direct), self.observed_runtime())
        self.assertTrue(report["ok"])
        self.assertEqual(report["distribution_version"], "2.9.2")
        self.assertEqual(report["original_wheel_filename"], filename)
        self.assertEqual(report["archive_hash"], "sha256=abc123")
        self.assertEqual(report["wheel_cuda"], "13.2")
        self.assertEqual(report["wheel_torch"], "2.13.0")
        self.assertTrue(report["wheel_cxx11_abi"])
        self.assertTrue(report["checks"]["blackwell_kernel_build"])
        self.assertEqual(report["wheel_tags"], ["cp312-cp312-win_amd64"])

    def test_mismatched_direct_wheel_filename_is_rejected(self):
        filename = ("flash_attn-2.9.2+cu128torch2.13.0cxx11abiTRUE.legacy-"
                    "cp312-cp312-win_amd64.whl")
        from packaging.tags import Tag
        with mock.patch("packaging.tags.sys_tags",
                        return_value=iter([Tag("cp312", "cp312", "win_amd64")])):
            report = core.windows_flash_distribution_report(
                self.Distribution({"url": "file:///C:/" + filename}),
                self.observed_runtime())
        self.assertFalse(report["ok"])
        self.assertFalse(report["checks"]["cuda_build_matches"])
        self.assertFalse(report["checks"]["blackwell_kernel_build"])

    def test_missing_wheel_origin_has_clear_error_and_explicit_fallback(self):
        distribution = self.Distribution()
        with self.assertRaisesRegex(core.CoreFixtureError, "wheel origin identity unavailable"):
            core.wheel_filename_from_origin(distribution)
        origin = core.wheel_filename_from_origin(
            distribution, r"C:\Wheels\flash_attn-2.9.2-py3-none-any.whl")
        self.assertEqual(origin["identity_source"], "explicit-wheel-path")
        self.assertEqual(origin["original_wheel_filename"],
                         "flash_attn-2.9.2-py3-none-any.whl")

    def test_known_community_direct_origin_is_recorded(self):
        origin = core.wheel_filename_from_origin(self.Distribution({
            "url": "https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/flash_attn-2.9.2-py3-none-any.whl"}))
        self.assertTrue(origin["known_community_windows_source"])

    def test_custom_classes_import_only_after_flash_smoke(self):
        expected = core.EXPECTED_DEPENDENCIES
        core.EXPECTED_DEPENDENCIES = {"available": ("available-dist", None)}
        events = []
        def flash_probe(*args):
            events.append("flash-smoke")
            return {"ok": True}
        def custom_probe(*args):
            events.append("custom-classes")
            return {"ok": True}
        try:
            with mock.patch.object(core.importlib.metadata, "version", return_value="1"), \
                 mock.patch.object(core.importlib, "import_module", return_value=object()), \
                 mock.patch.object(core, "runtime_probe", return_value={"ok": True}), \
                 mock.patch.object(core, "flash_attention_probe", side_effect=flash_probe), \
                 mock.patch.object(core, "import_local_custom_classes", side_effect=custom_probe):
                report = core.dependency_preflight(
                    "blackwell-compatible", "cuda", model_dir="fixture")
        finally:
            core.EXPECTED_DEPENDENCIES = expected
        self.assertTrue(report["ok"])
        self.assertEqual(events, ["flash-smoke", "custom-classes"])

    def test_custom_classes_are_skipped_when_flash_fails(self):
        expected = core.EXPECTED_DEPENDENCIES
        core.EXPECTED_DEPENDENCIES = {"available": ("available-dist", None)}
        try:
            with mock.patch.object(core.importlib.metadata, "version", return_value="1"), \
                 mock.patch.object(core.importlib, "import_module", return_value=object()), \
                 mock.patch.object(core, "runtime_probe", return_value={"ok": True}), \
                 mock.patch.object(core, "flash_attention_probe",
                                   return_value={"ok": False, "error": "ABI failed"}), \
                 mock.patch.object(core, "import_local_custom_classes") as loader:
                report = core.dependency_preflight(
                    "blackwell-compatible", "cuda", model_dir="fixture")
        finally:
            core.EXPECTED_DEPENDENCIES = expected
        loader.assert_not_called()
        self.assertTrue(report["custom_code"]["skipped"])
        self.assertIn("FlashAttention", report["custom_code"]["reason"])

    def test_einops_is_unpinned_and_reports_installed_version(self):
        self.assertEqual(core.EXPECTED_DEPENDENCIES["einops"], ("einops", None))
        expected = core.EXPECTED_DEPENDENCIES
        core.EXPECTED_DEPENDENCIES = {"einops": ("einops", None)}
        try:
            with mock.patch.object(core.importlib.metadata, "version", return_value="0.8.1"), \
                 mock.patch.object(core.importlib, "import_module", return_value=object()), \
                 mock.patch.object(core, "runtime_probe", return_value={"ok": True}), \
                 mock.patch.object(core, "flash_attention_probe", return_value={"ok": True}):
                report = core.dependency_preflight("blackwell-compatible", "cpu")
        finally:
            core.EXPECTED_DEPENDENCIES = expected
        self.assertTrue(report["ok"])
        self.assertEqual(report["packages"]["einops"]["installed_version"], "0.8.1")
        self.assertIsNone(report["packages"]["einops"]["official_pinned_version"])

    def test_missing_einops_is_reported_before_custom_code(self):
        expected = core.EXPECTED_DEPENDENCIES
        core.EXPECTED_DEPENDENCIES = {"einops": ("einops", None)}
        try:
            with mock.patch.object(core.importlib.metadata, "version",
                                   side_effect=importlib.metadata.PackageNotFoundError("einops")), \
                 mock.patch.object(core.importlib, "import_module",
                                   side_effect=ModuleNotFoundError("No module named 'einops'")), \
                 mock.patch.object(core, "runtime_probe", return_value={"ok": True}), \
                 mock.patch.object(core, "import_local_custom_classes") as loader:
                report = core.dependency_preflight(
                    "blackwell-compatible", "cuda", model_dir="fixture")
        finally:
            core.EXPECTED_DEPENDENCIES = expected
        self.assertFalse(report["ok"])
        self.assertIn("einops", report["packages"])
        self.assertIn("import failed", report["packages"]["einops"]["error"])
        loader.assert_not_called()

    def test_official_profile_rejects_newer_flash_attention(self):
        torch = types.SimpleNamespace()
        flash = types.SimpleNamespace(__file__=r"C:\fixture\flash_attn\__init__.py")
        distribution = types.SimpleNamespace(version="2.9.2")
        with mock.patch.object(core.importlib, "import_module",
                               side_effect=lambda name: torch if name == "torch" else flash), \
             mock.patch.object(core.importlib.metadata, "distribution", return_value=distribution):
            report = core.flash_attention_probe({}, "cuda", "official-pinned")
        self.assertFalse(report["ok"])
        self.assertIn("requires flash-attn 2.7.4.post1", report["error"])

    def test_native_windows_newer_build_requires_and_passes_smoke(self):
        torch = types.SimpleNamespace()
        flash = types.SimpleNamespace(__file__=r"C:\fixture\flash_attn\__init__.py")
        distribution = types.SimpleNamespace(version="2.9.2+cu132torch2.13.0.blackwell")
        abi = {"ok": True, "community_unofficial_windows_build": True,
               "community_source": "fixture"}
        smoke = {"operation": "passed", "output_shape": [1, 4, 2, 16],
                 "finite_values": True, "cuda_synchronize": "passed"}
        with mock.patch("platform.system", return_value="Windows"), \
             mock.patch.object(core.importlib, "import_module",
                               side_effect=lambda name: torch if name == "torch" else flash), \
             mock.patch.object(core.importlib.metadata, "distribution", return_value=distribution), \
             mock.patch.object(core, "windows_flash_distribution_report", return_value=abi), \
             mock.patch.object(core, "perform_flash_attention_smoke", return_value=smoke) as probe:
            report = core.flash_attention_probe({}, "cuda", "blackwell-compatible")
        self.assertTrue(report["ok"])
        self.assertEqual(report["official_pinned_version"], "2.7.4.post1")
        self.assertEqual(report["installed_distribution_version"], distribution.version)
        self.assertTrue(report["version_departure_from_official"])
        self.assertEqual(report["provenance"], "community/unofficial native Windows build")
        self.assertFalse(report["wsl_required"])
        probe.assert_called_once_with(torch, flash)

    def test_native_windows_abi_rejects_python_tag_mismatch(self):
        report = core.windows_flash_abi(
            "2.9.2+cu132torch2.13.0cxx11abiTRUE.blackwell",
            ["cp313-cp313-win_amd64"], ["cp312-cp312-win_amd64"],
            "2.13.0+cu132", "13.2", (12, 0), True)
        self.assertFalse(report["ok"])
        self.assertFalse(report["checks"]["python_abi_platform_tag_matches"])
        self.assertEqual(report["executing_compatible_tags"], [])

    def test_native_windows_abi_accepts_matching_blackwell_wheel(self):
        tag = "cp313-cp313-win_amd64"
        report = core.windows_flash_abi(
            "2.9.2+cu132torch2.13.0cxx11abiTRUE.blackwell", [tag], [tag],
            "2.13.0+cu132", "13.2", [12, 0], True)
        self.assertTrue(report["ok"])
        self.assertEqual(report["wheel_cuda"], "13.2")
        self.assertEqual(report["wheel_torch"], "2.13.0")


if __name__ == "__main__":
    unittest.main()
