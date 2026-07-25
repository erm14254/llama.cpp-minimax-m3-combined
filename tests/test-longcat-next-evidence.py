#!/usr/bin/env python3
"""Tests for the LongCat-Next read-only evidence harness."""

import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

inventory = load("longcat_next_inventory", "scripts/longcat-next/inventory.py")
fixtures = load("longcat_next_fixtures", "scripts/longcat-next/make-reference-fixtures.py")

SUB = inventory.EXPECTED_SUBFAMILIES

def names(prefix, count):
    return [f"{prefix}tensor_{i}" for i in range(count)]

class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        modal = []
        for prefix, count in SUB.items():
            modal.extend(names(prefix, count))
        text = names("model.layers.synthetic.", 11143)
        self.next_names = text + modal
        self.lite_names = text + names("model.mtp.synthetic.", 17)
        self.next_path = self.write("next.json", {
            "metadata": {"total_size": inventory.EXPECTED_PAYLOAD},
            "weight_map": {name: "shard" for name in self.next_names},
        })
        self.lite_path = self.write("lite.json", {
            "metadata": {"total_size": 1},
            "weight_map": {name: "shard" for name in self.lite_names},
        })
        self.config_path = self.write("config.json", dict(inventory.EXPECTED_VOCAB))

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def mutate_next(self, callback):
        data = json.loads(self.next_path.read_text())
        callback(data)
        self.next_path.write_text(json.dumps(data), encoding="utf-8")

    def test_valid_inventory(self):
        report = inventory.validate(self.next_path, self.lite_path, self.config_path)
        self.assertEqual(report["main_tensor_count"], 13450)
        self.assertEqual(report["text_names_in_lite"], 11143)

    def assert_inventory_error(self, substring):
        with self.assertRaisesRegex(inventory.InventoryError, substring):
            inventory.validate(self.next_path, self.lite_path, self.config_path)

    def test_wrong_tensor_count(self):
        self.mutate_next(lambda d: d["weight_map"].pop(next(iter(d["weight_map"]))))
        self.assert_inventory_error("tensor count")

    def test_unexpected_mtp_tensor(self):
        def mutate(data):
            old = next(iter(data["weight_map"]))
            value = data["weight_map"].pop(old)
            data["weight_map"]["model.mtp.unexpected.weight"] = value
        self.mutate_next(mutate)
        self.assert_inventory_error("MTP tensor count")

    def test_wrong_vocabulary_extent(self):
        config = json.loads(self.config_path.read_text())
        config["text_vocab_size"] = 131071
        self.config_path.write_text(json.dumps(config))
        self.assert_inventory_error("text_vocab_size")

    def test_unclassified_modality_family(self):
        def mutate(data):
            old = next(k for k in data["weight_map"] if k.startswith("model.visual_tokenizer.visual_model."))
            value = data["weight_map"].pop(old)
            data["weight_map"]["model.visual_tokenizer.unknown.tensor"] = value
        self.mutate_next(mutate)
        self.assert_inventory_error("unclassified or ambiguous")

    def test_wrong_payload_size(self):
        self.mutate_next(lambda d: d["metadata"].update(total_size=7))
        self.assert_inventory_error("payload bytes")

    def test_missing_required_field(self):
        self.mutate_next(lambda d: d.pop("metadata"))
        self.assert_inventory_error("missing required field 'metadata'")

    def hift_metadata(self, wrong_total=False):
        tensors = {
            "conv.weight_g": {"dtype": "F32", "shape": [1]},
            "conv.weight_v": {"dtype": "F32", "shape": [1]},
        }
        for index in range(325):
            tensors[f"tensor_{index}"] = {"dtype": "F32", "shape": [1]}
        remainder = inventory.EXPECTED_HIFT_PARAMETERS - 327
        tensors["tensor_final"] = {"dtype": "F32", "shape": [remainder + int(wrong_total)]}
        return self.write("hift.json", {"tensors": tensors})

    def test_hift_exact_parameter_and_payload_totals(self):
        report = inventory.validate_hift(self.hift_metadata())
        self.assertEqual(report["parameters"], inventory.EXPECTED_HIFT_PARAMETERS)
        self.assertEqual(report["tensor_payload_bytes"], inventory.EXPECTED_HIFT_PAYLOAD_BYTES)

    def test_hift_wrong_parameter_total(self):
        with self.assertRaisesRegex(inventory.InventoryError, "parameter count"):
            inventory.validate_hift(self.hift_metadata(wrong_total=True))

class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / ".longcat-next-revision").write_text(fixtures.HF_REVISION, encoding="ascii")
        source_bytes = b"# synthetic pinned source for harness tests\n"
        (self.source / "modeling_longcat_ngram.py").write_bytes(source_bytes)
        self.original_source_hash = fixtures.NGRAM_SOURCE_SHA256
        self.original_official_runner = fixtures.official_hash_batches
        fixtures.NGRAM_SOURCE_SHA256 = hashlib.sha256(source_bytes).hexdigest()
        fixtures.official_hash_batches = lambda source, sequences: [fixtures.hashes(ids) for ids in sequences]
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"text_vocab_size": 131072, "eos_token_id": 2,
            "bos_token_id": 1, "ngram_vocab_size_ratio": 78,
            "emb_neighbor_num": 4, "emb_split_num": 4}), encoding="utf-8")
        self.tokenizer = self.root / "tokenizer_config.json"
        self.tokenizer.write_text("{}", encoding="utf-8")
        self.original_config_hash = fixtures.CONFIG_SHA256
        self.original_tokenizer_hash = fixtures.TOKENIZER_CONFIG_SHA256
        fixtures.CONFIG_SHA256 = hashlib.sha256(self.config.read_bytes()).hexdigest()
        fixtures.TOKENIZER_CONFIG_SHA256 = hashlib.sha256(self.tokenizer.read_bytes()).hexdigest()

    def tearDown(self):
        fixtures.NGRAM_SOURCE_SHA256 = self.original_source_hash
        fixtures.official_hash_batches = self.original_official_runner
        fixtures.CONFIG_SHA256 = self.original_config_hash
        fixtures.TOKENIZER_CONFIG_SHA256 = self.original_tokenizer_hash
        self.tmp.cleanup()

    def args(self, **overrides):
        values = dict(official_source=self.source, source_revision=fixtures.HF_REVISION,
            inference_revision=fixtures.INFERENCE_REVISION, model_revision=fixtures.HF_REVISION,
            config=self.config, tokenizer_config=self.tokenizer, output_dir=self.root / "out",
            mode="ngram", model_dir=None, max_output_bytes=fixtures.DEFAULT_MAX_BYTES)
        values.update(overrides)
        return type("Args", (), values)()

    def test_weight_free_generation_covers_contract(self):
        path, size = fixtures.run(self.args())
        data = json.loads(path.read_text())
        names = {case["name"] for case in data["cases"]}
        self.assertIn("all_ignored_ids", names)
        self.assertIn("two_independent_histories", names)
        ignored = next(case for case in data["cases"] if case["name"] == "all_ignored_ids")
        self.assertEqual(ignored["input_ids"], list(range(131072, 131125)))
        self.assertEqual({(h["order"], h["split"]) for h in ignored["official_hashes"]},
                         {(order, split) for order in (2, 3, 4) for split in range(4)})
        prompt = next(case for case in data["cases"] if case["name"].startswith("prompt_at_once"))
        self.assertTrue(prompt["equal"])
        self.assertLess(size, fixtures.DEFAULT_MAX_BYTES)

    def test_generation_fails_on_official_mismatch(self):
        def mismatch(source, sequences):
            result = [fixtures.hashes(ids) for ids in sequences]
            result[0][0]["ids"][0] += 1
            return result
        fixtures.official_hash_batches = mismatch
        with self.assertRaisesRegex(fixtures.FixtureError, "differs from official"):
            fixtures.run(self.args())

    def test_mutable_unpinned_revision(self):
        with self.assertRaisesRegex(fixtures.FixtureError, "immutable 40-character"):
            fixtures.run(self.args(model_revision="main"))

    def test_unverifiable_source_revision(self):
        (self.source / ".longcat-next-revision").unlink()
        with self.assertRaisesRegex(fixtures.FixtureError, "revision"):
            fixtures.run(self.args())

    def test_oversized_fixture_output(self):
        with self.assertRaisesRegex(fixtures.FixtureError, "above limit"):
            fixtures.run(self.args(max_output_bytes=100))
        self.assertFalse((self.root / "out" / "ngram-cases.json").exists())

if __name__ == "__main__":
    unittest.main()
