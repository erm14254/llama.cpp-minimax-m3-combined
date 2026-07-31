#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path
from typing import Any

import torch

from conversion.base import gguf
from conversion.longcat_next import round_router_correction_bias_for_outtype

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "longcat_next_inventory", ROOT / "conversion/longcat_next_inventory.py")
assert SPEC is not None and SPEC.loader is not None
inventory: Any = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class LongCatNextInventoryTests(unittest.TestCase):
    def test_three_vocabulary_extents_and_ignored_interval(self):
        self.assertEqual(inventory.HASH_VOCAB_SIZE, 131072)
        self.assertEqual(inventory.CORE_VOCAB_SIZE, 131125)
        self.assertEqual(inventory.SOURCE_VOCAB_SIZE, 282624)
        self.assertEqual((inventory.IGNORED_START, inventory.IGNORED_COUNT), (131072, 53))

    def names(self):
        names = {f"model.core.synthetic_{i}" for i in range(11143)}
        for prefix, count in inventory.MODAL_PREFIX_COUNTS.items():
            names.update(f"{prefix}synthetic_{i}" for i in range(count))
        return names

    def test_exact_core_and_deferred_inventory(self):
        core, deferred = inventory.classify_longcat_next_names(self.names())
        self.assertEqual(len(core), 11143)
        self.assertEqual(len(deferred), 2307)
        self.assertEqual(len(core | deferred), 13450)

    def test_mtp_is_rejected(self):
        names = self.names()
        names.remove("model.core.synthetic_0")
        names.add("model.mtp.layers.0.synthetic")
        with self.assertRaisesRegex(ValueError, "MTP"):
            inventory.classify_longcat_next_names(names)

    def test_unclassified_or_wrong_modal_family_is_rejected(self):
        names = self.names()
        names.remove("visual_head.synthetic_0")
        names.add("unknown.modal.synthetic")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            inventory.classify_longcat_next_names(names)


class LongCatNextPrecisionTests(unittest.TestCase):
    def test_router_correction_bias_rounding_matches_export_outtype(self):
        source = torch.tensor(
            [0.0036110980436205864, -0.008998246863484383, 0.0],
            dtype=torch.float32,
        )

        bf16 = round_router_correction_bias_for_outtype(
            source, gguf.LlamaFileType.MOSTLY_BF16)
        f16 = round_router_correction_bias_for_outtype(
            source, gguf.LlamaFileType.MOSTLY_F16)
        f32 = round_router_correction_bias_for_outtype(
            source, gguf.LlamaFileType.ALL_F32)

        self.assertEqual(bf16.dtype, torch.float32)
        self.assertEqual(f16.dtype, torch.float32)
        self.assertTrue(torch.equal(bf16, source.bfloat16().float()))
        self.assertTrue(torch.equal(f16, source.half().float()))
        self.assertIs(f32, source)
        self.assertFalse(torch.equal(bf16, source))
        self.assertFalse(torch.equal(f16, source))


if __name__ == "__main__":
    unittest.main()
