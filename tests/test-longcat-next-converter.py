#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "longcat_next_inventory", ROOT / "conversion/longcat_next_inventory.py")
inventory = importlib.util.module_from_spec(SPEC)
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


if __name__ == "__main__":
    unittest.main()
