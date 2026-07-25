#!/usr/bin/env python3
"""Execute pinned official LongCat-Next weight-free n-gram methods in isolation."""

import argparse
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch

METHODS = {"_shift_right_ignore_eos", "_precompute_vocab_mods", "_get_ngram_ids"}


def load_official_class(path):
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    target = next((node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "NgramEmbedding"), None)
    if target is None:
        raise RuntimeError("official source has no NgramEmbedding class")
    methods = [node for node in target.body
               if isinstance(node, ast.FunctionDef) and node.name in METHODS]
    found = {node.name for node in methods}
    if found != METHODS:
        raise RuntimeError(f"official source methods mismatch: expected {METHODS}, got {found}")
    isolated = ast.Module(body=[ast.ClassDef(name="OfficialNgramEmbedding", bases=[], keywords=[],
                                              body=methods, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace = {"torch": torch, "Dict": Dict, "List": List, "Tuple": Tuple}
    exec(compile(isolated, str(path), "exec"), namespace)
    return namespace["OfficialNgramEmbedding"]


def official_hashes(instance, tokens):
    # This is the exact preprocessing performed by NgramEmbedding.forward before
    # calling the three extracted official methods.
    tokens = [0 if 131072 <= token < 131125 else token for token in tokens]
    tensor = torch.tensor([tokens], dtype=torch.long)
    shifted = {order: instance._shift_right_ignore_eos(tensor, order - 1, eos_token_id=2)
               for order in (2, 3, 4)}
    mods = instance._precompute_vocab_mods()
    result = []
    for order in (2, 3, 4):
        for split in range(4):
            index = (order - 2) * 4 + split
            modulus = instance.m + 2 * index + 1
            ids = instance._get_ngram_ids(tensor, shifted, mods[(order, split)], order) % modulus
            values = ids[0].tolist()
            result.append({"order": order, "split": split, "table_index": index,
                           "modulus": modulus, "power_mods": mods[(order, split)],
                           "ids": values, "lookup_mask": [value > 0 for value in values]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    request = json.load(sys.stdin)
    sequences = request.get("sequences")
    if not isinstance(sequences, list) or not all(isinstance(x, list) for x in sequences):
        raise SystemExit("input must contain a 'sequences' array of integer arrays")
    cls = load_official_class(args.source)
    instance = cls.__new__(cls)
    instance.config = SimpleNamespace(text_vocab_size=131072)
    instance.m = 78 * 131072
    instance.k = 4
    instance.n = 4
    instance._vocab_mods_cache = None
    json.dump({"hashes": [official_hashes(instance, sequence) for sequence in sequences]}, sys.stdout,
              sort_keys=True)
    sys.stdout.write("\n")

if __name__ == "__main__":
    main()
