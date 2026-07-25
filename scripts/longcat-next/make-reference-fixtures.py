#!/usr/bin/env python3
"""Create small, deterministic LongCat-Next reference fixtures without model weights."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import sys
from pathlib import Path

HF_REVISION = "0cf0631862402ff36366e513e4023d22e7e5c84c"
SOURCE_REVISION = "49dc718151f9943a9dca2c1169541934bb85d83e"
INFERENCE_REVISION = "70ab100beecaaa77c1f45c0cf9ec89a4faf20fd8"
SCHEMA_VERSION = 1
DEFAULT_MAX_BYTES = 1024 * 1024
NGRAM_SOURCE_SHA256 = "f7c6fb4de561e3311a67adea22b8a9467044d3c503d59afe0dde542e55b17e09"
CONFIG_SHA256 = "9115e9785603b04382a45ebece9092235281f309f56f35eb4e43bcf53150b2a2"
TOKENIZER_CONFIG_SHA256 = "22dddd0eb59965adf6e4861a7c8a9ed803595cd16bc86ed6e2d4ed915b9718d4"
SEEDS = {"python": 20260725, "torch": 20260725, "numpy": 20260725}

class FixtureError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise FixtureError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned_revision(value, expected, label):
    require(value is not None, f"{label} must be recorded")
    require(re.fullmatch(r"[0-9a-f]{40}", value) is not None,
            f"{label} must be an immutable 40-character lowercase commit, got {value!r}")
    require(value == expected, f"{label}: expected pinned revision {expected}, got {value}")


def verify_source(path, recorded_revision):
    path = Path(path)
    require(path.exists(), f"official source path does not exist: {path}")
    pinned_revision(recorded_revision, HF_REVISION, "official source revision")
    if (path / ".git").exists():
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        require(head == recorded_revision,
                f"official source checkout HEAD {head} does not match recorded {recorded_revision}")
        dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)
        require(not dirty, "official source checkout is mutable: working tree is dirty")
    else:
        marker = path / ".longcat-next-revision"
        require(marker.is_file(),
                "non-git official source must contain .longcat-next-revision")
        require(marker.read_text(encoding="ascii").strip() == recorded_revision,
                "official source revision marker does not match the recorded revision")
    ngram_source = path / "modeling_longcat_ngram.py"
    require(ngram_source.is_file(),
            "official source is missing modeling_longcat_ngram.py")
    require(sha256(ngram_source) == NGRAM_SOURCE_SHA256,
            "modeling_longcat_ngram.py does not match the pinned official implementation")


def load_json(path, label):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureError(f"cannot read {label} {path}: {exc}") from exc
    require(isinstance(data, dict), f"{label} must contain a JSON object")
    return data


def shifted(tokens, distance, eos=2):
    result = []
    for pos in range(len(tokens)):
        source = pos - distance
        if source < 0 or tokens[pos] == 0 or any(tokens[i] in (0, eos) for i in range(source, pos)):
            result.append(0)
        else:
            result.append(tokens[source])
    return result


def hashes(tokens, text_vocab=131072, ratio=78, ignored_start=131072, ignored_end=131125):
    context = [0 if ignored_start <= token < ignored_end else token for token in tokens]
    out = []
    for order in (2, 3, 4):
        shifts = {distance: shifted(context, distance) for distance in range(1, order)}
        for split in range(4):
            index = (order - 2) * 4 + split
            modulus = ratio * text_vocab + 2 * index + 1
            mods, power = [], 1
            for _ in range(order - 1):
                power = (power * text_vocab) % modulus
                mods.append(power)
            ids = []
            for pos, token in enumerate(context):
                value = token
                for distance in range(1, order):
                    value += shifts[distance][pos] * mods[distance - 1]
                ids.append(value % modulus)
            out.append({"order": order, "split": split, "table_index": index,
                        "modulus": modulus, "power_mods": mods, "ids": ids,
                        "lookup_mask": [value > 0 for value in ids]})
    return out


def make_cases(config):
    for field, value in {"text_vocab_size": 131072, "eos_token_id": 2,
                         "ngram_vocab_size_ratio": 78, "emb_neighbor_num": 4,
                         "emb_split_num": 4}.items():
        require(field in config, f"config missing required field {field!r}")
        require(config[field] == value, f"config {field}: expected {value}, got {config[field]!r}")
    bos = config.get("bos_token_id", 1)
    cases = [
        ("bos_left_zero", [0, 0, bos, 17, 23]),
        ("literal_zero", [19, 0, 29, 31, 37]),
        ("maximum_text_token", [7, 11, 13, 131071]),
    ]
    for offset in range(4):
        sequence = [41, 43, 47, 53]
        sequence[offset] = 2
        cases.append((f"eos_window_position_{offset}", sequence + [59]))
    cases.append(("all_ignored_ids", list(range(131072, 131125))))
    rendered = [{"name": name, "input_ids": ids, "hashes": hashes(ids)} for name, ids in cases]

    prompt = [bos, 101, 103, 107, 109, 113]
    prompt_hashes = hashes(prompt)
    streamed = []
    history = []
    for token in prompt:
        combined = history[-3:] + [token]
        current = hashes(combined)
        streamed.append([entry["ids"][-1] for entry in current])
        history.append(0 if 131072 <= token < 131125 else token)
    rendered.append({"name": "prompt_at_once_vs_token_at_a_time", "input_ids": prompt,
                     "prompt_hash_ids": [[entry["ids"][i] for entry in prompt_hashes] for i in range(len(prompt))],
                     "stream_hash_ids": streamed,
                     "equal": all([entry["ids"][i] for entry in prompt_hashes] == streamed[i]
                                  for i in range(len(prompt)))})

    sequences = {"sequence_0": [bos, 211, 223, 227], "sequence_1": [bos, 307, 2, 311, 313]}
    rendered.append({"name": "two_independent_histories", "sequences": {
        name: {"input_ids": ids, "hashes": hashes(ids)} for name, ids in sequences.items()
    }})
    return rendered


def versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "transformers", "safetensors", "numpy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def write_limited(path, data, limit):
    payload = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("ascii")
    require(len(payload) <= limit,
            f"fixture output would be {len(payload)} bytes, above limit {limit}; refusing to write")
    Path(path).write_bytes(payload)
    return len(payload)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--inference-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tokenizer-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("ngram", "core"), default="ngram")
    parser.add_argument("--model-dir", type=Path,
                        help="local full checkpoint for future core fixtures; never downloaded")
    parser.add_argument("--max-output-bytes", type=int, default=DEFAULT_MAX_BYTES)
    return parser.parse_args(argv)


def run(args):
    pinned_revision(args.model_revision, HF_REVISION, "model revision")
    pinned_revision(args.inference_revision, INFERENCE_REVISION, "inference revision")
    verify_source(args.official_source, args.source_revision)
    require(0 < args.max_output_bytes <= DEFAULT_MAX_BYTES,
            f"max output bytes must be in [1,{DEFAULT_MAX_BYTES}], got {args.max_output_bytes}")
    config = load_json(args.config, "config")
    load_json(args.tokenizer_config, "tokenizer config")
    require(sha256(args.config) == CONFIG_SHA256,
            "config does not match the pinned official identity")
    require(sha256(args.tokenizer_config) == TOKENIZER_CONFIG_SHA256,
            "tokenizer config does not match the pinned official identity")
    random.seed(SEEDS["python"])
    if args.mode == "core":
        require(args.model_dir is not None and args.model_dir.is_dir(),
                "core fixtures require --model-dir pointing to a local official checkpoint")
        raise FixtureError("core fixture interface is reserved until numerical tolerances are frozen; no output written")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    require(not any(p.name.endswith((".safetensors", ".bin", ".pt")) for p in args.output_dir.iterdir()),
            "output directory contains a possible model-weight file")
    fixture_path = args.output_dir / "ngram-cases.json"
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": "longcat-next-weight-free-ngram",
        "source_revisions": {"longcat_next": args.source_revision,
                             "longcat_next_repository": SOURCE_REVISION,
                             "longcat_next_inference": args.inference_revision,
                             "hugging_face_model": args.model_revision},
        "software_versions": versions(), "deterministic_seeds": SEEDS,
        "identities": {"config_sha256": sha256(args.config),
                       "tokenizer_config_sha256": sha256(args.tokenizer_config)},
        "cases": make_cases(config),
    }
    size = write_limited(fixture_path, document, args.max_output_bytes)
    return fixture_path, size


def main(argv=None):
    args = parse_args(argv)
    try:
        path, size = run(args)
    except (FixtureError, subprocess.CalledProcessError) as exc:
        print(f"fixture error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"fixture": str(path), "bytes": size, "sha256": sha256(path)}, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
