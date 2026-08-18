#!/usr/bin/env python
"""Offline determinism comparator for the 2050 Type-S/Type-P protocol.

Implements the PRE-REGISTERED verdict semantics (bootstrap handoff
WIN11_HANDOFF_2026-08-18_DELTA_LSA_MEASUREMENT_APPARATUS.md) verbatim -- it
never re-decides them. NO HF comparison anywhere; final logits are compared
only across P1/P2/P3.

Verdict stack (per owner block, per row, across repeats):
  1. Structural top-K validity FIRST (exactly 2048 recovered indices,
     integral under exact lrint recovery, in [0, n_kv_lid), unique). A
     validity failure is a round-stopping anomaly, NOT a determinism verdict.
  2. V-input  -- 8-surface S1/S2/S3 byte-stability table (mandatory
     observability, reported FIRST): variation => STOP FOR REVIEW
     (input_variability), classified pre-selection/input variability, never
     attributed to CUB top-K; anchor variation is additionally an upstream
     reproducibility anomaly.
  3. V-ord      -- raw index ordering: characterization only (this build's
     ggml_top_k is CUB DeviceTopK::MaxPairs, CCCL 3.2.0,
     determinism::not_guaranteed + output_ordering::unsorted).
  4. V-mem-raw  -- complete raw selected set: characterization; reported
     split rows p <= 2047 vs p in {2048, 2049}; variation confined to
     causally invisible -inf fillers is NONBLOCKING.
  5. V-mem-effective -- selected AND causally visible: BLOCKING.
  6. V-mask     -- reconstructed effective attention mask: BLOCKING.
  7. V-logit    -- P1/P2/P3 final logits byte/SHA + top-1: BLOCKING.

Positive expectations (per run): rows p <= 2047 select their full causal set
exactly; rows 2048/2049 (the only truly sparse-selective rows, reported
separately) have raw == effective, |effective| = 2048, and contain all 1,040
forced positions.

Exit codes:
  0 = protocol valid, no blocking variation, inputs stable
  2 = protocol valid, STOP FOR REVIEW (reasons: blocking_variation |
      positive_expectation_failure | input_variability)
  3 = protocol/validity anomaly (no determinism verdict exists)
  4 = refused to start (result dir exists / usage error); nothing computed
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

N_TOKENS = 2050
K = 2048
N_INIT = 16
N_LOCAL = 1024
EXPECT_N_KV_LID = 2304
VOCAB = 131072
EXPECT_TOKENS_SHA = "eb04e101e452e3bb60911f02b5b5ac538e8a183efcb564d7c9859d7e03266bed"
EXPECT_TOKENS_BYTES = N_TOKENS * 4
EXPECT_LOGITS_BYTES = VOCAB * 4
MASK_LITERAL = ("LONGCAT_LSA_AUDIT mask seq=0 query_pos=2049 visible=2050 "
                "forced=1040 init_pos=[0,15] local_pos=[1026,2049]")
AUDIT_RE = re.compile(
    r"LONGCAT_LSA_AUDIT (owner|reuse) block=(\d+)"
    r"(?: owner_block=(\d+))? n_kv=(\d+) top_k=(\d+) tensor=(\S+)")
PLACEMENT_IDENTITY_FIELDS = (
    "git_head", "placement_line", "offloaded_line", "id_dense_start",
    "n_kv_lid_real")

# The 8 non-top-K Type-S surfaces (V-input set). source_type provenance:
# the five below-threshold surfaces are 512-proven (sidecars of
# cpp_lsa_dump_proof_lsaE_512/); the sparse-only trio (q_proj/q_2d/weights)
# is SOURCE-DERIVED (longcat-flash-ngram.cpp:885-941) and receives its first
# runtime proof at 2050 -- a mismatch there is a review-worthy anomaly, by
# design.
S_SURFACES = [
    # (bin_name, tensor_name, ne0, source_type, provenance)
    ("lsa_anchor_attn_norm0_full.bin", "attn_norm-0",           3072, "f32",  "512-proven"),
    ("lsa_anchor_q_a_norm0_full.bin",  "q_a_norm-0",            1536, "bf16", "512-proven"),
    ("lsa_indexer_k_proj_full.bin",    "lsa_indexer_k_proj-0",   128, "f32",  "512-proven"),
    ("lsa_indexer_k_norm_full.bin",    "lsa_indexer_k_norm-0",   128, "bf16", "512-proven"),
    ("lsa_indexer_k_full.bin",         "lsa_indexer_k_2d-0",     128, "bf16", "512-proven"),
    ("lsa_indexer_q_proj_full.bin",    "lsa_indexer_q_proj-0",  2048, "bf16", "source-derived"),
    ("lsa_indexer_q_full.bin",         "lsa_indexer_q_2d-0",    2048, "bf16", "source-derived"),
    ("lsa_indexer_weights_full.bin",   "lsa_indexer_weights-0",   16, "f32",  "source-derived"),
]
# Owner top-K artifacts: file lsa_top_k_owner<NN>_full.bin records owner
# block NN = il - 1; the tensor is name-keyed on the reuse marker (the owner
# cb name is renamed in place and never survives to eval time).
TOPK_FILES = [("lsa_top_k_owner%02d_full.bin" % (il - 1),
               "lsa_top_k_reuse-%d" % il, il - 1)
              for il in range(1, 28, 2)]
ALL_S_BINS = ([name for name, _, _, _, _ in S_SURFACES] +
              [name for name, _, _ in TOPK_FILES])


class ProtocolAnomaly(Exception):
    """Phase 0/1 failure: the protocol run set is invalid; no determinism
    verdict exists (exit 3)."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_provenance(run_dir):
    p = run_dir / "run_provenance.json"
    if not p.is_file():
        raise ProtocolAnomaly("missing run_provenance.json in %s" % run_dir)
    with open(p, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def verify_log_binding(tag, prov):
    """Hygiene 1: rehash + size-check both logs against provenance BEFORE
    any content is parsed. Log paths come EXCLUSIVELY from provenance."""
    bound = {}
    for stream in ("stdout", "stderr"):
        path_field = "%s_log" % stream
        for field in (path_field, path_field + "_sha256", path_field + "_bytes"):
            if field not in prov:
                raise ProtocolAnomaly("%s: provenance lacks %s" % (tag, field))
        log_path = Path(prov[path_field])
        if not log_path.is_file():
            raise ProtocolAnomaly("%s: bound log missing: %s" % (tag, log_path))
        data = log_path.read_bytes()
        if len(data) != int(prov[path_field + "_bytes"]):
            raise ProtocolAnomaly(
                "%s: %s log size %d != bound %s" %
                (tag, stream, len(data), prov[path_field + "_bytes"]))
        digest = sha256_bytes(data)
        if digest != str(prov[path_field + "_sha256"]).lower():
            raise ProtocolAnomaly(
                "%s: %s log sha %s != bound %s" %
                (tag, stream, digest, prov[path_field + "_sha256"]))
        bound[stream] = data.decode("utf-8", errors="replace")
    return bound["stdout"], bound["stderr"]


def verify_inventory_and_manifest(tag, run_dir, expected_names_and_sizes):
    """Hygiene/amendment 4: exact inventory (names AND sizes where pinned),
    manifest entry-set equality, and a full rehash of every manifest line."""
    actual = {f.name: f for f in run_dir.iterdir() if f.is_file()}
    expected_all = set(expected_names_and_sizes) | {
        "run_provenance.json", "SHA256SUMS.txt"}
    if set(actual) != expected_all:
        extra = sorted(set(actual) - expected_all)
        missing = sorted(expected_all - set(actual))
        raise ProtocolAnomaly(
            "%s: inventory mismatch (extra=%s missing=%s)" %
            (tag, extra, missing))
    for name, size in expected_names_and_sizes.items():
        if size is not None and actual[name].stat().st_size != size:
            raise ProtocolAnomaly(
                "%s: %s is %d B (expected %d)" %
                (tag, name, actual[name].stat().st_size, size))
    manifest_path = run_dir / "SHA256SUMS.txt"
    entries = {}
    for line in manifest_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([0-9a-fA-F]{64})\s+(\S.*)$", line)
        if not m:
            raise ProtocolAnomaly("%s: unparseable manifest line: %r" % (tag, line))
        entries[m.group(2)] = m.group(1).lower()
    expected_manifest = set(expected_names_and_sizes) | {"run_provenance.json"}
    if set(entries) != expected_manifest:
        raise ProtocolAnomaly(
            "%s: manifest entry set mismatch (got %d entries, expected %d: "
            "extra=%s missing=%s)" %
            (tag, len(entries), len(expected_manifest),
             sorted(set(entries) - expected_manifest),
             sorted(expected_manifest - set(entries))))
    for name, recorded in entries.items():
        actual_sha = sha256_file(run_dir / name)
        if actual_sha != recorded:
            raise ProtocolAnomaly(
                "%s: manifest rehash FAIL for %s (%s != %s)" %
                (tag, name, actual_sha, recorded))
    return entries


def verify_sidecars(tag, run_dir):
    """Amendment/hygiene 2: semantic validation of ALL 22 sidecars."""
    specs = []
    for name, tensor, ne0, source_type, _prov in S_SURFACES:
        specs.append((name, tensor, ne0, source_type))
    for name, tensor, _owner in TOPK_FILES:
        specs.append((name, tensor, K, "i32"))
    for bin_name, tensor, ne0, source_type in specs:
        sc_path = run_dir / (bin_name[:-4] + ".json")
        with open(sc_path, "r", encoding="utf-8-sig") as f:
            sc = json.load(f)
        expect = {
            "tensor": tensor,
            "shape": [N_TOKENS, ne0],
            "order": "token-major",
            "dtype": "float32-le",
            "bytes": ne0 * N_TOKENS * 4,
            "source_type": source_type,
            "source_contiguous": True,
            "source_ne": [ne0, N_TOKENS, 1, 1],
        }
        for field, want in expect.items():
            got = sc.get(field)
            if got != want:
                raise ProtocolAnomaly(
                    "%s: sidecar %s field %s = %r (expected %r)" %
                    (tag, sc_path.name, field, got, want))


def verify_structural_evidence(tag, stderr_text):
    """Real-sparse structural evidence, independently re-parsed from the
    cryptographically bound stderr. Reserve/fit n_kv=4608 lines are ignored
    and are never proof of sparse execution."""
    if not re.search(r"graphs reused =\s+0", stderr_text):
        raise ProtocolAnomaly("%s: graphs-reused != 0" % tag)
    for needle in ("llama_kv_cache_dsa: creating main KV cache, size = 4608 cells",
                   "creating indexer KV cache, size = 4608 cells"):
        if needle not in stderr_text:
            raise ProtocolAnomaly("%s: DSA cache line missing: %s" % (tag, needle))
    real = []
    reserve = 0
    for m in AUDIT_RE.finditer(stderr_text):
        n_kv = int(m.group(4))
        if n_kv == EXPECT_N_KV_LID:
            real.append(m)
        elif n_kv == 4608:
            reserve += 1
        else:
            raise ProtocolAnomaly("%s: unexpected audit n_kv=%d" % (tag, n_kv))
    if len(real) != 28:
        raise ProtocolAnomaly(
            "%s: %d real-decode audit lines (expected 28)" % (tag, len(real)))
    for pair in range(14):
        o, r = real[2 * pair], real[2 * pair + 1]
        ok = (o.group(1) == "owner" and r.group(1) == "reuse"
              and int(o.group(2)) == 2 * pair
              and int(r.group(2)) == 2 * pair + 1
              and int(r.group(3)) == 2 * pair
              and int(o.group(5)) == K and int(r.group(5)) == K
              and o.group(6) == r.group(6))
        if not ok:
            raise ProtocolAnomaly(
                "%s: audit sweep/pairing FAIL at pair %d (owner=%r reuse=%r)" %
                (tag, pair, o.group(0), r.group(0)))
    if stderr_text.count(MASK_LITERAL) != 1:
        raise ProtocolAnomaly(
            "%s: real-data mask line count %d != 1" %
            (tag, stderr_text.count(MASK_LITERAL)))
    return reserve


def forced_positions(p):
    return np.unique(np.concatenate(
        [np.arange(N_INIT), np.arange(p - N_LOCAL + 1, p + 1)]))


def effective_bitmap(idx):
    """Reconstructed effective attention mask [N_TOKENS, EXPECT_N_KV_LID]:
    selected AND causally visible (single sequence, contiguous fill: cell j
    holds position j; visible(p) = {0..p})."""
    rows = np.repeat(np.arange(N_TOKENS)[:, None], K, axis=1)
    visible_sel = idx <= rows
    bitmap = np.zeros((N_TOKENS, EXPECT_N_KV_LID), dtype=bool)
    bitmap[rows[visible_sel], idx[visible_sel]] = True
    return bitmap


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--s-dir", action="append", required=True,
                    help="Type-S run dir (pass 3x, order S1 S2 S3)")
    ap.add_argument("--p-dir", action="append", required=True,
                    help="Type-P run dir (pass 3x, order P1 P2 P3)")
    ap.add_argument("--out-dir", default="lsa_determinism_2050",
                    help="fresh result dir for verdict.json (refused if it exists)")
    args = ap.parse_args()

    if len(args.s_dir) != 3 or len(args.p_dir) != 3:
        print("REFUSED: exactly three --s-dir and three --p-dir are required")
        return 4
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        print("REFUSED: result dir already exists: %s (fresh dir per verdict; "
              "never overwrite)" % out_dir)
        return 4

    s_dirs = [Path(d) for d in args.s_dir]
    p_dirs = [Path(d) for d in args.p_dir]
    report = {
        "protocol": "2050 Type-S/Type-P determinism (pre-registered semantics)",
        "s_dirs": [str(d) for d in s_dirs],
        "p_dirs": [str(d) for d in p_dirs],
        "reasons": [],
        "anomaly": None,
    }
    review_reasons = set()

    try:
        # ------------------------------------------------------ Phase 0
        provs = {}
        logs = {}
        for d in s_dirs + p_dirs:
            if not d.is_dir():
                raise ProtocolAnomaly("run dir missing: %s" % d)
            prov = load_provenance(d)
            tag = str(prov.get("tag", d.name))
            provs[str(d)] = prov
            logs[str(d)] = verify_log_binding(tag, prov)

        ref_prov = provs[str(s_dirs[0])]
        identity = {}
        for field in PLACEMENT_IDENTITY_FIELDS:
            values = {str(d): provs[str(d)].get(field) for d in s_dirs + p_dirs}
            identity[field] = values
            if any(v is None for v in values.values()):
                raise ProtocolAnomaly("provenance field %s missing in some run" % field)
            if len({json.dumps(v, sort_keys=True) for v in values.values()}) != 1:
                raise ProtocolAnomaly(
                    "cross-run identity FAIL on %s: %s (attribution invalid)" %
                    (field, values))
        bin_sets = {str(d): provs[str(d)].get("binaries") for d in s_dirs + p_dirs}
        if len({json.dumps(v, sort_keys=True) for v in bin_sets.values()}) != 1:
            raise ProtocolAnomaly("cross-run binary-set identity FAIL")
        n_kv_lid = int(ref_prov["n_kv_lid_real"])
        if n_kv_lid != EXPECT_N_KV_LID:
            raise ProtocolAnomaly(
                "recorded n_kv_lid_real %d != expected %d" %
                (n_kv_lid, EXPECT_N_KV_LID))
        report["placement_identity"] = identity

        s_expected = {}
        for name, _t, ne0, _st, _pv in S_SURFACES:
            s_expected[name] = ne0 * N_TOKENS * 4
            s_expected[name[:-4] + ".json"] = None
        for name, _t, _o in TOPK_FILES:
            s_expected[name] = K * N_TOKENS * 4
            s_expected[name[:-4] + ".json"] = None

        reserve_counts = {}
        for d in s_dirs:
            tag = provs[str(d)]["tag"]
            verify_inventory_and_manifest(tag, d, s_expected)
            verify_sidecars(tag, d)
            _out_text, err_text = logs[str(d)]
            reserve_counts[str(d)] = verify_structural_evidence(tag, err_text)
            tok_sha = str(provs[str(d)].get(
                "token_stream_sha256_reconstructed", "")).lower()
            if tok_sha != EXPECT_TOKENS_SHA:
                raise ProtocolAnomaly(
                    "%s: reconstructed token stream sha %s != %s" %
                    (tag, tok_sha, EXPECT_TOKENS_SHA))

        p_logits_paths = []
        p_tokens_shas = []
        for d in p_dirs:
            tag = provs[str(d)]["tag"]
            files = {f.name: f for f in d.iterdir() if f.is_file()}
            tokens = [n for n in files if n.endswith("-tokens.bin")]
            bins = [n for n in files
                    if n.endswith(".bin") and not n.endswith("-tokens.bin")]
            prompts = [n for n in files if n.endswith("-prompt.txt")]
            txts = [n for n in files
                    if n.endswith(".txt") and not n.endswith("-prompt.txt")
                    and n != "SHA256SUMS.txt"]
            if not (len(tokens) == 1 and len(bins) == 1
                    and len(prompts) == 1 and len(txts) == 1):
                raise ProtocolAnomaly(
                    "%s: Type-P artifact classes wrong (tokens=%d logits=%d "
                    "prompt=%d txt=%d)" %
                    (tag, len(tokens), len(bins), len(prompts), len(txts)))
            p_expected = {tokens[0]: EXPECT_TOKENS_BYTES,
                          bins[0]: EXPECT_LOGITS_BYTES,
                          prompts[0]: None, txts[0]: None}
            verify_inventory_and_manifest(tag, d, p_expected)
            _out_text, err_text = logs[str(d)]
            reserve_counts[str(d)] = verify_structural_evidence(tag, err_text)
            tok_sha = sha256_file(d / tokens[0])
            if tok_sha != EXPECT_TOKENS_SHA:
                raise ProtocolAnomaly(
                    "%s: tokens bin sha %s != %s" %
                    (tag, tok_sha, EXPECT_TOKENS_SHA))
            p_tokens_shas.append(tok_sha)
            logits = np.fromfile(d / bins[0], dtype="<f4")
            if logits.size != VOCAB:
                raise ProtocolAnomaly(
                    "%s: logits count %d != %d" % (tag, logits.size, VOCAB))
            nonfinite = int((~np.isfinite(logits)).sum())
            if nonfinite != 0:
                raise ProtocolAnomaly(
                    "%s: %d nonfinite logits (protocol-invalid, not a "
                    "determinism verdict)" % (tag, nonfinite))
            p_logits_paths.append(d / bins[0])
        report["phase0"] = {
            "log_binding": "ok",
            "inventory_manifest_rehash": "ok",
            "sidecars_22_semantic": "ok",
            "structural_evidence": "ok",
            "sp_input_identity": EXPECT_TOKENS_SHA,
            "p_logits_finiteness": "ok",
            "reserve_audit_line_counts": reserve_counts,
        }

        # ------------------------------------------------------ Phase 1
        # Structural top-K validity FIRST (round-stopping, no verdict).
        topk = {}  # (s_index, owner) -> int64 [N_TOKENS, K]
        for si, d in enumerate(s_dirs):
            tag = provs[str(d)]["tag"]
            for name, _tensor, owner in TOPK_FILES:
                raw = np.fromfile(d / name, dtype="<f4")
                if raw.size != N_TOKENS * K:
                    raise ProtocolAnomaly(
                        "%s/%s: element count %d != %d" %
                        (tag, name, raw.size, N_TOKENS * K))
                raw = raw.reshape(N_TOKENS, K)
                if not np.isfinite(raw).all():
                    raise ProtocolAnomaly("%s/%s: nonfinite index values" % (tag, name))
                rounded = np.rint(raw)
                if not (raw == rounded).all():
                    bad = int((raw != rounded).sum())
                    raise ProtocolAnomaly(
                        "%s/%s: %d values fail exact lrint recovery" %
                        (tag, name, bad))
                idx = rounded.astype(np.int64)
                if idx.min() < 0 or idx.max() >= n_kv_lid:
                    raise ProtocolAnomaly(
                        "%s/%s: index out of [0, %d): min=%d max=%d" %
                        (tag, name, n_kv_lid, int(idx.min()), int(idx.max())))
                srt = np.sort(idx, axis=1)
                if not (np.diff(srt, axis=1) != 0).all():
                    dup_rows = int((~(np.diff(srt, axis=1) != 0).all(axis=1)).sum())
                    raise ProtocolAnomaly(
                        "%s/%s: duplicate indices in %d rows" %
                        (tag, name, dup_rows))
                topk[(si, owner)] = idx
        report["phase1_structural_validity"] = "ok (3 runs x 14 owners x %d rows)" % N_TOKENS

        # ------------------------------------------------------ Phase 2
        # Per-run positive expectations (failure => exit-2 class).
        expectation_failures = []
        rows_idx = np.arange(N_TOKENS)[:, None]
        for si, d in enumerate(s_dirs):
            tag = provs[str(d)]["tag"]
            for _name, _tensor, owner in TOPK_FILES:
                idx = topk[(si, owner)]
                vis_count = (idx <= rows_idx).sum(axis=1)
                below = np.arange(K)  # rows p <= 2047
                bad_below = np.nonzero(vis_count[below] != below + 1)[0]
                if bad_below.size:
                    p = int(bad_below[0])
                    expectation_failures.append(
                        "%s owner %02d: row %d effective != visible "
                        "(count %d != %d) [+%d more rows]" %
                        (tag, owner, p, int(vis_count[p]), p + 1,
                         bad_below.size - 1))
                for p in (K, K + 1):  # rows 2048, 2049
                    row = idx[p]
                    if not (row <= p).all():
                        expectation_failures.append(
                            "%s owner %02d: sparse row %d raw != effective "
                            "(%d invisible entries)" %
                            (tag, owner, p, int((row > p).sum())))
                        continue
                    forced = forced_positions(p)
                    missing = int((~np.isin(forced, row)).sum())
                    if missing:
                        expectation_failures.append(
                            "%s owner %02d: sparse row %d missing %d forced "
                            "positions" % (tag, owner, p, missing))
        report["phase2_positive_expectations"] = (
            expectation_failures if expectation_failures else "ok")
        if expectation_failures:
            review_reasons.add("positive_expectation_failure")

        # ------------------------------------------------------ Phase 3
        s_tags = [provs[str(d)]["tag"] for d in s_dirs]
        pairs = [(0, 1), (0, 2), (1, 2)]

        # V-input FIRST (mandatory observability) so top-K attribution reads
        # correctly.
        v_input = {"surfaces": {}, "stable": True}
        for name, _tensor, _ne0, _st, provclass in S_SURFACES:
            shas = [sha256_file(d / name) for d in s_dirs]
            equal = len(set(shas)) == 1
            v_input["surfaces"][name] = {
                "sha256": dict(zip(s_tags, shas)),
                "byte_equal": equal,
                "expectation_provenance": provclass,
            }
            if not equal:
                v_input["stable"] = False
        if not v_input["stable"]:
            review_reasons.add("input_variability")
            unstable = [n for n, e in v_input["surfaces"].items()
                        if not e["byte_equal"]]
            anchors = [n for n in unstable if n.startswith("lsa_anchor_")]
            v_input["classification"] = (
                "pre-selection/input variability: indexer inputs are NOT "
                "bitwise stable across S1/S2/S3; resulting top-K membership "
                "variation must NOT be attributed to CUB top-K")
            if anchors:
                v_input["anchor_anomaly"] = (
                    "upstream reproducibility anomaly: attribution anchors "
                    "varied: %s" % anchors)
        else:
            v_input["classification"] = (
                "indexer inputs bitwise stable across S1/S2/S3")
        report["v_input"] = v_input

        # V-ord (characterization only).
        v_ord = {}
        for _name, _tensor, owner in TOPK_FILES:
            per_pair = {}
            for a, b in pairs:
                same = (topk[(a, owner)] == topk[(b, owner)]).all(axis=1)
                diff_rows = np.nonzero(~same)[0]
                per_pair["%s-vs-%s" % (s_tags[a], s_tags[b])] = {
                    "rows_identical": int(same.sum()),
                    "rows_total": N_TOKENS,
                    "first_diff_row": (int(diff_rows[0]) if diff_rows.size else None),
                }
            v_ord["owner%02d" % owner] = per_pair
        report["v_ord"] = {
            "status": "characterization-only (CUB unsorted / not guaranteed)",
            "per_owner": v_ord,
        }

        # V-mem-raw (characterization) + V-mem-effective / V-mask (blocking).
        v_mem_raw = {}
        v_mem_eff = {}
        v_mask = {}
        eff_mask_blocking = False
        for _name, _tensor, owner in TOPK_FILES:
            sorted_idx = {si: np.sort(topk[(si, owner)], axis=1) for si in range(3)}
            bitmaps = {si: effective_bitmap(topk[(si, owner)]) for si in range(3)}
            raw_pair = {}
            eff_pair = {}
            mask_pair = {}
            for a, b in pairs:
                key = "%s-vs-%s" % (s_tags[a], s_tags[b])
                raw_same = (sorted_idx[a] == sorted_idx[b]).all(axis=1)
                raw_diff = np.nonzero(~raw_same)[0]
                invisible_only = 0
                visible_affecting = 0
                examples = []
                for p in raw_diff:
                    sa = set(topk[(a, owner)][p].tolist())
                    sb = set(topk[(b, owner)][p].tolist())
                    sym = sa.symmetric_difference(sb)
                    if all(v > p for v in sym):
                        invisible_only += 1
                    else:
                        visible_affecting += 1
                        if len(examples) < 3:
                            examples.append(int(p))
                raw_pair[key] = {
                    "rows_raw_set_equal": int(raw_same.sum()),
                    "rows_diff_le2047": int((raw_diff < K).sum()),
                    "rows_diff_sparse_2048_2049": int((raw_diff >= K).sum()),
                    "rows_diff_invisible_only_NONBLOCKING": invisible_only,
                    "rows_diff_visible_affecting": visible_affecting,
                    "visible_affecting_examples": examples,
                }
                eff_same = (bitmaps[a] == bitmaps[b]).all(axis=1)
                eff_diff = np.nonzero(~eff_same)[0]
                eff_pair[key] = {
                    "rows_effective_equal": int(eff_same.sum()),
                    "rows_effective_diff": int(eff_diff.size),
                    "first_diff_row": (int(eff_diff[0]) if eff_diff.size else None),
                }
                mask_equal = bool((bitmaps[a] == bitmaps[b]).all())
                mask_pair[key] = {"mask_equal": mask_equal}
                if eff_diff.size or not mask_equal:
                    eff_mask_blocking = True
            v_mem_raw["owner%02d" % owner] = raw_pair
            v_mem_eff["owner%02d" % owner] = eff_pair
            v_mask["owner%02d" % owner] = mask_pair
        report["v_mem_raw"] = {
            "status": "characterization (invisible-only variation NONBLOCKING)",
            "per_owner": v_mem_raw,
        }
        report["v_mem_effective"] = {"status": "BLOCKING", "per_owner": v_mem_eff}
        report["v_mask"] = {"status": "BLOCKING", "per_owner": v_mask}

        # V-logit (blocking): P family only; compared to NOTHING else.
        p_tags = [provs[str(d)]["tag"] for d in p_dirs]
        logit_shas = [sha256_file(p) for p in p_logits_paths]
        top1 = [int(np.argmax(np.fromfile(p, dtype="<f4"))) for p in p_logits_paths]
        logits_equal = len(set(logit_shas)) == 1
        top1_equal = len(set(top1)) == 1
        report["v_logit"] = {
            "status": "BLOCKING",
            "sha256": dict(zip(p_tags, logit_shas)),
            "byte_equal": logits_equal,
            "top1": dict(zip(p_tags, top1)),
            "top1_equal": top1_equal,
            "note": "cross-run C++ determinism only; no HF comparison exists in this round",
        }
        logit_blocking = (not logits_equal) or (not top1_equal)

        if eff_mask_blocking or logit_blocking:
            review_reasons.add("blocking_variation")

    except ProtocolAnomaly as exc:
        report["anomaly"] = str(exc)
        out_dir.mkdir(parents=True)
        with open(out_dir / "verdict.json", "w", encoding="ascii") as f:
            json.dump(report, f, indent=2)
        print("PROTOCOL ANOMALY (exit 3, no determinism verdict): %s" % exc)
        print("verdict record: %s" % (out_dir / "verdict.json"))
        return 3

    report["reasons"] = sorted(review_reasons)
    out_dir.mkdir(parents=True)
    with open(out_dir / "verdict.json", "w", encoding="ascii") as f:
        json.dump(report, f, indent=2)

    print("== 2050 determinism comparator summary ==")
    print("phase 0 (binding/inventory/rehash/sidecars/evidence/finiteness): ok")
    print("phase 1 structural top-K validity: ok")
    print("phase 2 positive expectations: %s" %
          ("ok" if not expectation_failures else
           "%d FAILURES" % len(expectation_failures)))
    print("V-input: %s" % ("stable" if v_input["stable"] else "VARIED (input_variability)"))
    print("V-ord / V-mem-raw: characterization recorded (never blocking)")
    print("V-mem-effective / V-mask: %s" %
          ("deterministic" if not eff_mask_blocking else "VARIED (BLOCKING)"))
    print("V-logit: byte_equal=%s top1_equal=%s top1=%s" %
          (logits_equal, top1_equal, sorted(set(top1))))
    print("verdict record: %s" % (out_dir / "verdict.json"))
    if review_reasons:
        print("VERDICT: STOP FOR REVIEW (exit 2): %s" % sorted(review_reasons))
        return 2
    print("VERDICT: no blocking variation; inputs stable (exit 0) -- "
          "STOP FOR REVIEW is still mandatory before any interpretation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
