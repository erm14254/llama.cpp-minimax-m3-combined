#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

EXPECTED_HEAD = "cb7729cf18088ae6cd6d9cac52e3ee536be02dc4"
TARGETS = (
    "common/debug.cpp",
    "examples/debug/debug.cpp",
)
MARKER = "LONGCAT_GATE4_NAN_AUDIT"

def fail(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")

def run(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    p = subprocess.run(
        list(args),
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and p.returncode != 0:
        fail(f"command failed ({' '.join(args)}):\n{p.stdout}{p.stderr}")
    return p

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        fail(f"{label}: expected anchor exactly once, found {n}")
    return text.replace(old, new, 1)

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Temporarily make llama-debug tensor filtering selective and abort on the first selected NaN."
    )
    ap.add_argument("--root", default=".")
    ns = ap.parse_args()
    root = Path(ns.root).resolve()

    if not (root / ".git").is_dir():
        fail(f"not a git checkout: {root}")

    branch = run(root, "git", "branch", "--show-current").stdout.strip()
    if branch != "longcat-sparse":
        fail(f"expected branch longcat-sparse, got {branch!r}")

    head = run(root, "git", "rev-parse", "HEAD").stdout.strip()
    if head != EXPECTED_HEAD:
        fail(f"expected HEAD {EXPECTED_HEAD}, got {head}")

    staged = run(root, "git", "diff", "--cached", "--name-only").stdout.strip()
    if staged:
        fail(f"staged changes present; refusing diagnostic patch:\n{staged}")

    dirty = run(root, "git", "diff", "--name-only", "--", *TARGETS).stdout.strip()
    if dirty:
        fail(f"diagnostic target files already modified; refusing to stack changes:\n{dirty}")

    debug_common = root / "common/debug.cpp"
    debug_example = root / "examples/debug/debug.cpp"
    for path in (debug_common, debug_example):
        if not path.is_file():
            fail(f"missing target: {path.relative_to(root)}")

    common = debug_common.read_text(encoding="utf-8")
    example = debug_example.read_text(encoding="utf-8")

    if MARKER in common or MARKER in example:
        fail("NaN audit instrumentation already present")

    common = replace_once(
        common,
        '''#include <cmath>
#include <regex>
#include <string>
#include <vector>
''',
        '''#include <cmath>
#include <cstdlib>
#include <regex>
#include <string>
#include <vector>
''',
        "common/debug.cpp include anchor",
    )

    common = replace_once(
        common,
        '''static void common_debug_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n, bool abort_on_nan) {
    GGML_ASSERT(n > 0);
    float sum = 0;
''',
        '''// LONGCAT_GATE4_NAN_AUDIT: return whether any actual element is NaN.
// Do not infer NaN from the aggregate sum: LongCat LSA score tensors
// legitimately contain both +inf and -inf, whose sum itself is NaN.
static bool common_debug_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    uint64_t nan_count = 0;
''',
        "common_debug_print_tensor signature",
    )

    common = replace_once(
        common,
        '''                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    sum += v;
''',
        '''                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    sum += v;
                    if (std::isnan(v)) {
                        ++nan_count;
                    }
''',
        "NaN counter",
    )

    common = replace_once(
        common,
        '''        LOG(INDENT "sum = %f\\n", sum);
    }

    if (abort_on_nan) {
        if (std::isnan(sum)) {
            LOG("encountered NaN - aborting\\n");
            exit(0);
        }
    }
}
''',
        '''        LOG(INDENT "sum = %f\\n", sum);
    }

    LOG(INDENT "nan_count = %llu\\n", (unsigned long long) nan_count);
    return nan_count != 0;
}
''',
        "NaN decision tail",
    )

    common = replace_once(
        common,
        '''    if (ask) {
        return true;  // Always retrieve data
    }

    bool matches_filter = pimpl->tensor_filters.empty();

    if (!matches_filter) {
        for (const auto & filter : pimpl->tensor_filters) {
            if (std::regex_search(t->name, filter)) {
                matches_filter = true;
                break;
            }
        }
    }
''',
        '''    bool matches_filter = pimpl->tensor_filters.empty();

    if (!matches_filter) {
        for (const auto & filter : pimpl->tensor_filters) {
            if (std::regex_search(t->name, filter)) {
                matches_filter = true;
                break;
            }
        }
    }

    // LONGCAT_GATE4_NAN_AUDIT: at ask time, request only tensors that
    // match --tensor-filter. The stock callback asks for every tensor,
    // forcing needless device-to-host copies even for filtered output.
    if (ask) {
        return matches_filter;
    }
''',
        "selective ask stage",
    )

    common = replace_once(
        common,
        '''    if (!ggml_is_quantized(t->type) && matches_filter) {
        uint8_t * data = is_host ? (uint8_t *) t->data : pimpl->data.data();
        common_debug_print_tensor(data, t->type, t->ne, t->nb, 3, pimpl->abort_on_nan);
    }

    return true;
}
''',
        '''    if (!ggml_is_quantized(t->type) && matches_filter) {
        uint8_t * data = is_host ? (uint8_t *) t->data : pimpl->data.data();
        const bool saw_nan = common_debug_print_tensor(data, t->type, t->ne, t->nb, 3);
        if (pimpl->abort_on_nan && saw_nan) {
            LOG("LONGCAT_GATE4_NAN_AUDIT FIRST_NAN tensor=%s\\n", t->name);
            std::exit(86);
        }
    }

    return true;
}
''',
        "first-NaN abort",
    )

    example = replace_once(
        example,
        '''    if (!params.save_logits) {
        cb_data.emplace(params, params.tensor_filter);
    }
''',
        '''    if (!params.save_logits) {
        // LONGCAT_GATE4_NAN_AUDIT: diagnostic llama-debug runs stop
        // at the first actual NaN in a selected tensor.
        cb_data.emplace(params, params.tensor_filter, true);
    }
''',
        "enable abort_on_nan",
    )

    debug_common.write_text(common, encoding="utf-8", newline="\n")
    debug_example.write_text(example, encoding="utf-8", newline="\n")

    pcheck = run(root, "git", "diff", "--check", "--", *TARGETS, check=False)
    if pcheck.returncode != 0:
        fail(f"git diff --check failed:\n{pcheck.stdout}{pcheck.stderr}")

    for rel in TARGETS:
        text = (root / rel).read_text(encoding="utf-8")
        if MARKER not in text:
            fail(f"diagnostic marker missing after patch: {rel}")

    print("GATE-4 SELECTIVE FIRST-NAN DEBUG PATCH: APPLIED")
    print(f"branch {branch}")
    print(f"HEAD {head}")
    print("git diff --check PASS")
    for rel in TARGETS:
        print(f"{rel} SHA256 {sha256_file(root / rel)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
