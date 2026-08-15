#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


MARKER = "LONGCAT_HIDDEN_VECTOR_DUMP"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ns = ap.parse_args()

    repo = Path(ns.repo).resolve()
    path = repo / "common" / "debug.cpp"

    if not path.is_file():
        stop(f"missing source: {path}")

    raw = path.read_bytes()
    before_sha = sha256(raw)

    newline = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf-8").replace("\r\n", "\n")

    print(f"target={path}")
    print(f"before_sha256={before_sha}")

    if MARKER in text:
        stop("hidden-vector dump patch already appears to be applied")

    required = [
        "LONGCAT_GATE4_NAN_AUDIT",
        "common_debug_print_tensor",
        "common_debug_cb_eval",
    ]
    for token in required:
        if token not in text:
            stop(f"expected diagnostic source marker missing: {token}")

    include_anchor = "#include <cstdlib>\n"
    if text.count(include_anchor) != 1:
        stop("expected exactly one <cstdlib> include anchor")

    extra_includes = (
        "#include <cstdlib>\n"
        "#include <filesystem>\n"
        "#include <fstream>\n"
        "#include <system_error>\n"
    )
    text = text.replace(include_anchor, extra_includes, 1)

    helper_anchor = (
        "/**\n"
        " * GGML operations callback during the graph execution.\n"
    )
    if text.count(helper_anchor) != 1:
        stop("callback documentation anchor not unique")

    helper = r'''
// LONGCAT_HIDDEN_VECTOR_DUMP:
// When LONGCAT_HIDDEN_DUMP_DIR is set, dump the final-token hidden vector
// for the 15 HF-comparable LongCat 512-token diagnostic surfaces.
// Files are always little/native-endian F32, 3072 values = 12288 bytes.
static bool common_debug_longcat_hidden_filename(
        const ggml_tensor * t,
        std::string & filename) {
    const std::string tensor_name = t->name;

    if (tensor_name == "inp_embd_ngram") {
        filename = "inp_embd_ngram.bin";
        return true;
    }

    if (tensor_name == "result_norm") {
        filename = "result_norm.bin";
        return true;
    }

    for (int logical = 0; logical < 13; ++logical) {
        const int physical = 2 * logical + 1;
        if (tensor_name == "l_out-" + std::to_string(physical)) {
            char buf[64];
            snprintf(buf, sizeof(buf), "logical_%02d.bin", logical);
            filename = buf;
            return true;
        }
    }

    return false;
}

static void common_debug_maybe_dump_longcat_hidden(
        uint8_t * data,
        const ggml_tensor * t) {
    const char * dump_dir = std::getenv("LONGCAT_HIDDEN_DUMP_DIR");
    if (dump_dir == nullptr || dump_dir[0] == '\0') {
        return;
    }

    std::string filename;
    if (!common_debug_longcat_hidden_filename(t, filename)) {
        return;
    }

    if (t->ne[0] != 3072 || t->ne[1] < 1 ||
        t->ne[2] != 1 || t->ne[3] != 1) {
        LOG_ERR(
            "LONGCAT_HIDDEN_VECTOR_DUMP bad shape tensor=%s "
            "ne={%lld,%lld,%lld,%lld}\n",
            t->name,
            (long long) t->ne[0],
            (long long) t->ne[1],
            (long long) t->ne[2],
            (long long) t->ne[3]);
        common_log_flush(common_log_main());
        std::exit(87);
    }

    const size_t final_i1 = (size_t) t->ne[1] - 1;

    std::vector<float> row(3072);
    for (size_t i0 = 0; i0 < row.size(); ++i0) {
        row[i0] = common_ggml_get_float_value(
            data, t->type, t->nb, i0, final_i1, 0, 0);

        if (!std::isfinite(row[i0])) {
            LOG_ERR(
                "LONGCAT_HIDDEN_VECTOR_DUMP nonfinite "
                "tensor=%s i0=%zu value=%f\n",
                t->name, i0, row[i0]);
            common_log_flush(common_log_main());
            std::exit(88);
        }
    }

    std::filesystem::path root(dump_dir);
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    if (ec) {
        LOG_ERR(
            "LONGCAT_HIDDEN_VECTOR_DUMP mkdir failed: %s\n",
            ec.message().c_str());
        common_log_flush(common_log_main());
        std::exit(89);
    }

    const auto output_path = root / filename;

    std::ofstream out(
        output_path,
        std::ios::binary | std::ios::trunc);

    if (!out) {
        LOG_ERR(
            "LONGCAT_HIDDEN_VECTOR_DUMP open failed: %s\n",
            output_path.string().c_str());
        common_log_flush(common_log_main());
        std::exit(90);
    }

    out.write(
        reinterpret_cast<const char *>(row.data()),
        (std::streamsize) (row.size() * sizeof(float)));
    out.close();

    if (!out) {
        LOG_ERR(
            "LONGCAT_HIDDEN_VECTOR_DUMP write failed: %s\n",
            output_path.string().c_str());
        common_log_flush(common_log_main());
        std::exit(91);
    }

    LOG(
        "LONGCAT_HIDDEN_VECTOR_DUMP tensor=%s "
        "file=%s final_i1=%zu type=%s\n",
        t->name,
        output_path.string().c_str(),
        final_i1,
        ggml_type_name(t->type));
}

'''
    text = text.replace(helper_anchor, helper + helper_anchor, 1)

    call_anchor = (
        "        uint8_t * data = is_host ? "
        "(uint8_t *) t->data : pimpl->data.data();\n"
        "        const bool saw_nan = "
        "common_debug_print_tensor(data, t->type, t->ne, t->nb, 3);\n"
    )

    if text.count(call_anchor) != 1:
        stop("selected-tensor data anchor not unique")

    call_replacement = (
        "        uint8_t * data = is_host ? "
        "(uint8_t *) t->data : pimpl->data.data();\n"
        "        common_debug_maybe_dump_longcat_hidden(data, t);\n"
        "        const bool saw_nan = "
        "common_debug_print_tensor(data, t->type, t->ne, t->nb, 3);\n"
    )

    text = text.replace(call_anchor, call_replacement, 1)

    out_raw = text.replace("\n", newline).encode("utf-8")
    path.write_bytes(out_raw)

    print(f"after_sha256={sha256(out_raw)}")
    print("LONGCAT HIDDEN VECTOR DUMP PATCH: APPLIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())