# LongCat-Next Stage-0 evidence harness

## Scope

This stage adds read-only checkpoint classification and weight-free reference
fixtures only. It does not implement conversion, GGUF, runtime graphs, loading,
server, MTMD, media, MTP, CUDA, or numerical model parity.

## Accepted inputs

`scripts/longcat-next/inventory.py` requires three local JSON files:

* the pinned LongCat-Next `model.safetensors.index.json`;
* the pinned LongCat-Flash-Lite `model.safetensors.index.json`;
* the pinned LongCat-Next `config.json`.

It optionally accepts `--image-header`, which may be the complete image-decoder
safetensors file or a prefix containing its complete header, and `--hift-metadata`,
a JSON object whose `tensors` maps names to `dtype` and `shape`. It deliberately does
not deserialize a `.pt` pickle. All inputs are read-only.

The fixture generator requires local official source, config, and tokenizer-config
paths plus all three pinned revisions. A Git source checkout must be clean and at the
recorded commit. A non-Git snapshot must have an ASCII `.longcat-next-revision`
marker. Mutable names such as `main`, abbreviated commits, dirty checkouts, missing
revision records, and mismatched commits are rejected. `--model-dir` reserves the
future local checkpoint interface; the current `core` mode refuses to write until
numerical tolerances are frozen.

## Commands used

Fetch only revision-pinned metadata (not model shards):

```bash
curl -fL -o /tmp/next-index.json \
  https://huggingface.co/meituan-longcat/LongCat-Next/resolve/0cf0631862402ff36366e513e4023d22e7e5c84c/model.safetensors.index.json
curl -fL -o /tmp/lite-index.json \
  https://huggingface.co/meituan-longcat/LongCat-Flash-Lite/resolve/b62b68827ead0b7fef3ba98b57f18484acaaec06/model.safetensors.index.json
curl -fL -o /tmp/next-config.json \
  https://huggingface.co/meituan-longcat/LongCat-Next/resolve/0cf0631862402ff36366e513e4023d22e7e5c84c/config.json
python3 scripts/longcat-next/inventory.py \
  --next-index /tmp/next-index.json \
  --lite-index /tmp/lite-index.json \
  --config /tmp/next-config.json
```

Optional header verification uses a complete revision-pinned header range:

```bash
curl -fL --range 0-1048575 -o /tmp/image-header.bin \
  https://huggingface.co/meituan-longcat/LongCat-Next/resolve/0cf0631862402ff36366e513e4023d22e7e5c84c/image_decoder/image_decoder.safetensors
python3 scripts/longcat-next/inventory.py \
  --next-index /tmp/next-index.json --lite-index /tmp/lite-index.json \
  --config /tmp/next-config.json --image-header /tmp/image-header.bin
```

Generate the checked weight-free fixture from a pinned local source snapshot:

```bash
python3 scripts/longcat-next/make-reference-fixtures.py \
  --official-source /path/to/pinned/LongCat-Next \
  --source-revision 0cf0631862402ff36366e513e4023d22e7e5c84c \
  --inference-revision 70ab100beecaaa77c1f45c0cf9ec89a4faf20fd8 \
  --model-revision 0cf0631862402ff36366e513e4023d22e7e5c84c \
  --config /path/to/pinned/config.json \
  --tokenizer-config /path/to/pinned/tokenizer_config.json \
  --output-dir tests/fixtures/longcat-next
sha256sum tests/fixtures/longcat-next/ngram-cases.json
```

Run Stage-0 checks:

```bash
python3 -m py_compile scripts/longcat-next/inventory.py \
  scripts/longcat-next/make-reference-fixtures.py \
  tests/test-longcat-next-evidence.py
python3 tests/test-longcat-next-evidence.py
git diff --check
```

## Verified now

The inventory checker asserts, rather than warns about, all of the following:

* 13,450 Next main names and 150,825,367,872 tensor payload bytes;
* zero Next and exactly 17 Lite `model.mtp.*` names;
* vocabulary extents 131072, 131125, and 282624;
* main counts 11143/425/71/1740/71;
* modality subfamily counts 385/5/30/5/71 and 487/31/149/163/910/71;
* all 11,143 text/trunk names occur in Lite;
* every modal name matches exactly one documented subfamily.

If supplied, the image header is checked for its exact length, 1,771 BF16 entries,
three prefix counts, parameter totals, and payload totals. HiFT JSON metadata is
checked for 328 F32 tensors, 20,821,295 parameters, 83,285,180 tensor payload
bytes, and matched weight-normalization pairs.

The checked JSON fixture uses torch but no model weights. The generator AST-isolates
and executes the actual pinned official `_shift_right_ignore_eos`,
`_precompute_vocab_mods`, and `_get_ngram_ids` method bodies. `official_hashes`
fields are produced by those methods; `independent_hashes` fields are produced by
the standalone implementation. Generation fails on any mismatch, including direct,
incremental, ignored-ID, boundary, and independent-history cases. Integer hashes
and masks are exact comparisons and have no tolerance.

## Blocked until official weights are local

No base-embedding, per-table contribution, fused pre-trunk embedding, physical-block
hidden state, final norm, logits, or greedy continuation fixture exists yet. No core
numerical parity is claimed. Future `core` mode will require a local pinned checkpoint,
record every shard hash and installed software version, and emit only small selected
results. The harness never downloads checkpoint shards itself.

## Tolerance freeze policy

`manifest.json` intentionally records BF16 and F16 absolute and relative tolerances
as `null` with status `pending pre-runtime selection`. Reviewers must choose and
approve those values using official-reference variability and dtype error analysis
before any C++ output is inspected. A failed comparison is investigated as a bug or
an explicitly reviewed fixture-policy change; tolerances must never be widened in
response to observed runtime failures. Manifest and fixture hashes make any later
change reviewable.
