from __future__ import annotations


MODAL_PREFIX_COUNTS = {
    "model.visual_tokenizer.": 425,
    "visual_head.": 71,
    "model.audio_tokenizer.": 1740,
    "audio_head.": 71,
}

HASH_VOCAB_SIZE = 131072
CORE_VOCAB_SIZE = 131125
SOURCE_VOCAB_SIZE = 282624
IGNORED_START = 131072
IGNORED_COUNT = 53


def classify_longcat_next_names(names: set[str]) -> tuple[set[str], set[str]]:
    mtp = {name for name in names if name.startswith("model.mtp.")}
    if mtp:
        raise ValueError(f"LongCat-Next checkpoint unexpectedly contains {len(mtp)} MTP tensors")
    deferred = {name for name in names if name.startswith(tuple(MODAL_PREFIX_COUNTS))}
    core = names - deferred
    if len(names) != 13450 or len(core) != 11143 or len(deferred) != 2307:
        raise ValueError(
            f"LongCat-Next inventory mismatch: total={len(names)}, "
            f"core={len(core)}, deferred={len(deferred)}")
    actual = {
        prefix: sum(name.startswith(prefix) for name in deferred)
        for prefix in MODAL_PREFIX_COUNTS
    }
    if actual != MODAL_PREFIX_COUNTS:
        raise ValueError(f"LongCat-Next deferred modality inventory mismatch: {actual}")
    return core, deferred
