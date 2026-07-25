from __future__ import annotations

import json
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf
from .longcat_flash_ngram import LongcatFlashNgramModel
from .longcat_next_inventory import (
    CORE_VOCAB_SIZE,
    HASH_VOCAB_SIZE,
    IGNORED_COUNT,
    IGNORED_START,
    MODAL_PREFIX_COUNTS,
    SOURCE_VOCAB_SIZE,
    classify_longcat_next_names,
)


@ModelBase.register("LongcatNextForCausalLM")
class LongcatNextModel(LongcatFlashNgramModel):
    """Provisional text-core-only LongCat-Next converter."""

    model_arch = gguf.MODEL_ARCH.LONGCAT_NEXT
    HASH_VOCAB_SIZE = HASH_VOCAB_SIZE
    CORE_VOCAB_SIZE = CORE_VOCAB_SIZE
    SOURCE_VOCAB_SIZE = SOURCE_VOCAB_SIZE
    IGNORED_START = IGNORED_START
    IGNORED_COUNT = IGNORED_COUNT

    MODAL_PREFIXES = tuple(MODAL_PREFIX_COUNTS)

    @classmethod
    def classify_source_names(cls, names: set[str]) -> tuple[set[str], set[str]]:
        return classify_longcat_next_names(names)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mtp_count != 0:
            raise ValueError("LongCat-Next text-core conversion rejects all model.mtp.* tensors")

        index_path = self.dir_model / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = set(index.get("weight_map", {}))
        self.classify_source_names(names)

    def set_gguf_parameters(self):
        source_vocab = int(self.hparams["vocab_size"])
        if source_vocab != self.SOURCE_VOCAB_SIZE:
            raise ValueError(f"LongCat-Next source vocab must be {self.SOURCE_VOCAB_SIZE}, got {source_vocab}")
        if int(self.hparams.get("text_vocab_size", -1)) != self.HASH_VOCAB_SIZE:
            raise ValueError("LongCat-Next text/hash vocabulary extent mismatch")
        if int(self.hparams.get("text_vocab_plus_multimodal_special_token_size", -1)) != self.CORE_VOCAB_SIZE:
            raise ValueError("LongCat-Next input/output vocabulary extent mismatch")
        if float(self.hparams.get("rope_theta", 0)) != 10000000.0:
            raise ValueError("LongCat-Next requires plain RoPE theta 10000000")

        self.hparams["vocab_size"] = self.CORE_VOCAB_SIZE
        super().set_gguf_parameters()
        self.gguf_writer.add_ngram_hash_vocab_size(self.HASH_VOCAB_SIZE)
        self.gguf_writer.add_ngram_input_output_size(self.CORE_VOCAB_SIZE)
        self.gguf_writer.add_ngram_source_vocab_size(self.SOURCE_VOCAB_SIZE)
        self.gguf_writer.add_ngram_ignored_start(self.IGNORED_START)
        self.gguf_writer.add_ngram_ignored_count(self.IGNORED_COUNT)

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("model.mtp."):
            raise ValueError(f"LongCat-Next rejects MTP tensor {name!r}")
        if name.startswith(self.MODAL_PREFIXES):
            return
        if name == "model.embed_tokens.weight":
            if data_torch.shape[0] != self.SOURCE_VOCAB_SIZE:
                raise ValueError(
                    f"LongCat-Next source embedding rows must be {self.SOURCE_VOCAB_SIZE}, "
                    f"got {data_torch.shape[0]}")
            data_torch = data_torch[:self.CORE_VOCAB_SIZE].contiguous()
        if name == "lm_head.weight" and data_torch.shape[0] != self.CORE_VOCAB_SIZE:
            raise ValueError(
                f"LongCat-Next lm_head rows must be {self.CORE_VOCAB_SIZE}, got {data_torch.shape[0]}")
        yield from super().modify_tensors(data_torch, name, bid)
