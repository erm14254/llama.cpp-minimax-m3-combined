from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("LongcatFlashNgramForCausalLM")
class LongcatFlashNgramModel(TextModel):
    model_arch = gguf.MODEL_ARCH.LONGCAT_FLASH_NGRAM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Each logical HF layer has two sub-blocks in the GGUF representation.
        n_layers = self.find_hparam(["num_hidden_layers", "num_layers", "n_layers", "n_layer"])
        self.block_count = 2 * int(n_layers)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_vocab(self):
        self._set_vocab_gpt2()

        # The LongCat tokenizer config enables automatic EOS insertion. Disable it
        # for llama.cpp prompt processing.
        self.gguf_writer.add_add_eos_token(False)

    def set_gguf_parameters(self):
        hparams = self.hparams

        # MLA is represented as MQA with one compressed KV head in GGUF.
        hparams["num_key_value_heads"] = 1

        super().set_gguf_parameters()

        self.gguf_writer.add_feed_forward_length(hparams["ffn_hidden_size"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        if hparams.get("q_lora_rank") is not None:
            self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])

        self.gguf_writer.add_kv_lora_rank(hparams["kv_lora_rank"])
        self.gguf_writer.add_key_length(
            hparams["kv_lora_rank"] + hparams["qk_rope_head_dim"]
        )
        self.gguf_writer.add_value_length(hparams["kv_lora_rank"])
        self.gguf_writer.add_key_length_mla(
            hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"]
        )
        self.gguf_writer.add_value_length_mla(hparams["v_head_dim"])

        self.gguf_writer.add_expert_feed_forward_length(
            hparams["expert_ffn_hidden_size"]
        )
        self.gguf_writer.add_expert_count(hparams["n_routed_experts"])
        self.gguf_writer.add_expert_shared_count(1)
        self.gguf_writer.add_expert_used_count(hparams["moe_topk"])
        self.gguf_writer.add_expert_weights_scale(
            hparams["routed_scaling_factor"]
        )
        self.gguf_writer.add_expert_zero_count(hparams["zero_expert_num"])
        self.gguf_writer.add_leading_dense_block_count(0)

        self.gguf_writer.add_ngram_neighbor_num(hparams["emb_neighbor_num"])
        self.gguf_writer.add_ngram_split_num(hparams["emb_split_num"])
        self.gguf_writer.add_ngram_vocab_size_ratio(
            hparams["ngram_vocab_size_ratio"]
        )

        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])

        if (
            rope_mscale_all := self.rope_parameters.get("mscale_all_dim")
        ) is not None:
            self.gguf_writer.add_rope_scaling_yarn_log_mul(
                0.1 * rope_mscale_all
            )

    _experts: list[dict[str, Tensor]] | None = None

    def _remap_double_block(
        self, name: str, bid: int | None
    ) -> tuple[str, int | None]:
        """Map each logical LongCat layer to two GGUF blocks."""

        match = re.match(
            r"model\.layers\.(\d+)\.self_attn\.(\d+)\.(.*)", name
        )
        if match:
            new_bid = 2 * int(match.group(1)) + int(match.group(2))
            return (
                f"model.layers.{new_bid}.self_attn.{match.group(3)}",
                new_bid,
            )

        match = re.match(
            r"model\.layers\.(\d+)\."
            r"(input_layernorm|post_attention_layernorm)\.(\d+)\.(.*)",
            name,
        )
        if match:
            new_bid = 2 * int(match.group(1)) + int(match.group(3))
            return (
                f"model.layers.{new_bid}.{match.group(2)}.{match.group(4)}",
                new_bid,
            )

        match = re.match(
            r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.*)", name
        )
        if match:
            new_bid = 2 * int(match.group(1))
            return (
                f"model.layers.{new_bid}.mlp.experts."
                f"{match.group(2)}.{match.group(3)}",
                new_bid,
            )

        match = re.match(
            r"model\.layers\.(\d+)\.mlp\.router\.classifier\.(.*)", name
        )
        if match:
            new_bid = 2 * int(match.group(1))
            return (
                f"model.layers.{new_bid}.mlp.gate.{match.group(2)}",
                new_bid,
            )

        match = re.match(
            r"model\.layers\.(\d+)\.mlp\.router\."
            r"e_score_correction_bias",
            name,
        )
        if match:
            new_bid = 2 * int(match.group(1))
            return (
                f"model.layers.{new_bid}.mlp.gate."
                "e_score_correction.bias",
                new_bid,
            )

        match = re.match(r"model\.layers\.(\d+)\.mlps\.0\.(.*)", name)
        if match:
            new_bid = 2 * int(match.group(1))
            return (
                f"model.layers.{new_bid}.mlp.shared_experts."
                f"{match.group(2)}",
                new_bid,
            )

        match = re.match(r"model\.layers\.(\d+)\.mlps\.1\.(.*)", name)
        if match:
            new_bid = 2 * int(match.group(1)) + 1
            return (
                f"model.layers.{new_bid}.mlp.{match.group(2)}",
                new_bid,
            )

        return name, bid

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("model.mtp"):
            return

        # These names are already handled by the LongCat tensor map.
        if "ngram_embeddings" in name:
            yield from super().modify_tensors(data_torch, name, bid)
            return

        name, bid = self._remap_double_block(name, bid)

        # Split the MLA KV-B projection into separate K-B and V-B tensors.
        if name.endswith("kv_b_proj.weight"):
            name_kb = name.replace("kv_b_proj", "k_b_proj")
            name_vb = name.replace("kv_b_proj", "v_b_proj")

            n_heads = self.hparams["num_attention_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]

            expected = n_heads * (v_head_dim + qk_nope_head_dim)
            assert data_torch.shape[0] == expected

            kv_b = data_torch.view(
                n_heads,
                v_head_dim + qk_nope_head_dim,
                data_torch.shape[-1],
            )
            k_b, v_b = torch.split(
                kv_b,
                [qk_nope_head_dim, v_head_dim],
                dim=1,
            )
            k_b = k_b.transpose(1, 2)

            yield from super().modify_tensors(k_b, name_kb, bid)
            yield from super().modify_tensors(v_b, name_vb, bid)
            return

        # Merge individual expert weights into llama.cpp's 3-D tensors.
        if (
            "mlp.experts" in name
            and re.search(r"mlp\.experts\.\d+\.", name)
        ):
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) < n_experts * 3:
                return

            for weight_name in ("down_proj", "gate_proj", "up_proj"):
                tensors: list[Tensor] = []

                for expert_id in range(n_experts):
                    expert_name = (
                        f"model.layers.{bid}.mlp.experts."
                        f"{expert_id}.{weight_name}.weight"
                    )
                    tensors.append(self._experts[bid][expert_name])
                    del self._experts[bid][expert_name]

                merged = torch.stack(tensors, dim=0)
                merged_name = (
                    f"model.layers.{bid}.mlp.experts."
                    f"{weight_name}.weight"
                )
                yield from super().modify_tensors(
                    merged, merged_name, bid
                )
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            remaining = [
                name
                for block_experts in self._experts
                for name in block_experts
            ]
            if remaining:
                raise ValueError(f"Unprocessed experts: {remaining}")
