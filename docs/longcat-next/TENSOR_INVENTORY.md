# LongCat-Next tensor inventory

## Main checkpoint

Source: official `model.safetensors.index.json` at HF revision `0cf06318`.

| Family selector | Names |
|---|---:|
| everything except the four modal prefixes below | 11,143 |
| `model.visual_tokenizer.*` | 425 |
| `visual_head.*` | 71 |
| `model.audio_tokenizer.*` | 1,740 |
| `audio_head.*` | 71 |
| **Total** | **13,450** |

Index metadata is **150,825,367,872 bytes**. HF metadata reports:

| dtype | Parameters | Payload bytes |
|---|---:|---:|
| BF16 | 73,101,777,568 | 146,203,555,136 |
| F32 | 1,155,453,184 | 4,621,812,736 |
| **Total** | **74,257,230,752** | **150,825,367,872** |

MTP query results: Next has zero names beginning `model.mtp.`; Flash-Lite has 17.
The Next HF repository has no sibling filename containing `mtp`.

Key exact names/shapes derived from config plus indexed headers are:

* `model.embed_tokens.weight` — `[282624,3072]`.
* `lm_head.weight` — `[131125,3072]`.
* `model.ngram_embeddings.embedders.{0..11}.weight` —
  `[10223616+2*i+1,256]`.
* `model.ngram_embeddings.post_projs.{0..11}.weight` — `[3072,256]`.
* `visual_head.heads.{0..7}.{weight,bias}` and
  `audio_head.heads.{0..7}.{weight,bias}` include each level's extra class.

## External image decoder

Command:

```bash
curl -L --range 0-1048575 \
  https://huggingface.co/meituan-longcat/LongCat-Next/resolve/main/image_decoder/image_decoder.safetensors \
  -o /tmp/image-head.bin
```

The first little-endian uint64 is `226408`; parsing that many following JSON bytes
produces 1,771 entries. All are BF16.

| Prefix | Tensors | Parameters | Payload bytes |
|---|---:|---:|---:|
| `image_decoder.*` | 558 | 433,743,858 | 867,487,716 |
| `image_refiner.*` | 828 | 4,058,323,163 | 8,116,646,326 |
| `visual_model.*` | 385 | 631,975,680 | 1,263,951,360 |
| **Total** | **1,771** | **5,124,042,701** | **10,248,085,402** |

Payload + 226,408-byte JSON header + 8-byte header length equals the exact
10,248,311,818-byte file. Representative exact entries:

```text
image_decoder.decoder_head.0.bias                 BF16 [2730]
image_decoder.decoder_head.0.weight               BF16 [2730,1024]
image_decoder.decoder_head.2.bias                 BF16 [1176]
image_decoder.decoder_head.2.weight               BF16 [1176,2730]
image_refiner.base_transformer.transformer_blocks.0.attn.to_q.weight
image_refiner.cond_proj.weight
image_refiner.vae.encoder.conv_in.weight
visual_model.blocks.0.attn.qkv.weight
```

The `visual_model.*` family is significant: conversion must determine by payload
hash/comparison whether it intentionally duplicates the input visual encoder and
whether an output-only package needs it. It must not be silently dropped based only
on the Work audit's earlier prefix list.

## External HiFT checkpoint

File: `cosy24k_vocoder/hift.pt`, exactly 83,364,158 bytes. It is a PyTorch ZIP
archive whose root pickle object is an ordered state dict of **328 tensors**. The
storage classes are FloatStorage, hence all stored tensors are F32.

Representative exact names and shapes:

```text
m_source.l_linear.weight                         [1,9]
m_source.l_linear.bias                           [1]
conv_pre.bias                                    [512]
conv_pre.weight_g                                [512,1,1]
conv_pre.weight_v                                [512,80,7]
ups.0.bias                                       [256]
ups.0.weight_g                                   [512,1,1]
ups.0.weight_v                                   [512,256,16]
ups.1.weight_v                                   [256,128,11]
ups.2.weight_v                                   [128,64,7]
source_downs.0.weight                            [256,18,30]
source_downs.1.weight                            [128,18,6]
source_downs.2.weight                            [64,18,1]
source_resblocks.0.convs1.0.weight_g             [256,1,1]
source_resblocks.0.convs1.0.weight_v             [256,256,7]
source_resblocks.0.activations1.0.alpha           [256,1]
conv_post.weight_g                               (weight-normalized)
conv_post.weight_v                               (weight-normalized)
f0_predictor.condnet.0.weight_g                  (weight-normalized)
f0_predictor.condnet.0.weight_v                  (weight-normalized)
f0_predictor.classifier.weight                   (ordinary weight)
f0_predictor.classifier.bias                     (ordinary bias)
```

The exact conversion rule for every `weight_g`/`weight_v` pair is PyTorch weight
normalization along the module's configured dimension. Do not assume a universal
axis from filename alone: instantiate the official `HiFTGenerator`, load the state
dict, call weight-norm removal (or reproduce its formula per module), and compare
folded outputs before writing GGUF. Snake `*.alpha` tensors must remain explicit.

## Reproduction snippets

```python
# Main family counts
import json, collections
w = json.load(open('/tmp/next-index.json'))['weight_map']

# Safetensors header only
import struct, json
with open('/tmp/image-head.bin', 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(n))
```

For HiFT without importing torch, read `hift/data.pkl` from the ZIP with a restricted
metadata unpickler that records `_rebuild_tensor_v2` size/stride and persistent
storage type; no tensor storage bytes need to be materialized.

