# LongCat-Next local core reference fixtures

## Purpose and scope

This Stage-0B tool generates small derived activations from an already present,
pinned official LongCat-Next checkpoint. It validates the checkpoint before model
construction and runs only the official Transformers implementation. It does not
download a model, implement llama.cpp, establish numerical parity, or select BF16
or F16 comparison tolerances. Generated weight-backed files remain local and are
not committed.

Codex Cloud did not have the checkpoint and did not execute core generation. The
commands and the 88 GiB GPU / 220 GiB CPU defaults are unproven workstation starting
points, not a demonstrated 96 GiB deployment profile.

## Local prerequisites

The checkpoint directory must contain the official pinned files, including
config/tokenizer files, pinned generation_config.json, 15 referenced safetensors
shards, and all custom Python code.
Inspection enforces 13,450 names, 150,825,367,872 tensor payload bytes, 15 shards,
150,827,115,056 aggregate shard-file bytes, the three vocabulary extents, zero MTP
names, no missing or unreferenced model shards, and pinned custom-code/config/tokenizer
SHA-256 identities.

Use a dedicated 64-bit Python environment. Python 3.10 is the official-environment
provenance, not the only executable Python for native Windows Blackwell. Two runtime
profiles are explicit:

* official-pinned enforces the published package versions on hardware where they
  actually execute;
* blackwell-compatible retains those versions as provenance but requires an actual
  sm_120-capable torch 2.7+ CUDA 12.8+ runtime and records every departure. On
  Windows, the executing Python version must match the installed wheel's Python,
  ABI, platform, PyTorch, CUDA, and Blackwell tags.

The official requirements pin torch
2.6.0, torchvision 0.21.0, torchaudio 2.6.0, Accelerate 1.10.0, Transformers
4.57.6, librosa 0.11.0, diffusers 0.34.0, and flash-attn 2.7.4.post1. Safetensors
and NumPy are also required but are not exactly pinned by the published requirements.
The custom model additionally imports einops; because the official short requirements
do not pin it, the preflight records its installed version as an unpinned transitive
dependency rather than inventing an official version.
Install a PyTorch build appropriate for the local NVIDIA driver and CUDA environment
using current official PyTorch instructions; this document does not invent a PyTorch
wheel index or CUDA build. Native Windows FlashAttention execution is supported
through ABI-matched community wheels, which remain unofficial and must pass the full
preflight below. Core generation must not begin until every check is OK.

### Windows cmd.exe environment setup

Run these commands in an x64 Native Tools Command Prompt. They are cmd.exe commands,
not PowerShell syntax.

```bat
py -0p
py -3.10 -m venv D:\LongCat-Next-fixture-env
call D:\LongCat-Next-fixture-env\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install transformers==4.57.6 accelerate==1.10.0 safetensors numpy librosa==0.11.0 diffusers==0.34.0 einops
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import transformers; assert transformers.__version__ == '4.57.6'"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1
```

The `py -3.10` command recreates the official Python provenance profile. For a native
Windows Blackwell profile, select from `py -0p` the Python interpreter matching the
wheel's CPython tag and use that interpreter in the venv command instead; the tool
records the exact executing version and rejects a tag mismatch. It does not hard-code
a second Python version because available wheel combinations change.

Install mutually compatible torch, torchvision, and torchaudio builds using official
PyTorch instructions. The official LongCat provenance remains torch 2.6.0,
torchvision 0.21.0, torchaudio 2.6.0, and flash-attn 2.7.4.post1. For native Windows
Blackwell, an optional source of newer community/unofficial, Blackwell-specific wheels
is [Flash-Attention-2 for Windows](https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows).
Choose no fixed filename: its Python, PyTorch, CUDA, ABI, platform, and Blackwell tags
must match the environment. This repository never downloads or installs that wheel.
WSL2/Linux is a fallback only if no matching native Windows wheel passes preflight;
it is not required when native preflight succeeds.
Keep the three offline
variables set for inspection and generation. The loader also sets them internally
and passes local_files_only=True, trust_remote_code=True, use_safetensors=True, and
low_cpu_mem_usage=True.

## Checkpoint inspection without model construction

From the repository root:

```bat
python scripts\longcat-next\make-reference-fixtures.py ^
  --mode inspect ^
  --model-dir D:\LongCat-Next
```

This mode runs before importing torch or Transformers. To spend the additional time
reading and hashing every shard:

```bat
python scripts\longcat-next\make-reference-fixtures.py ^
  --mode inspect ^
  --model-dir D:\LongCat-Next ^
  --hash-shards
```

Any mismatch is fatal and is reported before model construction.

## Exact dependency preflight

After inspection, run the local import/version preflight. It validates the checkpoint
again and then checks torch, torchvision, torchaudio, Accelerate, Transformers,
librosa, diffusers, flash_attn, einops, safetensors, and NumPy. It reports installed versions,
required versions, import failures, and mismatches without constructing the model.

```bat
python scripts\longcat-next\make-reference-fixtures.py ^
  --mode preflight ^
  --model-dir D:\LongCat-Next ^
  --runtime-profile blackwell-compatible ^
  --placement cuda
```

Do not run BF16 or F16 generation until this command finishes successfully. In
particular, do not treat a package being listed by pip as sufficient when its import
fails because a DLL, CUDA runtime, or compiled extension is unavailable.
For CUDA it also records the OS/platform, torch and torch-CUDA versions, GPU name,
compute capability, torch architecture list, sm_120 availability, BF16 support, a
real synchronized CUDA tensor operation, torchvision/torchaudio compatibility, and
a tiny causal BF16 FlashAttention forward with shape, finite-value, and synchronized
CUDA checks. On native Windows it additionally records the distribution identity,
module and install paths, exact Python version, wheel tags, unofficial community
provenance, and official-versus-executed version departure. It then dynamically
imports the pinned local config/model classes through Transformers' own remote-code
loader before any shard is loaded.

For a directly installed wheel, preflight reads its PEP 610 `direct_url.json`, decodes
the original URL, and parses the ABI-qualified original wheel filename. The normalized
installed distribution version (for example, `2.9.2`) is recorded separately because
it does not retain CUDA, PyTorch, C++ ABI, or Blackwell identity. If PEP 610 metadata
is unavailable, pass the original wheel path for identity validation only:

```bat
python scripts\longcat-next\make-reference-fixtures.py ^
  --mode preflight ^
  --model-dir D:\LongCat-Next ^
  --runtime-profile blackwell-compatible ^
  --placement cuda ^
  --flash-wheel-path D:\path\to\the-original-ABI-qualified-wheel.whl
```

The fallback never installs or opens the wheel payload. Without either PEP 610 origin
metadata or this explicit path, preflight fails with `wheel origin identity unavailable`.

## BF16 generation

The default automatic placement starts with advertised limits of 88 GiB GPU and
220 GiB CPU memory, repeat count 2, eight greedy tokens, and a 64 MiB aggregate
fixture-output ceiling. Accelerate may place or offload modules within those limits.
These defaults require local measurement.

```bat
mkdir D:\LongCat-Next-reference
mkdir D:\LongCat-Next-offload
python scripts\longcat-next\make-reference-fixtures.py ^
  --mode core ^
  --model-dir D:\LongCat-Next ^
  --precision bf16 ^
  --runtime-profile blackwell-compatible ^
  --placement auto ^
  --gpu-memory 88GiB ^
  --cpu-memory 220GiB ^
  --offload-dir D:\LongCat-Next-offload ^
  --repeat-count 2 ^
  --max-output-bytes 67108864 ^
  --output-dir D:\LongCat-Next-reference\bf16
```

## F16 generation

Run F16 separately so its source dtypes and reproducibility measurements remain
independent:

```bat
python scripts\longcat-next\make-reference-fixtures.py ^
  --mode core ^
  --model-dir D:\LongCat-Next ^
  --precision f16 ^
  --runtime-profile blackwell-compatible ^
  --placement auto ^
  --gpu-memory 88GiB ^
  --cpu-memory 220GiB ^
  --offload-dir D:\LongCat-Next-offload ^
  --repeat-count 2 ^
  --max-output-bytes 67108864 ^
  --output-dir D:\LongCat-Next-reference\f16
```

Use --placement cpu for CPU-only placement or --placement cuda for explicit device
0 placement. Omit --offload-dir only when no disk offload is desired. Memory values
accept GB, GiB, MB, or MiB strings. Increase the output ceiling only after reviewing
which derived arrays require it; the hard ceiling is 256 MiB. Use --hash-shards on a
core run to include all 15 shard hashes, at the cost of reading the full 150 GB first.

The pinned checkpoint is `model_type=longcat_next` and declares `BloomTokenizer`; it
is not a Mistral tokenizer. AutoTokenizer normally materializes the pinned
tokenizer.json as `BloomTokenizerFast` with `is_fast=true`. The harness records these
declared and executed identities separately, plus the slow-tokenizer class name when
available; the expected `Fast` suffix is not an identity mismatch and `use_fast=False`
is not forced. It passes `fix_mistral_regex=False` to Transformers 4.57.6 so its
large-local-tokenizer false-positive path cannot rewrite the pinned pre-tokenizer.
Metadata records both tokenizer file hashes, the source directory, backend
pre-tokenizer state hash, exact prompts, and input IDs; direct-forward and greedy
tokenization must match exactly.

Model loading uses the Transformers 4.57.6 `dtype` argument rather than deprecated
`torch_dtype`. Before capture, the requested BF16/F16 dtype must match both the
effective model dtype and base embedding weight dtype. Greedy generation uses a copy
of the model generation configuration with sampling disabled, sampling-only values
cleared, caching and return dictionaries enabled, and the CLI token limit recorded.
The model's original generation configuration is never mutated.

Transformers 4.57.6 can replace a copied `do_sample=False` value with LongCat's
model-specific `do_sample=True` default. The harness therefore also passes one shared
set of direct `generate()` keyword overrides to both prompts: `use_model_defaults=False`,
`do_sample=False`, null `temperature`/`top_p`/`top_k`, the CLI `max_new_tokens`,
`use_cache=True`, and `return_dict_in_generate=True`. Before every call it resolves
the policy through the model's GenerationMixin preparation path and requires the
effective policy to remain greedy. Metadata separates the copied base configuration,
direct call overrides, and resolved effective policy.

Any generation exception is fatal and writes no fixture. Its error identifies the
prompt, repeat index, requested policy, and known effective decoding mode. In
particular, a CUDA device-side assertion is never caught for retry in the same Python
process.

### Expected workstation messages

The tokenizer `fix_mistral_regex` warning, deprecated `torch_dtype` warning, and
ignored `temperature`/`top_p`/`top_k` warning would indicate a harness regression and
should no longer appear. This pinned remote-code/composite model may still emit the
configuration-time warning that FlashAttention 2 lacks a specified torch dtype even
though the harness passes Transformers 4.57.6's explicit `dtype` argument. Treat that
warning as a non-blocking false positive only when the later authoritative gate
confirms both `model.dtype` and the base embedding weight match the requested BF16 or
F16 dtype. Any gate mismatch remains fatal; do not restore deprecated `torch_dtype`
or suppress warnings globally. The audio autocast and diffusers
`LoRACompatibleLinear` FutureWarnings, visual/audio offset diagnostic prints, and
Accelerate messages about parameters on the meta device due to CPU offload are also
non-blocking for this text-core fixture run; do not edit official checkpoint code
merely to suppress them.

## Captured official anchors

For ordinary text-only inputs, hooks resolve these exact official module boundaries:

* model.model.embed_tokens output: base token embeddings;
* model.model.ngram_embeddings.post_projs[0..11] outputs: twelve raw projected
  n-gram outputs, plus separately recorded float32 analytical contributions that
  are not official captured intermediates;
* input to model.model.layers[0]: fused pre-trunk embedding;
* input to logical layer 0 input_layernorm[1]: physical block 0 output;
* logical layer 0 output: physical block 1 output;
* input to logical layer 1 input_layernorm[1]: physical block 2 output;
* logical layer 13 output: physical block 27 output;
* model.model.norm output: final normalized hidden state;
* LongcatNextForCausalLM output with logits_to_keep=1: final-position logits.

The physical mapping follows the pinned Transformers 4.57.6
LongcatFlashDecoderLayer.forward implementation: each logical layer executes two
physical attention/MLP sublayers. Hook resolution fails closed if this structure,
the 14 logical layers, or twelve post projections is absent.

Every direct forward constructs the exact dynamically loaded official
LongcatNextForCausalLMGenerationStatus from the pinned generation_config.json visual
and audio generation dictionaries, explicitly switches it to text mode, and passes
the status plus both GenerationConfig objects. Greedy generation requests the
official return-dictionary contract and validates its two-dimensional sequences;
the guarded four-item official tuple contract is also understood.

The bos_left_zero case masks only its leading zero padding. The literal_zero case
keeps token zero visible, and all other explicit cases use attended tokens unless a
future case declares otherwise. Position IDs follow the pinned Transformers 4.57.6
GenerationMixin preparation exactly: attention-mask cumulative sum minus one, with
masked positions set to one. Cache positions remain the ordinary sequential range.

The corpus combines explicit checked token-ID cases with two prompts rendered by the
local pinned official tokenizer. It records input, attention, and position arrays;
all captured activation families; selected logits; one complete 131,125-entry
final-position logit vector; top-10 IDs/values; argmax; and fixed greedy continuations.
Metadata records shapes, source and serialized dtypes, per-array hashes, software,
seeds, checkpoint facts, and hook anchors. BF16 arrays are serialized as float32
because NumPy has no portable BF16 storage; the original torch dtype remains explicit.
No model parameters are serialized.

The twelve ngram_projection_raw_XX arrays are direct official hook outputs. Arrays
named ngram_analytical_f32_* are explicitly derived float32 diagnostics, not official
intermediates. Their reconstruction and absolute/relative error are compared with
the directly captured fused_pre_trunk_embedding, which remains the parity authority.
This exposes BF16/F16 accumulation and rounding differences rather than relabeling a
float32 division as official execution.

## Expected local outputs

BF16 produces:

```text
longcat-next-core-bf16.npz
longcat-next-core-bf16.json
longcat-next-core-reproducibility.json
```

F16 uses the corresponding longcat-next-core-f16 names. The JSON layouts are defined
by tests/fixtures/longcat-next/core-fixture-schema.json and
core-reproducibility-schema.json. Local outputs must not be copied into the repository
until separately reviewed for size, privacy, and accidental weight content.

## Repeats, interruption, and tolerance freeze

Repeat count greater than one replays every official input. Token IDs, masks, shapes,
and greedy continuations must match exactly. Every numeric family reports byte
identity plus maximum absolute and relative differences. Nondeterminism is retained
in the report; it is not averaged or discarded. Both comparison tolerances remain
null, and the tool never selects them.

Output files are written only after all repeats complete, using temporary files and
atomic replacement. After interruption, keep the checkpoint and optional offload
cache, delete any .tmp files in the output directory, verify free disk space, rerun
inspection, and rerun the same generation command. Do not combine partial BF16 and
F16 directories.

Reviewers will use official-only repeated-run variation to propose and freeze BF16
and F16 tolerances before seeing any llama.cpp result. Stop the core implementation
stage if checkpoint inspection fails, required hooks cannot be resolved, official
repeats change exact outputs, numeric instability cannot be bounded before tolerance
selection, complete logits are not 131,125 wide, or derived fixtures cannot remain
small without exposing weight-like data.
