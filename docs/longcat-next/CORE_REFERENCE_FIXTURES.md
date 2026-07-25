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
config/tokenizer files, 15 referenced safetensors shards, and all custom Python code.
Inspection enforces 13,450 names, 150,825,367,872 tensor payload bytes, 15 shards,
150,827,115,056 aggregate shard-file bytes, the three vocabulary extents, zero MTP
names, no missing or unreferenced model shards, and pinned custom-code/config/tokenizer
SHA-256 identities.

Use a dedicated 64-bit Python 3.10 environment. Transformers must be exactly 4.57.6.
Install numpy, safetensors, Accelerate, and the official model's other Python
requirements. Install a PyTorch build appropriate for the local NVIDIA driver and
CUDA environment using the current official PyTorch instructions; this document
deliberately does not invent or pin an unverified torch build. Verify torch and CUDA
before loading 150 GB of weights.

### Windows cmd.exe environment setup

Run these commands in an x64 Native Tools Command Prompt. They are cmd.exe commands,
not PowerShell syntax.

```bat
py -3.10 -m venv D:\LongCat-Next-fixture-env
call D:\LongCat-Next-fixture-env\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install transformers==4.57.6 accelerate safetensors numpy
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import transformers; assert transformers.__version__ == '4.57.6'"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1
```

Install PyTorch before the two verification commands. Keep the three offline
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

## Captured official anchors

For ordinary text-only inputs, hooks resolve these exact official module boundaries:

* model.model.embed_tokens output: base token embeddings;
* model.model.ngram_embeddings.post_projs[0..11] outputs: twelve raw projected
  n-gram outputs, plus separately recorded effective contributions after the
  official conditional division by 13;
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

The corpus combines explicit checked token-ID cases with two prompts rendered by the
local pinned official tokenizer. It records input, attention, and position arrays;
all captured activation families; selected logits; one complete 131,125-entry
final-position logit vector; top-10 IDs/values; argmax; and fixed greedy continuations.
Metadata records shapes, source and serialized dtypes, per-array hashes, software,
seeds, checkpoint facts, and hook anchors. BF16 arrays are serialized as float32
because NumPy has no portable BF16 storage; the original torch dtype remains explicit.
No model parameters are serialized.

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
