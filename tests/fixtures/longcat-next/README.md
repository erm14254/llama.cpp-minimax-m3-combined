# LongCat-Next reference fixtures

This directory contains small, reviewable evidence fixtures. It must never contain
model weights, checkpoint shards, arbitrary pickle files, or generated output above
1 MiB per file.

`ngram-cases.json` is weight-free and ASCII JSON. For each input sequence it stores
official and independently recomputed copies of all 12 integer hash streams (orders 2, 3, and 4; four splits each), table moduli,
polynomial power residues, and lookup masks. It covers BOS/left-zero padding, literal
zero, EOS in every order-4 history position, token 131071, every ignored ID from
131072 through 131124, prompt-at-once versus incremental history, and two independent
histories.

Regenerate it with the pinned command in `docs/longcat-next/EVIDENCE_HARNESS.md`.
The generator accepts only immutable recorded revisions, verifies a Git checkout or
`.longcat-next-revision` snapshot marker, uses fixed seeds, and rejects output above
1 MiB. Review the resulting SHA-256 and update `manifest.json` only when an intentional
fixture schema or official source change is approved.

Weight-backed embedding, selected-layer, logits, and greedy fixtures are deliberately
absent. Their expected interfaces are reserved in the manifest. Before generating
or comparing them, reviewers must select separate BF16 and F16 tolerances without
seeing C++ results. Pending `null` tolerances are not permission to compare loosely.

Fields named `official_hashes` come from AST-isolated methods executed directly from the
pinned official source. Fields named `independent_hashes` come from the standalone
implementation. Generation fails unless they match exactly for every case.

The two core-*.json schema files define local-only weight-backed metadata and
official reproducibility reports. They contain no generated activations. Actual
longcat-next-core-*.npz/json files remain outside the repository pending review.

The manifest preserves the official-pinned runtime versions separately from the
Blackwell-compatible execution policy. Local core metadata must record exact installed
versions and every departure; it must never rewrite the official provenance.

`stage1-tolerances.json` is the immutable cross-implementation policy selected
before the first local C++ parity run. Unlike the Stage-0 manifest's still-pending
historical fields, the Stage-1 comparator applies these checked values and does not
tune them after observing candidate output.
