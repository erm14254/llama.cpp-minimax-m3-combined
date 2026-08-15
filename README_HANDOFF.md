# LongCat Claude Code Handoff Package

Copy/extract these files into:

`D:\llama.cpp-longcat-pre-gate4`

Then:

1. Run `powershell -ExecutionPolicy Bypass -File .\BOOTSTRAP_LONGCAT_HANDOFF.ps1`.
2. Open the repo in Claude Code Desktop.
3. Tell Claude Code: **Read CLAUDE.md, HANDOFF_MEMORANDUM_2026-08-15.md, and NEXT_ACTION.md; inspect the local state; continue autonomously and minimize manual copy/paste.**
4. When ready to preserve the diagnostic tree remotely, run `powershell -ExecutionPolicy Bypass -File .\PUSH_HANDOFF.ps1`.

Package files:

- `HANDOFF_MEMORANDUM_2026-08-15.md` — authoritative engineering state/provenance.
- `CLAUDE.md` — operating instructions/guardrails for Claude Code Desktop.
- `NEXT_ACTION.md` — immediate next diagnostic.
- `BOOTSTRAP_LONGCAT_HANDOFF.ps1` — checks hashes, paths, Git state and local scripts.
- `PUSH_HANDOFF.ps1` — safe staging/commit/push helper.
- `handoff_manifest.json` — machine-readable key state.
- `SHA256SUMS.txt` — package hashes.
