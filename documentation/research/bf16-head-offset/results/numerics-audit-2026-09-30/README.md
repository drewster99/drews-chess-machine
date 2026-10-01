# Numerics audit, 2026-09-30

`--analyze-numerics` (head numerics plan, Phase 0 tool) run on 10 checkpoints with the Release binary built from the source of commit `22c4783` (build 2263; code identical to `e37ea66`), macOS 27.2.

- Positions: 4,971 = start position + 874 from corpus `20260624-192615-w3aA5b` shard 45 + 4,096 from 63 Lichess bot games.
- Each checkpoint is run as fp32, bf16 and fp16 and compared with fp32. Weights are read **as stored** (`CheckpointManager.loadModelFileAsStored`): no value-head recentering on load, so Ejp0's stored offset is visible and only the fp32 head tails (plan Phase 1a) are exercised.
- Files: one `numerics_audit_*.json` per checkpoint; `audit_summary.txt` is the human-readable summary the tool printed.

Command:

```sh
"$BIN" --analyze-numerics <folder of the 10 checkpoints> \
  --numerics-corpus ~/Library/Application\ Support/DrewsChessMachine/Corpora/20260624-192615-w3aA5b/shard-00045.dcmgames \
  --numerics-out <out dir>
```

Checkpoints: Ejp0 @681k (`20260702-Qeu8-resume3-replay-step681000`), mUF5 (fp32 control, `20260627-v3_8block_3x3-step28797-FINAL-frozen`), and the SE-experiment finals (`20260929-test_SE_*` at steps 33014 / 33012 / 32036 / 7282 / 7289 / 7019 / 5030 / 5004).
