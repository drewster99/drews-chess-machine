# v5 source artifacts

Kept because they have no other home once `v5-continue-bundle/` is retired, and
because they are the *pre-image* of numbers that are otherwise only derived.

| file | what it is |
|---|---|
| `new_ckpts.jsonl` | segment 3 (cont-run1) — 269 probe results |
| `new_ckpts_run2.jsonl` | segment 4 (cont-run2) — 337 |
| `new_ckpts_run3.jsonl` | segment 5 (cont-run3) — 107 |
| `new_ckpts_run4.jsonl` | segment 6 (cont-run4) — 46 |
| `new_ckpts_run5.jsonl` | segment 7 (cont-run5) — 2 |
| `checkpoint_inventory.json` | SHA-256 + safetensors metadata for all 650 surviving checkpoints |
| `v5-run-parameters.json` | the hyperparameters all five continuation runs used, byte-identical across them |

These 761 probes were imported into `../v5.csv` (`replay.py import-probes`), which is
the **source of truth**. They are retained because each probe JSON holds more than the
four fields the tracker keeps — `argmaxCorrect`/`top5Correct` (top-1/top-5 accuracy),
`avgProb`, `avgRank`, and a 13-category `themes` breakdown (fork, pin, skewer, mateIn1/2,
endgame, sacrifice, …). None of that is recoverable from the CSV, and most of the
checkpoints they describe no longer exist, so the probes cannot be re-run.

`checkpoint_inventory.json` is what made the merge into `Models/` verifiable: it maps
every file to its `model_id` + `training_step`, which is the only trustworthy identity —
four distinct files have been named `v5-cont-replay-step1000.safetensors`.

See `documentation/v5-lineage.md`.
