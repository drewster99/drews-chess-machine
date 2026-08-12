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
| `checkpoint_inventory.json` | SHA-256 + safetensors metadata for all 650 surviving checkpoints. **`path` is ephemeral** — a snapshot of where each file sat on 2026-08-12, after the bundle's 335 ambiguous root names were removed. The durable identity is `model_id` + `training_step` + `sha256`; once the bundle is deleted the paths mean nothing, the hashes still do. |
| `v5-run-parameters.json` | the hyperparameters all five continuation runs **actually used**, byte-identical across them |
| `v5-bundle-staging-parameters.json` | a pre-finalisation snapshot — **never used for training**, see below |

These 761 probes were imported into `../v5.csv` (`replay.py import-probes`), which is
the **source of truth**. They are retained because each probe JSON holds more than the
four fields the tracker keeps — `argmaxCorrect`/`top5Correct` (top-1/top-5 accuracy),
`avgProb`, `avgRank`, and a 13-category `themes` breakdown (fork, pin, skewer, mateIn1/2,
endgame, sacrifice, …). None of that is recoverable from the CSV, and most of the
checkpoints they describe no longer exist, so the probes cannot be re-run.

`checkpoint_inventory.json` is what made the merge into `Models/` verifiable: it maps
every file to its `model_id` + `training_step`, which is the only trustworthy identity —
four distinct files have been named `v5-cont-replay-step1000.safetensors`.

## Where segment 0–2's hyperparameters live

The two files here cover **segments 3–7 only**. Segments 0–2 (the M5-VM era) have no
parameters file — their session logs died with the VM and safetensors headers carry no
hyperparameters. Their settings survive solely as prose in
`documentation/v5-layernorm-output.md`, and are now recorded structurally in
`registry.json` under `segments[i].hparams`. See `documentation/v5-lineage.md` §2a.

## The two parameter files

`v5-continue-bundle/` existed in two places: the working copy in `~/Downloads` and a
staging copy at the repo root. They were byte-identical except for `parameters.json`,
and the difference is exactly two values:

| | staging (repo root, Jul 2 17:41) | shipped (`~/Downloads`, Jul 2 18:36) |
|---|---|---|
| `momentum_coeff` | 0.90 | **0.93** |
| `weight_decay` | 1e-4 | **2.5e-4** |

The staging copy was captured 55 minutes into assembling the bundle, carrying the app's
then-current settings; the values were corrected before it shipped. **No training run
ever used 0.90 / 1e-4** — all five continuation segments (3–7) ran on 0.93 / 2.5e-4, as
every `[REPLAY-HPARAMS]` line in their session logs confirms.

**The repo-root copy was deleted on 2026-08-12** — this file is all that survives of it,
and is the reason it could be deleted. 57 of its 58 files were byte-identical to the
`~/Downloads` bundle (corpus, checkpoint, README, all verified by hash); `parameters.json`
was the only one that was not, and its exact bytes are here.
Do not mistake `wd 1e-4` here for segment 0's `wd1e-4` label — that segment ran on
2026-06-27, five days before this file was written.

## `bundle-provenance/`

Everything that existed **only** in `~/Downloads/v5-continue-bundle/` and nowhere else —
the retired monitor's tooling, the forensic record of the fabricated-probe incident, the
contemporaneous narrative docs, and each run's sampled console output. That subdirectory
is what makes the bundle safe to delete. See its own README.

See `documentation/v5-lineage.md`.
