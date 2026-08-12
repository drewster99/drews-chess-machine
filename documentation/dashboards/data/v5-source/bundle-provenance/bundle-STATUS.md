# This bundle is RETIRED — its contents live elsewhere now

_2026-08-12. `README.md` describes how this bundle was originally used (July 2); it is
no longer accurate about the layout. Read this first._

Everything here has been merged into normal storage and verified. The bundle is kept
only until Time Machine has captured the merged `Models/` on the master-record machine,
then it can be deleted.

## Where it all went

| was here | now lives |
|---|---|
| 650 checkpoints | `~/Library/Application Support/DrewsChessMachine/Models/` on the master machine — 615 files under per-segment names (`20260702-v5cont`, `20260713-v5cont-resume`, `20260728-v5cont-resume2`, `20260802-v5cont-resume3`, `20260804-v5cont-resume4`) |
| `archive/logs/*.gz` | the raw originals, in that machine's `~/Library/Logs/DrewsChessMachine/` |
| `monitor/new_ckpts*.jsonl`, `checkpoint_inventory.json` | the repo — `documentation/dashboards/data/v5-source/` |
| `parameters.json` | the repo — `data/v5-source/v5-run-parameters.json` |
| metrics | the repo — `documentation/dashboards/data/v5.csv`, 856 rows |
| `README.md`, `archive/MANIFEST.md`, `monitor/HANDOFF.md`, `quarantine/README.md` | folded into `documentation/v5-lineage.md` |
| `corpus/` | redundant — `Corpora/` on the master machine is canonical, and this was a copy of it |

Verified by content hash: **all 650 checkpoints exist on the master machine** (616 distinct
contents = 615 merged + the entry point that was already there).

## What changed in this folder

The 335 loose `v5-cont-replay-step*.safetensors` at the root were **deleted on
2026-08-12**. They were hardlinks — each shared an inode with a file in `run2/`…`run5/`,
so removing them freed no space and lost no data. It removed the ambiguity: those
filenames were the single most misleading thing here, because all five runs numbered
their checkpoints from 1 and overwrote each other, so a name like `step1000` has meant
four different files over this bundle's life.

Every checkpoint now has exactly **one** name, and that name says which run it came from:

```
run1/   268   run2/  231   run3/   58   run4/   44   run5/    2
preserved-best/  40        quarantine/   5
v5-checkpoint.safetensors            the lineage entry point (step 39,419)
v5-cont-replay-latest.safetensors    run 5's final state (step 2,000)
```

**Identify a checkpoint by its safetensors `model_id` + `training_step`, never its
filename.** `documentation/dashboards/ckpt_inventory.py` reads that;
`monitor/checkpoint_inventory.json` is the manifest for all 650.

## The one trap still in here

`quarantine/` holds run 4's final save at step 49,374. Its metadata claims a completed
epoch (`replay_epoch=1, replay_next_game_index=0`) when only 6,360,368 of ~20,935,171
games were actually fed — macOS revoked read access to corpus shards 14–45 mid-run.
**Never `--resume-exact` from it.** On the master machine it carries the marker
`…-resume3-replay-step49374-DO-NOT-RESUME.safetensors`.

Backup while this folder still exists: `~/v5-consolidation-backup-20260811/`.
