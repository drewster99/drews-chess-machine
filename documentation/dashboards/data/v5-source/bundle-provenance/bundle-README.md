# Continue the v5 net — run bundle

This folder has everything needed to **continue training the v5 network** (8.45M-param,
5-block 7×7 lc0-style net) on another **Apple Silicon Mac**. It will run for roughly
**7 days** (see "How long" below), then you bring the results back.

## Contents
- `DrewsChessMachine.app` — prebuilt Release binary (try this first).
- `v5-checkpoint.safetensors` — the v5 weights to continue from (model_id `20260629-1-Uf4p`, ~99.9k steps, final pElo ~1583 / peak ~1604). Architecture is embedded, so the app rebuilds the exact net automatically.
- `parameters.json` — the training hyperparameters (lr 0.01, batch 4096, wd 2.5e-4, momentum 0.93, gradClip 30, replayRatio 0.48, pLabelSmooth 0.1, vLabelSmooth 0.013, …). Same knobs the run used here.
- `corpus/` — the training corpus: 46 `.dcmgames` shards + `corpus.json` (Lichess May 2026 standard-rated, ~20.9M games, ~2.9 GB, corpus id `20260624-192615-w3aA5b`).

## Requirements
- Apple Silicon Mac (M-series). The net runs on MetalPerformanceShadersGraph — Intel/other won't work.
- ~5 GB free disk for the run's output checkpoints.

## Run it (exact)
Open Terminal, `cd` into this folder, then:

```bash
cd ~/Downloads/v5-continue-bundle    # wherever you put this folder

# First launch of the prebuilt app may be blocked by Gatekeeper.
# If so: right-click DrewsChessMachine.app in Finder -> Open (once), or:
xattr -dr com.apple.quarantine ./DrewsChessMachine.app

./DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine \
  --replay-corpus "./corpus" \
  --start-model  "./v5-checkpoint.safetensors" \
  --out-model    "./v5-cont-replay-latest.safetensors" \
  --parameters   "./parameters.json" \
  --epochs 5
```

To keep it running unattended, prefix with `nohup … &` or use `caffeinate -i`.

### If the prebuilt app won't launch — build from source instead
```bash
git clone https://github.com/drewster99/drews-chess-machine.git
# open drews-chess-machine/DrewsChessMachine/DrewsChessMachine.xcodeproj in Xcode,
# select the DrewsChessMachine scheme, set Release, and Build.
# Binary lands at: ~/Library/Developer/Xcode/DerivedData/DrewsChessMachine-*/Build/Products/Release/DrewsChessMachine.app
```
Then run the same command with that binary path.

## What it does / how long
- This is a **warm-start**: it loads v5's weights but **restarts the corpus from the beginning** (fresh epoch 0; BN running stats + optimizer velocity reset). `--epochs 5` = up to 5 full passes over the ~20.9M-game corpus.
- One full epoch ≈ **162k steps**. On v5-class hardware (~1 s/step) that's ~1.5–2 days/epoch, so **5 epochs ≈ ~7–9 days**. Adjust `--epochs` if your machine is faster/slower.
- Output: it overwrites `v5-cont-replay-latest.safetensors` every 1000 steps (rolling latest). Per-step telemetry goes to a session log at `~/Library/Logs/DrewsChessMachine/dcm_log_<launch-timestamp>.txt`.
- **Stop anytime with Ctrl-C** — it does a clean final save before exiting. Safe to stop around the 7-day mark.

## Bring back (to the source-of-truth machine)
When done, copy back:
1. **`v5-cont-replay-latest.safetensors`** (the final trained weights). If you want the full pElo curve, also grab whatever intermediate snapshots you kept.
2. The **session log** `~/Library/Logs/DrewsChessMachine/dcm_log_<launch-timestamp>.txt` (needed to reconstruct the time axis + loss metrics on the tracking machine).

That's it — the tracking/registry/dashboard on the home machine will register a new v5 resume segment from those two artifacts.

## Advanced: exact mid-corpus continuation (optional)
v5 stopped ~62% through epoch 0 (game 12,933,495 of 20,935,171). To continue that *exact* stream position instead of a fresh pass, use `--resume-exact` **instead of** `--epochs` — but it requires a binary built from git commit `4d60ca2` (it refuses on encoder/build mismatch). Simpler approximation without the build constraint: add `--start-game-index 12933495` to the command above (starts near where v5 left off; approximate cold-refill). For a plain "keep improving the weights," the default warm-start above is fine.
