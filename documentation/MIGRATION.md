# DCM machine migration — local files to copy

Handoff for the LLM on the **target** Mac. This lists exactly what to pull from the
**source** Mac (paths below are on the source). The source is frozen: no training
process is running, all Models files are stable, and the repo is pushed. Corpus is
**not** in this list — the corpus DMG is already mounted on the target.

Source Mac: user `andrew`, home `/Users/andrew`.

---

## Copy these

### 1. Models  (9.4 GB, ~810 `.safetensors`) — REQUIRED
- **Source:** `/Users/andrew/Library/Application Support/DrewsChessMachine/Models/`
- **Destination:** `~/Library/Application Support/DrewsChessMachine/Models/`
- Contains every model: initial/build weights, per-run finals (`*-replay-latest.safetensors`),
  and the granular `*-replay-step*.safetensors` intermediates. This is the whole point of
  the migration — copy the entire directory.

### 2. Training-parameter preferences  (~2.8 KB) — OPTIONAL
- **Source:** `/Users/andrew/Library/Preferences/com.drewben.DrewsChessMachine.plist`
- **Destination:** `~/Library/Preferences/com.drewben.DrewsChessMachine.plist`
- The tuned `TrainingParameters` (stored in UserDefaults). Optional because the same values
  are committed to the repo at `documentation/parameters.json` and can be re-applied there.

### 3. Repo — PREFER GIT, don't copy the working dir
- **Source (only if not using git):** `/Users/andrew/Documents/drews-chess-machine/`
- **Preferred:** `git clone` the GitHub remote (or `git pull` if already cloned).
- The repo already contains all dashboard data/CSVs, `documentation/parameters.json`, and
  `documentation/dashboards/registry.json` (with pinned elapsed baselines). Nothing model-sized
  is in git.

---

## Do NOT copy
- **`~/Downloads/foo/*.dmg`** (the corpus) — already mounted on the target.
- **DerivedData / the built `.app`** — rebuild on the target in Xcode. The binary path is
  machine-specific (see warning 4).
- **`~/Library/Logs/DrewsChessMachine/`** — 100 MB–1 GB per file, disposable. The tracker no
  longer needs logs (every segment's elapsed baseline is pinned in `registry.json`).
- **`foobar`** in the repo root — stray untracked junk, ignore.

---

## Conflict / overwrite / merge warnings

1. **Models directory — MERGE, do not wipe.** If the target already has a `Models/` folder,
   add the missing files into it; don't replace it. Build/init and `*-replay-step*` names are
   timestamp+modelID unique, so they won't collide.
   **Overwrite risk = the rolling `*-replay-latest.safetensors` files** (one fixed name per run
   stem, overwritten on every save). Only a problem if the target already has a run with the
   **same stem** in a different state. Our stems are, e.g.: `20260629-mini2b-3MIV-resume3`,
   `20260702-Qeu8`, `20260702-Qeu8-resume2`, `20260701-nT8Y-resume3`, `20260629-mini1b-Coxw`,
   `20260628-v5_5block_7x7_lnout`, `20260704-Qeu8e3` (full list = `ls Models/*-replay-latest*`).
   If any of those already exist on the target, compare before overwriting.

2. **Params plist — full overwrite.** Copying `com.drewben.DrewsChessMachine.plist` REPLACES the
   target's DrewsChessMachine preferences wholesale. If the target has its own tuned params,
   back them up first. Harmless detail: the plist's `LastSessionPointer` points at source-Mac
   paths; the app clears a dangling pointer on first launch.

3. **Repo — don't copy over an existing checkout.** If the target already has the repo, `git pull`
   (don't overlay the working dir — it clobbers `.git` and any local changes).

4. **Hardcoded `/Users/andrew` + binary path.** The saved resume commands and the assistant's
   memory hardcode `/Users/andrew/...`. If the target's username differs, rewrite those paths.
   Also the built-binary path hardcodes THIS Mac's DerivedData hash
   (`DrewsChessMachine-duuojsefpqabteeapbtaobbiymfs`); the target's Release build will have a
   different hash — re-derive `BIN` from the target's own DerivedData after building.

5. **Corpus mount path.** Resume commands expect `/Volumes/20260624-192615-w3aA5b`. Confirm the
   target's already-mounted DMG is at exactly that path (it will be if the volume name matches
   the corpus id).

---

## CURRENT RUN & how to resume it (target Mac)

**We are currently on `nt8y`.** It was **stopped at 28.180 h of training time (cum_step 289883)**,
which is **3.12 h short of the 31.3 h parity target**. `mini2b` and `qeu8` are already DONE at
31.3 h; `coxw` has not started yet. Goal order: `nt8y` (finish) → `coxw`.

Latest nt8y checkpoint to warm-start from:
`20260701-nT8Y-resume3-replay-latest.safetensors`

### Step 1 — build the app, set BIN, confirm the corpus is mounted
```bash
# BIN is machine-specific — re-derive it from the TARGET's DerivedData after building:
BIN="$(ls -d ~/Library/Developer/Xcode/DerivedData/DrewsChessMachine-*/Build/Products/Release/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine | head -1)"
MODELS="$HOME/Library/Application Support/DrewsChessMachine/Models"
CORPUS=/Volumes/20260624-192615-w3aA5b          # the already-mounted corpus DMG
ls -d "$CORPUS" >/dev/null && echo "corpus OK" || echo "MOUNT THE CORPUS FIRST"
```

### Step 2 — resume nt8y (detached), running to the 31.3 h target
```bash
nohup "$BIN" --replay-corpus "$CORPUS" \
  --start-model "$MODELS/20260701-nT8Y-resume3-replay-latest.safetensors" \
  --out-model  "$MODELS/20260701-nT8Y-resume4-replay-latest.safetensors" \
  --epochs 5 --enumerate-checkpoints >/dev/null 2>&1 &
```

### Step 3 — register the resume segment so tracking stays correct (PIN the baseline)
In `documentation/dashboards/registry.json`, run key `nt8y`: set `out_model` to
`20260701-nT8Y-resume4-replay-latest.safetensors` and append a segment:
```json
{ "label": "resume4 (enumerated)", "cumstep_base": 289883,
  "elapsed_base_sec": 101447.6, "log": "<the new dcm_log_*.txt>", "date": "<yyyymmdd>" }
```
The `elapsed_base_sec` pin is **required** (nt8y's older logs are gone / unreliable — this keeps
its elapsed continuous instead of restarting low). Then `python3 tick.py nt8y && python3 master.py`.

### Step 4 — STOP nt8y at parity, then start coxw
Kill the process when nt8y's `max(elapsed_train_sec)/3600 >= 31.3` (≈ +3.12 h of training from now),
commit, then resume **`coxw`** the same way:
`--start-model 20260629-mini1b-Coxw-replay-latest.safetensors`
`--out-model 20260629-mini1b-Coxw-resume-replay-latest.safetensors`, registry `coxw` segment
`cumstep_base=55000`, `elapsed_base_sec=20093.5`; run coxw to 31.3 h (+25.7 h). Full loop logic is
in the assistant's memory `dcm-parity-goal`.
