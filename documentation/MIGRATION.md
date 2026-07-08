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

## After copying (target Mac) — resume the parity goal
State is in the assistant's memory (`dcm-parity-goal`). In short: `mini2b` + `qeu8` are done at
31.3 h; **`nt8y`** is paused at 28.18 h (cum 289883, +3.12 h) — resume from
`20260701-nT8Y-resume3-replay-latest.safetensors` (new out-stem `-resume4`, registry segment
`cumstep_base=289883`, pin `elapsed_base_sec ≈ 101448`), run to 31.3 h; then **`coxw`** (+25.7 h).
Verify the corpus is mounted, build the app, then continue.
