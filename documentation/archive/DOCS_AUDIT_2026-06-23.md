# Docs audit — 2026-06-23

> **Resolution (2026-06-23).** All four tiers below have since been applied.
> Tier 1 (CLAUDE.md / ROADMAP / CHANGELOG) landed first; Tiers 2–4 plus the
> CLAUDE.md sampling-section fix landed in a follow-up pass — including the
> `replay_buffer_file_format.md` v7 rewrite with a full v1→v7 version-history
> table, and folding OVERNIGHT_INVESTIGATION's lr-stability-threshold finding
> into ARCH_EXPERIMENTS. This document is retained as the audit record.

Read-only audit of all ~43 substantive markdown docs (root + `documentation/` +
`scripts/` + `experiments/` notes; the 491 `experiments/*/proposal.md` autotrain
one-liners excluded). Each doc's claims were cross-checked against the current
working tree (`safetensors-storage` HEAD) and git history. Verdicts: ACCURATE /
DRIFTED / SUPERSEDED / STALE.

Dominant theme: **the docs lag the code by ~1–2 weeks.** The single biggest cause
is the **self-play corpus feature (shipped ~2026-06-20)**, which is still written
up as unbuilt in ROADMAP, absent from CHANGELOG, and "TABLED" in its own plan.

---

## Tier 1 — Authoritative docs with factual drift (fix first)

### CLAUDE.md — DRIFTED (highest stakes; it's the source-of-truth instructions)
- Reference-doc paths wrong: `chess-engine-design.md`, `sampling-parameters.md`,
  `mpsgraph-primitives.md` cited as repo-root but live under `documentation/`.
- `arenaPromoteThreshold` default is **0.53**, not 0.55 (`TrainingParameters.swift:555`).
- `policyEntropyAlarmThreshold` is **1.0 in `TrainingAlarmController.swift:43`**, not
  "5.0 in ContentView.swift".
- `[STATS]` cadence is **per-step (first 500) then every 60 s**, not "15-minute"
  (`SessionController+Training.swift:1746`).
- `absoluteMaxSelfPlayWorkers` lives on **`UpperContentView` and is 8192**, not
  `ContentView` (`UpperContentView.swift:454`).
- `BatchedSelfPlayDriver.stopAll`/`expectedSlotCount` deadlock caveat is **obsolete** —
  the file's own comments now say "No `expectedSlotCount` coordination".
- Arena driver is **`TickTournamentDriver`**, not `TournamentDriver`.
- "four currently-`liveTunable` params" is now **~50 of 60**; the named param is
  `SelfPlayConcurrency`, not `selfPlayWorkers`.
- Verified-correct (do NOT touch): `v4_5block_7x7` default, basic30/30-plane, 4864=76×64
  policy, 3-logit W/D/L value head, `--create-parameters-file`, the @TrainingParameter
  checklist touchpoints.

### ROADMAP.md — DRIFTED
- **MAJOR:** first "Future improvements (validated open)" item describes the self-play
  corpus as "(2026-06-20, **not yet built**)" — it shipped that day (commits
  `f0c0012`, `0ad27b3`, `c4243b5`, `461c95c`, `049e94f`, `5a93c54`; code in
  `Persistence/GameRecord.swift`, `GameCorpus.swift`, `CorpusRecorder.swift`,
  `CorpusReplayFeeder.swift`, `CLI/CorpusReplayRunner.swift`, `CLI/PGNImporter.swift`).
  Move to Completed.
- CLI-resume open issue #2 ("self-play recording not wired into CLI train path") is
  **false** — wired at `SessionController+Training.swift:1040`.
- `BatchFeedsInput` item "still open" — **shipped 2026-05-11** (`49878fa`,
  `ChessTrainer.swift:5659`).
- "Human-vs-model play" item "still open" — **shipped 2026-05-14** (`15613c9`,
  `PlayController.swift`).
- "Compiled `MPSGraphExecutable`" item partially stale — batched inference migrated
  2026-06-02 (`3f02ac4`, `ChessNetwork.swift:423`).
- Safetensors "Remaining:" list (uci/playchess/non-default training/precision) all
  shipped; CHANGELOG already notes this, ROADMAP is the laggard.

### CHANGELOG.md — STALE (entries present are hash/date-accurate; tail missing)
- Newest entry is 2026-06-17 fp16 (`b25f37e`); **19 commits behind HEAD.** Missing:
  the entire 2026-06-20 corpus feature (7 commits), the 2026-06-19 feature-skip axis
  (`449f625`), the 2026-06-23 commits (`52104f5`, `68f66d7`, `dd879b4`), and the
  2026-06-17/18 fp16 follow-ups. Append entries; ordering/timestamp discipline intact.

---

## Tier 2 — Reference docs describing superseded reality as current (misleading)

### documentation/replay_buffer_file_format.md — DRIFTED (material rewrite)
- Says "v4 … every current build produces"; code is **v7** (`ReplayBuffer.swift:2697`).
  Doc lists 4 body arrays, v7 has **9** (adds plyIndices/gameLengths/samplingTaus/
  stateHashes/workerGameIds at v5, materialCounts at v6). Doc documents the
  **`vBaselines` column that v7 deleted**. Header `pad` slot is now a live `encodingTag`.

### documentation/dcmmodel_file_format.md — STALE (presents legacy format as current)
- Model storage went **safetensors-native 2026-06-05**; `.dcmmodel` is now read-only
  legacy (`CheckpointManager.saveModel` → `SafetensorsModelIO.encode`). Doc gives no
  legacy framing. archHash described as 5-constant FNV-1a; code mixes **7** (adds
  valueHeadClasses + arch version), and archHash is no longer the model identity.
  Add LEGACY banner + correct the hash.

### documentation/dcm_architecture_v2.md — SUPERSEDED (filename invites misreading)
- It's an April-2026 plan + 2026-04-20 as-built snapshot, not a current ref. Describes
  **20 planes** (now 30/runtime-selectable), an **8×3×3 ~2.4M hardcoded tower** (now
  runtime 5×7×7 8.45M `v4_5block_7x7`), and a **scalar-tanh value head as future work**
  (WDL shipped 2026-05-12). Banner-mark historical or move to `documentation/archive/`.

### documentation/sampling-parameters.md — DRIFTED
- Preset table wrong: `.selfPlay`/`.arena` startTau listed 1.0/0.7, code is **2.0/2.0**
  (`MPSChessPlayer.swift:122`). Says policy is **4096**; it's 4864. Says ply is
  per-player; code decays per game-total ply. Missing the added Dirichlet root noise.
  ModelID mint/inherit section is accurate — leave it.

### SELFPLAY_CORPUS_PLAN.md — DRIFTED (status + body)
- Still "TABLED — deferred" but shipped 2026-06-20. **And** its "per-ply behavior-policy
  probability + value scalar + ModelID captured in v1 now" decision **did NOT ship** —
  real `GameRecord` is move-list-only (`startFEN`, `moves`, `outcome`, `terminationReason`;
  `Persistence/GameRecord.swift:36`). Reconcile body so readers don't think off-policy
  data is sitting in existing corpus files.

---

## Tier 3 — Status-line drift (feature shipped / tests run; doc says otherwise)

| Doc | Verdict | Fix |
|---|---|---|
| RUNTIME_ARCHITECTURE_CONFIG_PLAN.md | DRIFTED | "Phases 5–8 remaining" understates reality; nearly all shipped. Only headless `--architecture-preset`/`--architecture-file` CLI flags genuinely open. |
| HYBRID_FP32_MASTER_WEIGHTS_PLAN.md | DRIFTED | "NOT runtime-tested" stale — canonical bf16 trainer path, in production for weeks. |
| FULL10PLY200_PLAN.md | DRIFTED | "test run deferred" — tests ran + passed 2026-06-23. Phases 1–4 shipped. |
| REPLAY_HISTORY_RECONSTRUCTION_PLAN.md | DRIFTED | "tests written but not yet run" — ran + passed 2026-06-23. |
| FULL10PLY10REPS210_PLAN.md | ACCURATE | Shipped; only the §7 validation checkboxes are still unticked. |
| PREACTIVATION_SE_REZERO_PLAN.md | ACCURATE | Twice-stale "12-block" anchor (default now 5-block); archHash mechanics superseded by runtime-config. |
| DRAW_WATCH_PLAN.md | ACCURATE | Understates — termination-on-flag + tunable threshold shipped (v2/v3); buckets are 40-ply. |
| LICHESS_AUTO_COMPARE_PLAN.md | ACCURATE | "runtime validation pending" stale; manual Compare now nearest-set routing, no mismatch alert. |
| BUILD_WARNINGS_TODO.md | DRIFTED | Items 1/3/4 already fixed, not checked off; item 2 + test scaffolding still open. |
| scripts/README.md | DRIFTED | Documents 4 of 8 scripts; missing analyze_session_log / curate_lichess_probes / model_lineage_report / session_failure_analysis. |

---

## Tier 4 — Archive candidates (completed / abandoned / spent)

| Doc | Why |
|---|---|
| OVERNIGHT_INVESTIGATION.md | Falsely self-labeled "uncommitted." Track 1 solved + folded to CHANGELOG; Track 2 dups ARCH_EXPERIMENTS Exp 4. **Fold its unique iter-7/8 "lr above stability threshold for 3×3 towers" finding into ARCH_EXPERIMENTS first**, then archive. |
| ML_REVIEW_NOTES.md | 2026-05-05 snapshot; key findings now wrong — optimizer is SGD+momentum (not "plain SGD"), policy CE is on raw (not masked) logits, all line nums/defaults drifted. |
| COMMENT_AUDIT.md | Completed one-time sweep (acted on in its own commit); line refs no longer resolve. |
| ROADMAP_NOTES.md | Superseded by ROADMAP.md; itself stale (says replay buffer v6, now v7). |
| HISTORY_PLAN.md | Already self-declares ABANDONED/SUPERSEDED (22-plane/N-bank design, zero code footprint). Move out of active set. |
| CAPTURE_MOVE_MASK.md | Status (unimplemented) still true, but plan body stale: files moved to `Training/`, line nums off ~2000, its "bump to v7" scheme is dead (v7 already exists). Archive or re-anchor. ROADMAP Part D already lists it as scratch to retire. |
| experiments/PENDING_BATCH_STATS_SPEC.md | Shipped (`c1ec893`). Rename/archive out of "PENDING_". |
| experiments/PENDING_NEXT_ON_REJECT.md | Spent one-shot overlay; deliverables shipped. Archive or delete. |
| experiments/walkback-*.md (8 files) | Spent 2026-06-02 sweep; conclusions distilled into experiments/NOTES.md. Archive as a group. |

---

## Accurate — keep as-is (no action)

ARCHITECTURE_EXPANSION_PLAN.md · ARCH_EXPERIMENTS.md (2 dangling "results pending":
Exp 1§8 / Exp 6 / Exp 7) · ARCH_EXPERIMENT_ANALYSIS_PROMPT.md ·
STRATIFIED_REPLAY_SAMPLING_PLAN.md · TRAINING_DYNAMICS_PLAN.md · wdl-value-head.md ·
wdl-implementation-log.md · LICHESS_WIDE_PROBE_PLAN.md · CHART_X_AXIS_TOGGLE_PLAN.md
(accurate open TODO) · SESSION_PICKER_PLAN.md · documentation/chess-engine-design.md ·
documentation/mpsgraph-primitives.md · documentation/macos27-beta1-mpsgraph-findings.md
(cleanest doc audited) · experiments/NOTES.md

Several of these (ARCHITECTURE_EXPANSION, LICHESS_WIDE_PROBE, SESSION_PICKER) are
shipped features whose plans could optionally move to a Completed section.
