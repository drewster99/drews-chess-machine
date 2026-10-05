# Hyperparameter recording plan: a checkpoint states exactly how it was trained

Status (2026-10-05): **PLAN ONLY.** Nothing here is implemented.
- Independent review: **concurred on 2026-10-05, after five passes.** Every review item (A1–A20, B1–B9, C1–C6, D1–D7; N1–N6, NB1–NB7; P3-1, P3-2, NB-a–NB-c; P4-1 and nits; pass-5 nits) is listed, with what was done about it, in **Review reconciliation** at the end.
- Every `file:line` was checked against `main` at `a7d3b9ca`.
- Paths are relative to `DrewsChessMachine/DrewsChessMachine/` unless they start with `DrewsChessMachine/` (project folder), `DrewsChessMachineTests/` (= `DrewsChessMachine/DrewsChessMachineTests/`), `documentation/` or `scripts/`.
- Real-file evidence comes from header-only reads of the files under `~/Library/Application Support/DrewsChessMachine/`.

**The question.** Does every training hyperparameter that affects a run reach the model files the trainer saves, so that one checkpoint alone says how its weights were trained?

**The answer today.** No.
- A file written since lineage (format v7+, `dcm_lineage`) records almost every knob, but only for the segment that wrote it.
- A branch or derive drops the history of the weights it starts from.
- Some recorded values are stale, or contradict the file's other keys.
- Several knobs are never recorded: train-vs-UCI move selection and opponents, self-play Dirichlet noise, budgets, and the toolchain.
- 4,311 of the 4,334 model files on disk predate lineage and record almost nothing.

Rules this plan follows (CLAUDE.md files and the owner's standing rules):
- one source of truth;
- one code path shared by GUI, corpus replay and train-vs-UCI;
- no silent defaults and no fallbacks;
- no `try?` and no force unwraps;
- no migration code;
- the full parameter checklist for any new parameter (this plan adds none);
- tests are never modified or deleted without the owner's approval (each needed edit is listed under Owner decisions);
- bug-fix discipline: the regression test is written first and shown to fail, then the fix makes it pass unmodified.

Older files must keep decoding. A value a file did not record is read back as *unrecorded*, never filled in.

---

## Summary

| # | Gap | Severity | Decision | Phase |
|---|---|---|---|---|
| 1 | Earlier segments of a run lose their configuration: `segments[]` has no parameters, argv, policy tail or corpus | Critical | Fix | P4 |
| 1b | A branch or derive drops the parent run's history entirely (new run, `segments: []`) | Critical | Fix | P4 |
| 2 | 4,311 pre-lineage files (format ≤ 6) carry no parameter snapshot | High (historical) | Optional log-reconstruction report; never written into model files | P6 (O-8) |
| 3 | GUI: batch size, buffer capacity and minimum prefill are captured at run start but recorded at their *edited* value | High | Fix | P3 |
| 4 | GUI: live edits during a segment leave no trace in the file | High | Fix | P4 |
| 5 | CLI exact resume: the snapshot records the configured LR/momentum schedule, not the adopted checkpoint schedule. A GUI save, and an arena promotion record, can carry a schedule that did not train the weights. | Medium–High | Fix: one rule for every trainer-state save; arena start captures its schedule; the backstop ships with the GUI rule | P2 (CLI), P4 (GUI, arena, backstop) |
| 6 | `git_dirty` is `true` for every build; uncommitted code has no identity | High | Fix | P1 (flag + hash constant), P4 (record) |
| 7 | Mixed corpora: only the first corpus's ID and path are recorded | Medium | Fix (provenance only; no new refusal) | P4 |
| 8 | Policy-tail precision is not in the lineage record, champion files or segment summaries | Medium | Fix | P4 |
| 9 | The snapshot's `random_seed*` settings contradict the run's actual seed; `--seed` is recorded as `configured` | Low–Medium | Fix | P4 |
| 10 | Keys that do nothing on a path look meaningful | Low (replay); Low–Medium (vsuci) | Document now; code change deferred (O-12) | P5 |
| 11 | Run budgets (resolved step/time/epoch limits) and train-vs-UCI opponent identity are not recorded | Low (budget) / Medium (opponents) | Fix | P4 |
| 12 | The effective LR and momentum at save are not stored; the app computes the fed LR in two places (`buildFeeds` and the readout) | Low | Fix as a derived, never-read-back field from one shared function (O-16); with O-18, `buildFeeds` uses the same function | P4 |
| 13 | A `--resume-exact` that changes a training parameter is labelled `EXACT` with no per-key trace (CLI); a GUI resume replaces three saved values with no gap | Medium | Log now; gap semantics are O-4 | P2 |
| B1 | Train-vs-UCI game generation (argmax move selection, ply cap, eval-sync cadence) is unrecorded, and its `session.json` records the self-play ply cap and τ instead | Medium–High | Fix: `session.json` bug in P2; recording in P4 | P2, P4 |
| B2 | Self-play Dirichlet noise (α 0.3, ε 0.25, 30 plies) is unrecorded and not covered by the behavior fingerprint | Medium (GUI) | Fix | P4 |
| B3 | A branch loses the init seed and scheme of the weights it starts from | Medium | Fix (in `ancestry`) | P4 |
| B4 | `cum_*` totals mean "this run" after a branch but "these weights" after a derive; the doc claims the latter | Medium | Doc fix + ancestry-summing helper (O-15) | P4 |
| B5 | GUI: which champions generated the training data (promotion chain) is not in the trainer file | Medium (GUI) | Fix | P4 |
| B6 | GUI auto replay-ratio mode: the delays and effective target in force are not the snapshot's values | Low–Medium (GUI) | Document + derived GUI-only field | P4, P5 |
| B8 | Toolchain and build configuration (Xcode, SDK, Debug/Release) are not recorded | Low–Medium | Fix | P1, P4 |
| B9 | A load that recentered the value head changes the starting weights, and nothing records it | Low | Fix | P4 |

(B7, a GUI resume replacing three saved values silently, is folded into gap 13.)

Phases:
- **P1** build identity;
- **P2** CLI resume records what it trains, plus the train-vs-UCI `session.json` fix;
- **P3** GUI values in force;
- **P4** lineage schema 3;
- **P5** documentation;
- **P6** optional pre-lineage reconstruction report.

---

## Corrections to the audit (re-verified for this plan)

- **85 keys, not 83.** `TrainingParameters.allKeys` (`Training/TrainingParameters.swift:2479-2565`) and `collectValues` (`:1756-1860`) both list 85 keys. The earlier count dropped `ArenaSPRTElo0` and `ArenaSPRTElo1`. The real snapshots on disk have 85 entries. 66 declarations are `liveTunable: true` and 19 are `false`.
- **The arena reads its games and threshold live; the stats/chart sample does not.**
  - Both keys are `liveTunable: false` (`:846`, `:835`), but the arena reads them from the singleton at use (`App/SessionController+Arena.swift:389`, `:1064`). Criterion and SPRT are read at each arena start (`:110-117`).
  - The run-start copies `sessionTournamentGames` / `sessionPromoteThreshold` (`App/SessionController+Training.swift:677-678`) are what the stats/chart sample reports (`:2023-2024`). That reporting is fixed in P3.
  - Three values are truly captured at run start: batch size and minimum prefill (`:675-676`), and buffer capacity (`:202-205`, and the reused buffer on continue at `:192-195`).
- **`git_dirty` is `true` for every build.** `DrewsChessMachine/generate-build-info.sh:30` writes the tracked `build_counter.txt` before the dirty test at `:36`, and `App/BuildInfo.swift` is tracked and regenerated by every build. The script's git commands run at the repo root (`REPO_ROOT="$SCRIPT_DIR/.."`, `:18`), so any change under `experiments/` or `documentation/` also counts. All 23 lineage files on disk say `git_dirty: true`.
- **Gap 1 also covers branches and derives (1b).**
  - `LineageTracker.init` sets `segments = []` and `initialization = nil` for `.branch` (`Persistence/LineageTracker.swift:181-187`); `untrainedCopyRecord` uses `segments: []` (`:410`).
  - The live lrA/lrB files branch from the mint `20261005-r7b24-fresh` (model `20261005-22-yRzB`, `init_seed 20261005`, `init_scheme dcm-init-1`), but their own records say `rng.init_seed: null` (B3).
- **The GUI's other consumers of the three captured keys are stale too:**
  - `session.json`'s `batchSize` and `trainingPositionsSeen` (`App/SessionController+Checkpoint.swift:1219-1220`);
  - the heartbeat's effective-LR readouts (`App/SessionController+Heartbeat.swift:161-162`, `:582`);
  - the arena record and log (`App/SessionController+Arena.swift:889`, `:975`).

  The comment at `Heartbeat.swift:579-581` ("Same batch size the optimizer is actually stepping at") is wrong after an edit.
- **`ReplayParams` can disagree with itself.** Its fields are `var`s copied from the snapshot (`CLI/CorpusReplayRunner.swift:31-57`), so a caller can change `trainer` or `trainingBatchSize` without changing `lineageParameters`.
  - Production never does this.
  - Three test files do (`DrewsChessMachineTests/ResumeEquivalenceTests.swift:196-203`, `CorpusReplayRefusalTests.swift`, `FinalTrainerSaveFailureTests.swift`). The files they write record a 4096 batch while training at 32.
- **Records are re-emitted at their original schema.** `championFileLineageRecord` returns `record.withoutTrainerState()` for a recorded file origin (`App/SessionController+Lineage.swift:440-441`). That copies `schema: schema` (`Persistence/LineageRecord.swift:899-913`), so a champion file can carry an older record verbatim. S1 changes this (every write is schema 3).
- **Gap 13 (new): resume parameter changes.**
  - A corpus-replay or train-vs-UCI `--resume-exact` adds a `params` gap only when the parent has no snapshot (`CLI/CorpusReplayRunner.swift:1286`; `CLI/TrainVsUciRunner.swift:254`), the feed per step changed (`CorpusReplayRunner.swift:1268-1272`), or the buffer capacity changed (`:1306-1309`).
  - A GUI resume uses the *current* batch size, promote threshold and arena games, logging `[RESUME-PARAM] … DIFFERS … resume uses current` (`App/SessionParameterResume.swift:253-269`). It adds no gap.

---

## Evidence files used throughout

| File | Format / schema | What it shows |
|---|---|---|
| `Models/20261004-fatconv98-cont-replay-step6093.safetensors` | v8, schema 2, build 2320 `1ab52554`, dirty | `cum_trainer_step 39093`; parameter snapshot covers only `segment_local_step 6093`. Parent `20261004-15-Pm6B` unrecorded; `segments: []`. Snapshot `random_seed "0"` / `random_seed_mode 0` while `rng.streams.master_seed 18019510007828584227 seed_origin drawn`. Flat `trainer_policy_tail_precision fp32_from_pre_bn`, absent from the record. Its `not_exact_items` describe build 2320's resume of the *pre-lineage* `b2275-step33000`. |
| `Models/20261004-fatconv98-b2275-replay-step33000.safetensors` | v5, no lineage, build 2275 | Only `trainer_*` schedule keys, `replay_*` corpus keys and `built_by_*`. No snapshot, tail, seed or argv. Holds the first 33,000 of the 39,093 steps above. |
| `Models/20261003-seg-resume-check-replay-seg1-step1000.safetensors` | v8, schema 2 | A two-segment exact-resume chain. `segments[0]` holds counts, build and device only. Segment 0's parameters, its different `--start-model` argv and its tail are not in this file. Snapshot `random_seed "0"` while the run used `--seed 777` (`seed_origin configured`). |
| `Models/20261005-r7b24-fresh.safetensors` / `20261005-lrA-const01-replay-step1000.safetensors` | v8, schema 2 | Mint with `init_seed 20261005` / branch from it with `init_seed null` (B3). |
| `Sessions/20260921-224358-20260921-3-MNTv-sigusr2.dcmsession/trainer.safetensors` | v3 | Newest GUI session on disk; no lineage. There is no lineage-era GUI, train-vs-UCI, derive or graft file on disk, so those paths are verified from code only. |

**Census** (2026-10-05 03:15 CDT; header-only read of every `.safetensors` under `Models/` and `Sessions/*/`): 4,334 files.
- v3 3,923; v4 38; v5 315; v6 35; v8 23 (21 replay, 2 `new_model`).
- All 23 lineage records are schema 2, with `git_dirty: true`.
- Pre-lineage: 4,311.
- Two training runs (lrA, lrB) are adding files, so the v8 count grows.

Header dumper used by the validation steps (stdlib only, header bytes only):

```python
# one-off; run with python3 dump.py FILE
import json, struct, sys
with open(sys.argv[1], "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    md = json.loads(f.read(n))["__metadata__"]
rec = json.loads(md["dcm_lineage"])
print(json.dumps({k: v for k, v in md.items() if k not in ("architecture", "dcm_lineage")}, indent=1))
snap = rec.pop("parameters")
print(json.dumps(rec, indent=1, sort_keys=True))
if snap: print(json.dumps(json.loads(snap["snapshot_json"]), indent=1, sort_keys=True))
```

---

# Part S — Lineage record schema 3 (all of P4's format changes, in one place)

P4 introduces **lineage schema 3** in one commit. Every field any phase adds is part of it, so no build ever writes a schema-3 record of a different shape.

The architecture format version (`dcm_format_version`, `Network/ArchitectureFormat.swift`) is **not** bumped. The lineage schema is versioned on its own (`Persistence/LineageRecord.swift:40-44`), and schema 1 → 2 (`d962a239`) did not bump it either.

## S1. Reading older records, writing only schema 3 (no fabrication)

- **Accepted schemas.** Today `LineageRecord.init(from:)` refuses any schema but the current one (`Persistence/LineageRecord.swift:833-839`). Schema 3 adds `static let oldestDecodableSchema = 2` and accepts `2...currentSchema`.
  - Schema 1 stays refused. `ExactResumeCompletionTests.testARecordAtAnEarlierSchemaIsRefused` (`DrewsChessMachineTests/ExactResumeCompletionTests.swift:244-250`) keeps passing unchanged.
- **Schema-aware decoding without a defaulted key.** Changed sub-structs get an explicit `init(from decoder: Decoder, schema: Int)`. `LineageRecord.init(from:)` decodes `schema` first and hands it down:
  - keyed members through `c.superDecoder(forKey:)`;
  - `segments` / `ancestry` elements through `nestedUnkeyedContainer(forKey:)` + `superDecoder()`.

  At schema 3 every new key is required (the "every key required" rule, `:26-28`). At schema 2 the new keys must be **absent**, since no schema-2 writer produced them. They decode to the explicit `unrecorded` case.
- **"Unrecorded" in memory.** One generic `enum LineageRecord.Recorded<Value> { case recorded(Value); case unrecorded }`.
  - It encodes as `{"recorded": true, "value": …}` / `{"recorded": false}`.
  - It is used only where a value can come from a record that predates the field.
- **Every write is schema 3.** This includes:
  - a champion file of a loaded schema-2 model (`App/SessionController+Lineage.swift:440-441` → `withoutTrainerState()`, which now stamps `schema: currentSchema`);
  - a derive, graft or untrained copy of a schema-2 source (`Persistence/LineageTracker.swift:355-413`).

  Such a record is built from the source record:
  - `run`, `parent`, `steps`, `fed`, `time`, `parameters` are copied as they are;
  - `configuration` is `{"recorded": false}`;
  - `segments` are mapped by S2's rules for a schema-2 record;
  - `ancestry` follows gap 1b;
  - `build.git_diff_sha256` (and the B8 toolchain fields) are `{"recorded": false}` when the copied `build` is the source's;
  - `fed.corpus` becomes `corpus_identity: first_only` (gap 7).

  Nothing is invented: every field is either copied, built by this process, or marked unrecorded.
- **A derive, graft or untrained copy of a schema-3 source** (review N4) carries the source's `configuration` verbatim, as it carries `parameters` today (`Persistence/LineageTracker.swift:403`):
  - `{"recorded": false}` stays unrecorded;
  - `null` stays null;
  - a recorded value keeps its `path_kind` and every field.

  The copy's own `invocation` (e.g. `path_kind: derive`, `:405`) is unrelated, and the S2 invariants are keyed to `configuration.value.path_kind`. A legitimate derive of a train-vs-UCI or GUI file therefore always encodes and decodes.
- **Carried snapshots are never re-checked.** Gap 9's seed-key exclusion applies only to snapshots this build *composes* (`Parameters.init(values:)` refuses the seed keys). Decoding never checks key count or seed keys, so a carried 85-key schema-2 snapshot keeps its exact text and sha256.
- **`session.json`** embeds the same `LineageRecord` (`Persistence/SessionCheckpointFile.swift`), which decodes its own schema, so session files need no format bump.
- **Python mirrors.**
  - `scripts/dcm_lineage.py:51` `SUPPORTED_SCHEMA` stays equal to `currentSchema` (pinned, unchanged, by `documentation/dashboards/tests/test_lineage.py:133-134`).
  - A new `OLDEST_SUPPORTED_SCHEMA = 2` mirrors `oldestDecodableSchema`.
  - `validated_record` (`:169-171`) accepts the range; its required-key lists (`:119-132`, `:174-177`) gain the schema-3 keys, required only at schema 3. These include each `parameter_changes` entry's `{trainer_step, recorded_unix, id, old, new, restamped_from}` and each `champion_changes` entry's keys.
  - `documentation/dashboards/ckpt_inventory.py:71` reads both corpus shapes.
- **Forward compatibility is not provided.** A pre-P4 build refuses a schema-3 file (R1).

## S2. Schema-3 record shape

New or changed keys are marked; everything else is unchanged from schema 2.

```jsonc
"dcm_lineage": {
  "schema": 3,                                         // CHANGED
  "run": { … unchanged … },
  "parent": { … unchanged … },
  "steps": { … unchanged … },                          // doc of cum_* corrected (B4)
  "fed": {
    "cum_games": …, "cum_positions": …, "segment_games": …, "segment_positions": …,
    "corpus": {                                        // corpus replay only; null elsewhere (unchanged rule)
      "corpus_identity": {"listed": [                  // CHANGED (gap 7): replaces corpus_id / corpus_path
        {"corpus_id": "20260624-192615-w3aA5b", "corpus_path": "/Users/…/Corpora/20260624-192615-w3aA5b", "shard_count": 46}
      ]},                                              // or {"first_only": {"corpus_id", "corpus_path"}} for a carried schema-2 position
      "segment_start": {"epoch": 0, "next_game_index": 4254363},   // NEW (D7): where this segment's feed began
      "epoch": 0, "next_game_index": 5038653, "shard": 11, "populated_plies": 500000,
      "buffer_capacity": 500000, "feed_ahead_positions": -8488, "feed_per_step": 8533,
      "shard_sha256": [ … ]                            // for "listed": count == Σ shard_count (checked on encode and decode)
    }
  },
  "time": { … unchanged … },
  "parameters": {"snapshot_json": "…", "sha256": "…"}, // in force at save (gaps 3, 5); composed snapshots exclude random_seed* (gap 9)
  "configuration": null                                // NEW: null = no training behind these weights (mint)
                 | {"recorded": false}                 //   carried from a record that predates the field
                 | {"recorded": true, "value": {
      "path_kind": "replay",                           // N4: the path whose segment composed this configuration
      "policy_tail_precision": "fp32_from_pre_bn",     // gap 8
      "budget": {"training_step_limit": null, "training_time_limit_sec": null, "epoch_limit": 12},  // gap 11; resolved limits
      "parameter_changes": [],                         // gap 4 (+ recaptures, A3); always [] on CLI paths
      "champion_changes": [],                          // B5; [] unless configuration.value.path_kind == "gui"
      "vsuci": null,                                   // B1 + gap 11: object exactly when configuration.value.path_kind == "vsuci"
      "self_play_dirichlet": null,                     // B2: object exactly when configuration.value.path_kind == "gui"
      "start_value_head_recentered": false | true | null,  // B9: whether the segment's start weights were recentered by a load (false for weights built in-process, promoted, or a fresh CLI run); null only when the segment has no new start weights (continue, keep-trainer)
      "schedule_at_save": {"cycle_step": 38093, "learning_rate_fed": 0.000123, "momentum_fed": 0.87} | null,  // gap 12; derived (names per O-18, see gap 12)
      "replay_ratio_at_save": null                     // B6: object exactly when configuration.value.path_kind == "gui"; derived
    }},
  "build": {"build_number": …, "git_hash": "…", "git_branch": "main",
            "git_dirty": false,                        // meaning corrected from P1 (scope DrewsChessMachine/, generated files excluded)
            "git_diff_sha256": {"recorded": true, "value": null},     // NEW (gap 6): value null exactly when git_dirty is false
            "xcode_build": {"recorded": true, "value": "17A5241e"},   // NEW (B8)
            "sdk_build": {"recorded": true, "value": "26A5…"},        // NEW (B8)
            "configuration": {"recorded": true, "value": "Release"}}, // NEW (B8)
  "invocation": { … unchanged … },
  "device": { … unchanged … },
  "rng": { …, "streams": { …, "seed_origin": "configured" | "command_line" | "drawn" } },  // CHANGED (gap 9)
  "segments": [ <SegmentSummary v3> ],                 // CHANGED (gap 1)
  "ancestry": [ <AncestorRun> ],                       // NEW (gaps 1b, B3)
  "derivation_history": [ … unchanged … ]
}
```

The `Recorded` wrappers inside `build` exist because a schema-3 file can carry a schema-2 source's `build` (S1). For a build this process writes they are always `recorded`.

**Invariants (checked on encode and decode).**
- `parameters == null` ⇒ `configuration == null`. The converse does not hold: a carried schema-2 record has `parameters` and `configuration: {"recorded": false}`.
- Within a recorded `configuration.value` (review N4), the invariants are keyed to **`configuration.value.path_kind`**, never to the record's `invocation.path_kind`. A derive or untrained copy carries a source's configuration under its own `invocation`, so its two `path_kind`s legitimately differ:
  - `vsuci` non-null ⇔ `configuration.value.path_kind == "vsuci"`;
  - `self_play_dirichlet` and `replay_ratio_at_save` non-null ⇔ `configuration.value.path_kind == "gui"`;
  - `champion_changes` is `[]` unless `configuration.value.path_kind == "gui"`.
- `configuration.value.path_kind` is never `derive` or `new_model`, because those writers compose no configuration; they carry one or have none.

**`configuration.vsuci`** (B1, gap 11):
```jsonc
{"max_plies_per_game": 400, "eval_sync_every_steps": 10,
 "trainer_move_selection": {"start_tau": 0.01, "decay_per_ply": 0, "floor_tau": 0.01, "dirichlet": null},  // from SamplingSchedule.argmax
 "opponents": [{"command": "/opt/homebrew/bin/stockfish", "executable_sha256": "…", "count": 4, "go_limit": "nodes 1",
                "options": [{"name": "UCI_Elo", "value": "1400"}], "id_name": "Stockfish 17", "id_author": "…"}]}
```

**`configuration.self_play_dirichlet`** (B2): `{"alpha": 0.3, "epsilon": 0.25, "ply_limit": 30}`, read from `SamplingSchedule.selfPlay.dirichletNoise`.

**`configuration.replay_ratio_at_save`** (B6, GUI): `{"auto_adjust": true, "effective_target": 0.51, "computed_step_delay_ms": 40, "computed_self_play_delay_ms": 0}`.

**`SegmentSummary` at schema 3.** All existing keys (`Persistence/LineageRecord.swift:713-766`), plus:

```jsonc
"configuration": {"recorded": true, "value": { …same shape as the record's configuration value… } | null} | {"recorded": false},
"parameters":    {"recorded": true, "value": {"snapshot_json": "…", "sha256": "…"} | null} | {"recorded": false},
"corpus_identity": {"recorded": true, "value": {"listed": […]} | {"first_only": {…}} | null} | {"recorded": false},
"segment_start_corpus": {"recorded": true, "value": {"epoch": 0, "next_game_index": 0} | null} | {"recorded": false},
"path_kind":     {"recorded": true, "value": "replay"} | {"recorded": false},
"argv":          {"recorded": true, "value": ["…"]} | {"recorded": false}
```

What each summary field holds:

| Field | Summary of a schema-3 record | Summary of a schema-2 record | Summary carried from inside a schema-2 record's `segments` |
|---|---|---|---|
| `configuration` | the record's (null / unrecorded / value) | `{"recorded": false}` | `{"recorded": false}` |
| `parameters` | recorded | recorded (schema 2 has it) | `{"recorded": false}` |
| `corpus_identity` | recorded `listed` (or null when not replay) | recorded `first_only` | `{"recorded": false}` |
| `segment_start_corpus` | recorded | `{"recorded": false}` | `{"recorded": false}` |
| `path_kind`, `argv` | recorded | recorded (schema 2 has `invocation`) | `{"recorded": false}` |

A schema-2 parent's flat `trainer_policy_tail_precision` is not read into its summary (O-6). The summary's policy tail therefore lives inside `configuration`, which is unrecorded for such a parent.

**`AncestorRun`** (gaps 1b, B3), oldest first:

```jsonc
{"lineage_run_id": "C378E0D2-…",
 "left_by": "branch" | "derive",
 "left_at": {"model_id": "…", "content_sha256": "…" | null, "trainer_completed_steps": 2000 | null},
 "initialization": {"recorded": true, "value": {"init_seed": "20261005", "init_scheme": "dcm-init-1"} | null} | {"recorded": false},
 "segments": [ <SegmentSummary v3> ]}
```

- `left_at` is the next run's `parent` (copied from `LineageTracker.ParentFile.recordParent`, `Persistence/LineageTracker.swift:70-78`).
- `initialization` comes from the ancestor record's `rng.initialization` (`Persistence/LineageRecord.swift:508-516`), read together with how that ancestor *run* began (review N3, refined).
  - A null `rng.initialization` is ambiguous by the field's own doc: "started from another file's weights", **or** "a run continuing a file written before this field".
  - The run's first segment decides which. Its start is `segments.first?.start ?? run.start`: a resumed segment's own `run.start` is `resume`, so a run that began as a branch and was later resumed must be read from its first summary.

  | Ancestor `rng.initialization` | Run's first-segment start | `run.continues_unrecorded_history` | Recorded as |
  |---|---|---|---|
  | value | any | any | `recorded(value)` |
  | null | `branch` or `derive` | any (a derive of a pre-lineage file has `true`, `Persistence/LineageTracker.swift:381`) | `recorded(null)`: the run started from a file's weights. That file's own history is its `ancestry` entry, or unrecorded. |
  | null | `resume` (resume of a file with no lineage, or an untracked GUI trainer) | true | `{"recorded": false}` |
  | null | `fresh` | any | decode error: a fresh run always records its initialization (`Persistence/LineageTracker.swift:174-176`) |
  | null | any other combination | — | decode error, named in the message; never guessed |

**Size.** About 3.6 KB per summary. A 50-segment line is about 180 KB of header, against the 64 MB header guard (`Persistence/ModelFileCatalog.swift:252`). Nothing is ever truncated (R4).

## S3. Single source of truth, per value

| Value | Source | Copies, and how they are kept equal |
|---|---|---|
| Parameters in force | `parameters.snapshot_json` | `configuration` holds no parameter values. A summary's `parameters` is a copy of *that segment's* record, a different fact. |
| Schedule of a trainer-state file | The exported `TrainerResumeSnapshot.schedule`, written as flat `trainer_lr_warmup_steps` / `trainer_lr_momentum_cycle` / `_envelope` (`Training/TrainerResumeState.swift:127-157`) and read on resume | The snapshot's 21 schedule keys are composed *from the same exported schedule* (gap 5 rule), so they agree by construction. `SafetensorsModelIO.encode` refuses a disagreement as a backstop (`IOError.scheduleDisagreesWithLineage`). |
| Policy tail of a trainer-state file | Flat `trainer_policy_tail_precision`, read on resume (`Persistence/SafetensorsModelIO.swift:166-167`, `:329-330`) | `configuration.policy_tail_precision`. Encode refuses a disagreement when both are present (`IOError.policyTailDisagreesWithLineage`), the same pattern as `lineageStepDisagreesWithTrainerClock` (`:150-157`). |
| Seed | `rng.streams.master_seed` + `seed_origin` | None: composed snapshots drop `random_seed*` (gap 9). |
| Corpus identity | `fed.corpus.corpus_identity` | Summaries copy it. |
| Effective LR/momentum fed to the optimizer | Today **two** host-side implementations: the feed math in `buildFeeds` (`Training/ChessTrainer.swift:6102`; warmup `:6154-6158`, base LR `:6173`, √batch `:6174-6182`, warmup applied `:6183`, momentum `:6203`) and the readouts `effectiveLearningRate` / `effectiveMomentum` (`:4437-4462`, `:4473-4479`), which say they mirror it. This is a pre-existing single-source violation (review N6). With O-18 approved: one function, `LRMomentumCycleReadout.values(schedule:staticLearningRate:staticMomentum:batchSize:sqrtBatchScaling:)`, called by `buildFeeds` **and** both readouts. | `schedule_at_save` is computed by that function from the record's own inputs and never read back; with O-18 its values are the fed values by construction. If O-18 is declined, the fields are named `learning_rate_readout` / `momentum_readout` and documented as the status-bar readout that mirrors `buildFeeds` (gap 12). |
| Self-play Dirichlet | `SamplingSchedule.selfPlay.dirichletNoise` (`Network/MPSChessPlayer.swift:52-56`, used at `App/SessionController.swift:1040`) | `configuration.self_play_dirichlet`, read from it at save. |
| Train-vs-UCI move selection | `SamplingSchedule.argmax` (`Network/MPSChessPlayer.swift:153-160`, used at `CLI/TrainVsUciRunner.swift:497`) | `configuration.vsuci.trainer_move_selection`, read from the schedule the driver is given. |
| Build diff, toolchain | `BuildInfo` (generated) | `build.*` at save. |

---

# Part G — Per-gap design

## Gap 1 — earlier segments lose their configuration (Critical)

**Problem.**
- `SegmentSummary` (`Persistence/LineageRecord.swift:713-766`) holds index, ID, start, times, step bounds, games, positions, step/wall seconds, the exact flag, build and device. It holds no parameters, argv, path, policy tail or corpus.
- `LineageTracker.init` `.resume` appends `SegmentSummary(of: record)` of the parent (`Persistence/LineageTracker.swift:201`). So every resume throws away the parent segment's configuration.

**How it misleads.**
- `20261003-seg-resume-check-replay-seg1-step1000` reads as if all 2,000 steps ran under one argv and parameter set. Segment 0's argv (a different `--start-model`) and parameters are not in the file.
- With gap 13, a resume that changed weight decay leaves a file whose snapshot claims the new weight decay for every step.

**Decision.** Fix in code (P4).

**Design.**
1. `SegmentSummary` gains `configuration`, `parameters`, `corpus_identity`, `segment_start_corpus`, `path_kind` and `argv` (S2).
   - `SegmentSummary.init(of record:)` (`:750-766`) fills them from the record and the record's schema. Every value comes from the record being summarized.
2. `LineageTracker` stays the one builder (`Persistence/LineageTracker.swift:5-17`). The resume case keeps `record.segments + [SegmentSummary(of: record)]` (`:201`).
3. No save site changes for this gap. Every path records through `LineageTracker.record`:
   - corpus replay `CLI/CorpusReplayRunner.swift:1576-1592`;
   - train-vs-UCI `CLI/TrainVsUciRunner.swift:545-556`;
   - GUI `App/SessionController+Lineage.swift:377-389`.
4. Champion files: `withoutTrainerState()` (`Persistence/LineageRecord.swift:899-913`) keeps `segments`, `configuration` and `ancestry`, drops only RNG state (as today) and stamps `schema: currentSchema` (S1).

**Tests** (new `DrewsChessMachineTests/SegmentConfigurationRecordTests.swift`):
- `testAResumedRecordCarriesTheEarlierSegmentsParametersAndArgv` (regression).
  - A fresh tracker records with snapshot P1 and argv A1; a resumed tracker records with P2 and A2.
  - Asserts, on `JSONSerialization` of `jsonText()`, that `segments[0].parameters.value.sha256 == P1.sha256` and `segments[0].argv.value == A1`.
  - Written after P4's API-first step (O-2): it fails then and passes after recording, unmodified.
- `testASchemaTwoParentsSummaryKeepsItsParametersAndMarksTheRestUnrecorded`.
  - Uses the real `dcm_lineage` text of `seg-resume-check-replay-seg1-step1000`, held as a fixture literal (P4 step 5).
  - Asserts:
    - the new summary's `parameters` is recorded with sha `fbd09dd6…`;
    - `configuration` and `segment_start_corpus` are `{"recorded": false}`;
    - `corpus_identity` is `first_only 20260624-192615-w3aA5b`;
    - the carried `segments[0]` has every new field `{"recorded": false}`.
- `testSchemaThreeRequiresEverySummaryKey`.
- `testSchemaTwoRecordWithASchemaThreeKeyIsRefused`.
- `testAChampionFileOfALoadedSchemaTwoModelIsSchemaThreeWithUnrecordedConfiguration` (A8).
- `testACarriedEightyFiveKeySnapshotDecodesAtSchemaThree` (A8).
- `testChampionRecordKeepsSegmentsConfigurationAndAncestry` (complements `ChampionLineageRecordTests`, which is unchanged).

**Validation.** V4: the segment-1 file's `segments[0].parameters.value.sha256` equals segment 0's `parameters.sha256`, and `argv.value` equals segment 0's `invocation.argv`.

## Gap 1b — a branch or derive drops the parent run's history (Critical)

**Problem.**
- `.branch` sets `segments = []` and a new `lineage_run_id` (`Persistence/LineageTracker.swift:181-187`).
- `untrainedCopyRecord` does the same for `--derive-model`, a graft and an untrained GUI copy (`:371-412`, `segments: []` at `:410`). It carries only the source's `parameters` (`:403`).

Branching is the most common start in current experiments: lrA, lrB and `seg-resume-check-replay-step1000` are branches. The child names its parent (model ID and content hash) and nothing else about how those weights were trained.

**Decision.** Fix in code (P4). Severity raised to Critical (review C1).

**Design.**
- New record key `ancestry: [AncestorRun]` (S2).
- **Branch:** `ancestry = parentRecord.ancestry + [AncestorRun(lineageRunID: parentRecord.run.lineageRunID, leftBy: .branch, leftAt: file.recordParent, initialization: AncestorRun.initialization(of: parentRecord) /* S2 table */, segments: parentRecord.segments + [SegmentSummary(of: parentRecord)])]`.
- **Derive / graft / untrained copy:** the same with `leftBy: .derive`.
- **Resume:** carried verbatim. **Fresh:** `[]`.
- **Unrecorded parent** (e.g. `b2275-step33000`): `[]`. `run.continues_unrecorded_history` stays the signal (`:206-216`).
- `AncestorRun.initialization` follows S2's table: the ancestor's `rng.initialization`, read with how that run began. A schema-2 record carries `rng.initialization` (schema 2 has `init_seed` / `init_scheme`), so the same table applies to it.

**Tests** (in `SegmentConfigurationRecordTests.swift`):
- `testABranchCarriesTheParentRunAsAncestry` (regression).
- `testADeriveCarriesTheSourceRunAsAncestry`.
- `testADeriveOfASchemaTwoSourceCarriesItsParametersAndMarksConfigurationUnrecorded` (A8).
- `testABranchOfABranchKeepsBothAncestorsOldestFirst`.
- `testABranchOfAnUnrecordedParentHasNoAncestryAndSaysSo`.
- `testAResumeCarriesAncestryVerbatim`.
- `testABranchFromAMintCarriesTheMintsInitSeedInAncestry` (B3).
- `testAnAncestorThatResumedAnUnrecordedFileHasUnrecordedInitialization`: uses the fatconv98-cont fixture (first segment `resume`, `continues_unrecorded_history true`, `init_seed null`) (N3).
- `testAnAncestorThatBranchedThenResumedIsReadFromItsFirstSegment`: a branch run resumed once has `run.start resume`, and its first summary says `branch`, so the result is `recorded(null)` (N3).
- `testAFreshAncestorWithoutInitializationIsADecodeError` (N3).
- `testADeriveOfAVsuciFileKeepsItsVsuciConfiguration`: the copy's `invocation.path_kind` is `derive` and `configuration.value.path_kind` is `vsuci`; it encodes and decodes (N4).
- `testAnUntrainedGuiCopyOfAReplayFileKeepsTheReplayConfiguration` (N4).

**Validation.**
- V4: `--derive-model --from <segment-1 file of V4> …` produces `ancestry[0].segments` with two recorded summaries, and `ancestry[0].left_at.model_id == parent.model_id`.
- A branch from a mint carries the mint's `init_seed`.

## Gap 2 — the pre-lineage archive (High, historical)

**Problem.** 4,311 files at format ≤ 6 carry no `dcm_lineage`. `b2275-step33000` holds only the schedule and corpus keys. The parameters behind 33,000 of the fatconv98 line's 39,093 steps are in no model file.

**Decision.** No change to model files (rewriting is migration). Optional tooling (P6, O-8): a **read-only reconstruction report**.

**Design (if approved).**
- `scripts/reconstruct_pre_lineage_params.py`, built on `scripts/dcm_session_logs.py` (log ordering) and the `scripts/model_lineage_report.py` parsing conventions.
- **Inputs:** the session logs (`~/Library/Logs/DrewsChessMachine/`; 9,639 files at this writing, 138 containing `[REPLAY-HPARAMS]`), and each `experiments/*/parameters*.json` an experiment README names.
- **Join keys:**
  - `[REPLAY] start-model: … modelID=` (`CLI/CorpusReplayRunner.swift:1006`);
  - the run's own model ID on recorder/stats lines (`:1963`);
  - `[REPLAY] saved trainer model (…) step=… -> <file>` (`:1609`);
  - the file's `created_at_unix` against the log's timestamp window.

  A file is matched only when model ID, file name and time window all agree. Anything ambiguous is reported unmatched, never guessed.
- **Output:** JSON (`--out`, never over an existing file), one entry per file: `{model_id, file, status: reconstructed|unmatched, source_log, hparams_line, cycle_line, parameters_file?}`.
  - Every value is labelled *reconstructed from logs*.
  - It is never written into a `.safetensors` file, nor fed to lineage or dashboards as measured data.
- **Tests:** `documentation/dashboards/tests/test_reconstruct_pre_lineage_params.py`, with synthetic log snippets: a clean match, an ambiguous match, and a missing log.

**Why optional.**
- Coverage is partial: 138 replay logs against 4,311 files, and GUI runs have no single hyperparameter banner.
- The dashboards already pin those runs by hand in `registry.json`.

## Gap 3 — GUI records the edited value of run-start-captured parameters (High)

**Problem.**
- Each Play-and-Train start captures `training_batch_size` and `replay_buffer_min_positions_before_training` into locals (`App/SessionController+Training.swift:675-676`, used at `:1219` and `:1240`).
- The buffer is built with `replay_buffer_capacity` (`:202-205`) or reused on continue (`:192-195`).
- The popover writes these keys to the singleton mid-run (`App/UpperContentView/TrainingSettingsPopoverModel.swift:1232-1243`; `liveTunable: false` at `Training/TrainingParameters.swift:759`).
- Every save snapshots the singleton (`App/SessionController+Lineage.swift:385`).

**How it misleads.** A run stepping at batch 4096, with the field edited to 1024 mid-run, saves files that say 1024. Recomputing the sqrt-batch LR (`Training/ChessTrainer.swift:1230`, `:4437-4460`) from such a file is off by a factor of 2.

**Decision.** Fix in code (P3). O-5 asks whether the fields should also be disabled during a run.

**Design.**
1. New `Training/RunStartParameterCapture.swift`:
   - `struct RunStartParameterCapture: Sendable, Equatable { trainingBatchSize, replayBufferMinPositionsBeforeTraining, replayBufferCapacity }`;
   - `static let capturedKeyIDs` (the three IDs);
   - `func inForce(over: TrainingParametersSnapshot) -> TrainingParametersSnapshot`, which replaces exactly those three values;
   - `replayBufferCapacity` is read from the run's `ReplayBuffer.capacity`: the measured value, which on continue is the reused buffer's.
2. `SessionController` stores `runStartCapture: RunStartParameterCapture?`.
   - It is set by `beginRunStartCapture(buffer:)`, which `startRealTraining` calls at every start (continue included) right after the buffer is chosen.
   - It is kept through Stop, since a save between Stop and Continue describes the steps just trained. It is replaced at the next start and cleared only when the session is torn down.
   - The locals at `:675-676` read from it.
3. Every in-run reader takes the capture:
   - `lineageRecordForSave` (`:385`, composed as in gap 5);
   - the session-state builder (`App/SessionController+Checkpoint.swift:1219-1220`);
   - the heartbeat (`App/SessionController+Heartbeat.swift:161-162`, `:582`);
   - the arena record and log (`App/SessionController+Arena.swift:889`, `:975`);
   - the stats/chart sample (`App/SessionController+Training.swift:2021-2024`): batch from the capture, promote threshold and arena games read live, where the arena reads them.

   Inside a run a missing capture throws `LineageSegmentError.noSegment` (as the existing guards at `App/SessionController+Lineage.swift:362-376` do). There is no fallback to the singleton.
4. Popover: while a capture exists, committing a changed captured key logs `[PARAM] trainingBatchSize: 4096 -> 1024 (applies at the next Play-and-Train start; this run keeps 4096)`. The field's existing view shows the caption "Applies at the next Play-and-Train start" (`TrainingSettingsPopover.swift`).
5. **Captured keys inside one segment** (review A3).
   - The capture is re-taken at every start, and a segment can span several starts: Continue after Stop; "New Session, keep trainer", which keeps the tracker (`App/SessionController+Lineage.swift:182-185`) while building a new buffer at the current capacity (`App/SessionController+Training.swift:192-205`, `continueMode` false).
   - When a start that keeps the segment captures a value different from the previous capture, it appends one `parameter_changes` entry per changed key: `trainer_step` = `trainer.completedTrainSteps` at that start, `old` = previous capture, `new` = this capture.
   - Assignments to captured keys during a run are not journalled (gap 4), because they do not take effect.

**P3 order** (review A5):
1. API-first: `RunStartParameterCapture`, `runStartCapture`, `beginRunStartCapture(buffer:)`, with no reader switched yet.
2. Regression tests, run and seen to fail.
3. Switch the readers.

`GuiSaveHarness` and `GuiLineageLifecycleTests` install trackers directly. They need a `beginRunStartCapture` call next to each (O-14), or they would hit the no-capture error.

**Tests** (new `DrewsChessMachineTests/RunStartParameterCaptureTests.swift`; every test that assigns `TrainingParameters.shared` uses the `TrainerHyperparametersTests` pattern (`DrewsChessMachineTests/TrainerHyperparametersTests.swift:31-41`): snapshot in `setUp`, `suppressPersistence = true`, restore in `tearDown`):
- `testInForceReplacesOnlyTheCapturedKeys` (pure; all 85 keys).
- `testCapturedKeysAreNotLiveTunable`.
- `testAGuiSaveRecordsTheBatchSizeTheRunTrainsAt` (regression).
  - `GuiSaveHarness`: capture at batch 64, set the singleton to 128, call `lineageRecordForSave`, and assert 64.
  - If the harness cannot run without Metal, the test is gated like other GPU tests and run on the owner's machine.
- `testSessionStateBatchSizeIsTheRunsBatchSize` (regression).
- `testACapturedKeyEditDuringARunIsNotJournalled`, `testARecaptureAtContinueIsJournalledAtTheStartClock` (P4, with gap 4).

**Validation (V3, GUI; only with no training run live).**
1. Build New Model (smallest preset); Play-and-Train about 100 steps.
2. Settings ▸ batch 4096 → 1024 ▸ Save; File ▸ Save Session.
3. Expect `trainer.safetensors` `training_batch_size` to be the run's value, the `[PARAM] … this run keeps …` line, and `session.json` `batchSize` equal to it.
4. Stop ▸ Continue ▸ Save: after P4, `parameter_changes` holds one `training_batch_size` entry at the continue clock.

## Gap 4 — GUI live edits leave no trace (High)

**Problem.**
- The GUI snapshot is taken at save (`App/SessionController+Lineage.swift:385`).
- 66 of the 85 keys are `liveTunable: true`. A change at step 30,000 of a 60,000-step segment survives only as the value at save, and in `[PARAM]` log lines.

**How it misleads.** A file trained at LR 0.001 for most of a segment and at 0.0005 for its last 500 steps reads as an LR-0.0005 run.

**Decision.** Fix in code (P4).

**Design.**
1. The choke point is `TrainingParameters.commitAssignment` (`Training/TrainingParameters.swift:2449-2474`), called from every stored property's `didSet` (`:1533-1657`; the label-smoothing mode has a hand-written multi-line one at `:1541-1547`).
   - It gains the old value: each `didSet` passes `oldValue`, a mechanical edit per property.
   - **Placement:** the observer is called after validation succeeds and **before** the `if suppressPersistence { return true }` early return (`:2471`), so persistence-suppressed assignments (`releaseRunHolds`, CLI loads, tests) are seen.
   - It is also called on the `admittingSessionValueOutsideDeclaredRange` path (`:2454-2459`). Only a resume reaches that path, and it runs before the new segment's tracker exists, so those entries land in a journal that is discarded (point 3).
2. The observer is `nonisolated static let runChangeObserver: SyncBox<(@Sendable (String, ParameterValue, ParameterValue) -> Void)?>`, next to the existing static flags (declared at `:2270`, `:2276`, `:2291`).
   - The closure is `@Sendable` (review NB4).
   - `SessionController` installs and removes it on the main actor. It captures the segment's `ParameterChangeJournal` (a `Sendable` class) and the trainer's clock box **weakly**, so a lingering observer cannot keep a replaced trainer or journal alive.
   - **Skipped assignments:**
     - assignments made while `assigningRunHold` is set (a resume's run-only holds, `:2361-2364`, `:2403-2406`);
     - assignments to `LineageRecord.Parameters.excludedParameterIDs` (the seed settings; review NB2). A mid-run seed edit does not affect the running run, and those keys are not in composed snapshots, so the undo derivation stays over the snapshot's own keys.
3. `ParameterChangeJournal` (`Persistence/ParameterChangeJournal.swift`; a `final class @unchecked Sendable` with a `SyncBox<[ParameterChange]>`, lock discipline in its class comment) is owned by the GUI segment's `LineageTracker`.
   - Entries go to the journal of the tracker that is current when the assignment commits. A replaced tracker takes its journal with it, unsaved.
   - Consequences:
     - (a) a resume's `applyGuiSession` (`App/SessionController+Training.swift:107-118`) writes into the previous segment's journal, which `beginLineageSegment` then discards (`App/SessionController+Lineage.swift:230-233`). Correct: the new segment starts from those values.
     - (b) "New Session, keep trainer" releases run holds at the top of `startRealTraining` (`App/SessionController+Training.swift:38-45`; `releaseRunHolds` sets only `suppressPersistence`, `Training/TrainingParameters.swift:2310-2319`). Those releases are journalled in the kept segment, which is correct: its trainer takes the released values (`App/SessionController+Training.swift:105-122`, `!continueMode`).
   - Each entry is `{trainer_step, recorded_unix, id, old, new, restamped_from}`.
     - `trainer_step` is `trainer.completedTrainSteps` (a `SyncBox` read, `Training/ChessTrainer.swift:4357-4360`) at commit. The change applies from the next step started after it.
     - `restamped_from` is `null`, or the original step when an arena promotion rewound the clock past the entry (gap 5 design 2, P4-1).
   - **Encode check:** no entry's `trainer_step` exceeds the record's `steps.cum_trainer_step` (and, in a summary, its `end_trainer_step`). A violation is a writer bug, and the save refuses (`IOError.journalEntryAfterRecordClock`). The check is skipped when that step count is `null` (a record or summary with no recorded trainer clock): there is nothing to compare against, and nothing is assumed.
   - Captured keys follow gap 3 point 5.
4. `LineageTracker.record` writes `configuration.parameter_changes` from the journal. CLI paths use an explicitly empty journal: their run parameters are an immutable snapshot (P2), so `[]` is true by construction.

**Tests** (new `DrewsChessMachineTests/ParameterChangeJournalTests.swift`, persistence-safe as in gap 3):
- `testACommittedLiveChangeIsJournalledWithTheTrainerStep`.
- `testARejectedAssignmentIsNotJournalled`.
- `testRunHoldAssignmentsAreNotJournalled`.
- `testAResumesRestoresAreNotInTheNewSegmentsJournal`.
- `testReleasingHoldsAtNewSessionKeepTrainerIsJournalled`.
- `testAPersistenceSuppressedAssignmentIsJournalled`.
- `testASeedSettingEditIsNotJournalled` (NB2).
- `testAnEditDuringAPromotedArenaIsRestampedToTheArenaStartStep` (P4-1).
- `testAnEditDuringAKeptArenaKeepsItsStep` (P4-1).
- `testNoJournalEntryIsAfterTheRecordsClock` (P4-1; the encode check, including summaries).
- `testAnObserverDoesNotRetainAReplacedJournal` (NB4).
- `testUndoingTheChangesFromTheSaveSnapshotGivesTheSegmentStartSnapshot`: the segment-start snapshot is derivable, so no second copy is stored.
- `testAGuiSaveRecordsALiveLearningRateChange` (regression; `GuiSaveHarness`).
- `testAReplayRecordHasNoParameterChanges` (in `SegmentConfigurationRecordTests.swift`).

**Validation.** V3: change LR mid-run and save. Expect one `parameter_changes` entry whose `trainer_step` is within one step of the `[STATS]` step at that time, and whose `old` / `new` match the `[PARAM]` line.

## Gap 5 — the snapshot's schedule can contradict the schedule actually trained (Medium–High)

**Problem.**
- **CLI.** On `--resume-exact` the trainer adopts the checkpoint's warmup and cycle (`CLI/CorpusReplayRunner.swift:1046-1052`; `CLI/TrainVsUciRunner.swift:344`; `Training/TrainerResumeState.swift:356-361`), but the record gets the configured `p.lineageParameters` (`CLI/CorpusReplayRunner.swift:51`, `:1588`; `CLI/TrainVsUciRunner.swift:552`).
- **GUI** (review A2).
  - The session save pauses training and awaits `trainer.exportResumeSnapshot()` (`App/SessionController+Checkpoint.swift:432-447`), then `dropoutStreamState()`, and only then reads the singleton through `lineageRecordForSave` (`:469`; `App/SessionController+Lineage.swift:385`).
  - The main actor is free during those awaits. A popover commit can write the singleton and push new values to the trainer (`App/UpperContentView/TrainingSettingsPopoverModel.swift:1584`); the training pause does not gate `apply(to:)`.
  - The file then pairs old flat `trainer_*` keys with a new snapshot.

**How it misleads.** A run trained at `lr_cycle_period_steps 20000`, resumed under a parameters file that says 10000, records 10000 in every file of the resumed segment.

**Decision.** Fix in code: CLI in P2, GUI in P4, under one rule.

**Design.**
1. `extension TrainingParametersSnapshot { func adoptingSchedule(_ s: TrainerScheduleState) -> TrainingParametersSnapshot }`, in `Training/LRMomentumCycle.swift` beside the forward mapping (`:327-366`).
   - It writes `lr_warmup_steps` and the 20 cycle/envelope/follow keys from `s`; types are identical, so it is exact.
   - It builds the result with `TrainingParametersSnapshot(values:)` and **does not validate or clamp** against today's declared ranges. The checkpoint's value is what ran, even if a range has since narrowed. Its doc comment says so (review D1).
2. **One rule for every trainer-state save.** The lineage `Parameters` are `inForceSnapshot.adoptingSchedule(exported.schedule).lineageValues()`, where `exported` is the `TrainerResumeSnapshot` this save writes. The 21 schedule keys therefore come from the same value as the flat `trainer_*` keys.
   - **CLI (P2):** `ReplayParams` becomes derived from one snapshot (O-3): `let parameters`, with every other field a `let` computed in `init`. `func adoptingSchedule(_:) throws -> ReplayParams` rebuilds it. Both runners replace `trainerHyperparameters = p.trainer.adoptingSchedule(…)` with `p = try p.adoptingSchedule(…)`. A CLI trainer cannot change its schedule mid-run, so the run-start adoption equals every save's export.
   - **GUI (P4):** `lineageRecordForSave(at:trainerCompletedSteps:schedule:dropoutPhiloxState:dropoutStreamState:)` gains a required `schedule: TrainerScheduleState` (no default).
     - Saves pass `trainerSnapshot.schedule` (`App/SessionController+Checkpoint.swift:469`).
     - **Arena promotion (review N2).** Today the arena-start snapshot is a tuple of weights, velocity, completed steps and dropout state (`App/SessionController+Arena.swift:170-181`), with no schedule. The promotion rewinds those four (`:455-467`), not the warmup/cycle configuration, which a popover commit can change while the arena runs.
       - The arena start therefore also captures `TrainerScheduleState(currentlyRunningOn: trainer)` and the in-force snapshot `runStartCapture.inForce(over: TrainingParameters.shared.snapshot())`, in the same main-actor turn that starts the existing detached export, and **after** the arena has acquired the training pause (`App/SessionController+Arena.swift:134`; the export follows at `:170-181`). The clock in the captured `TrainerScheduleState` then equals the `completedSteps` read inside the detached export (`:173`), which the post-promotion save below relies on (review pass 3, NB-c).
       - The promotion record (`:501`) composes its `parameters` as `capturedSnapshot.adoptingSchedule(capturedSchedule).lineageValues()` and passes `capturedSchedule` as `schedule:`. `lineageRecordForSave` therefore takes the parameter snapshot as an explicit argument too: saves pass the in-force snapshot read right after the export, and the promotion passes the arena-start one.
       - **Training is not paused during the tournament** (review pass 4, P4-1). The arena pauses training only for the candidate snapshot (comment `App/SessionController+Arena.swift:126-133`, which says training "can continue through the full tournament"; pause `:134`–`:188`). On promotion it pauses both gates (`:423-425`) and rewinds weights, velocity and clock to the arena-start step S (`:455-467`, clock at `:463`). Settings are not rewound.
       - So a setting edited mid-arena is journalled at a step after S, a step the rewind discards. The rewound trainer uses it from step S+1, which can even come *before* the journalled step.
       - **Re-stamping at promotion.** Under the promotion's pauses, right after the rewind, every journal entry (`parameter_changes`, and gap 3.5 recapture entries if any) whose `trainer_step` is greater than S is re-stamped to S. Its original step is kept in a new field, `restamped_from`. The entries stay in commit order, so the last edit of a key still wins. The meaning is again "applies from the next step": S+1 in the rewound trainer.
       - **The promotion record** is built from the arena-start capture. It includes only the journal entries that existed at the arena-start capture (the length of each journal array, `parameter_changes` and `champion_changes`, is captured with the schedule and snapshot), so its `parameters` and `parameter_changes` agree and undo correctly to the segment start. Mid-arena edits appear in the trainer's later records, at S, with `restamped_from`.
       - **A kept arena** (no promotion): nothing is rewound, so nothing is re-stamped. Mid-arena edits keep their real steps.
     - **The inline post-promotion save** (review pass 3, P3-1).
       - The arena's `-promote` autosave writes `trainer.safetensors` itself, not through `saveSessionInternal`. Its trainer metadata schedule (`App/SessionController+Arena.swift:713-723`) is today built from the trainer's *current* `lrWarmupSteps` / `lrMomentumCycle`, and its lineage is `promotionLineage` (`:739`, passed at `:790`).
       - After N2, that record carries the arena-start schedule. An in-arena schedule edit would therefore make the flat `trainer_*` keys and the record disagree, and the P4 backstop would fail every such `-promote` save.
       - The fix: that metadata takes its schedule from the same arena-start capture (`capturedSchedule`, whose `completedTrainSteps` equals `trainerSnapshotCompletedSteps` because training was paused). The file's flat keys and its record then come from one value and describe the rewound state the file holds.
       - A later edit reaches files through the journal and the next save (gap 4). On a GUI resume of this file, `TrainerScheduleState.forSessionResume` uses the session's values and logs any difference (`Training/TrainerResumeState.swift:69-99`), as today.
     - Promote Trainee Now passes `TrainerScheduleState(currentlyRunningOn: trainer)` under its pause (`App/SessionController+ManualPromote.swift:169`).
     - The `[RUN]` start record and `lineageForResults` pass `TrainerScheduleState(currentlyRunningOn: trainer)` (`App/SessionController+Lineage.swift:241`, `:250`).
3. **Backstop (P4, not P2; review N1):** `SafetensorsModelIO.encode` (`Persistence/SafetensorsModelIO.swift:150-161`) refuses a trainer-state file whose lineage `parameters` is non-nil and whose schedule keys differ from `metadata.trainerSchedule` (`IOError.scheduleDisagreesWithLineage`).
   - Records with `parameters: nil` (`LineageRecord.forTests`, `DrewsChessMachineTests/LineageTestSupport.swift:13-27`; mints) are unaffected.
   - It lands in the **same commit as the GUI rule**. In P2/P3 the GUI still reads the singleton after the awaited export (`App/SessionController+Checkpoint.swift:445` → `:469`), so a guard in P2 would turn that race into failed session saves. P2 fixes only the CLI composition, which is correct by construction.
4. **Non-schedule keys in a GUI save** (review NB3). Static LR, weight decay and the other keys are read from the singleton after the awaited export (`:445` → `:469`). A popover commit in that window is recorded with its new value. That is consistent, not a contradiction:
   - the journal (gap 4) holds the change at the paused clock, i.e. "applies from the next step";
   - the trainer receives it before the next step.

   The snapshot is the *in-force set as of the save's clock + journal*. It is not claimed to be cut-consistent with the tensors for keys that do not travel in the trainer file.
5. Gap 13 is handled alongside.

**Tests** (new `DrewsChessMachineTests/ReplayResumeRecordedParametersTests.swift`; uses the static, internal `ResumeEquivalenceTests.writeCorpus(in:)`, `writeStartModel(in:)` and `architecture`, so no existing file is edited):
- `testAnExactResumeRecordsTheScheduleItTrainsUnder` (regression).
  - Segment 0: `runReplay` from `declaredDefaults(overriding:)` with `lr_warmup_steps 5` and `lr_cycle_period_steps 40`.
  - Segment 1: `--resume-exact` configured with 7 / 80.
  - Asserts segment 1's snapshot says 5 / 40 and equals the flat keys.
  - It uses only `ReplayParams.init(_:)` and `runReplay(config:params:abort:)`, which survive P2, so it compiles today and fails today.
- `testSnapshotAdoptingScheduleRoundTripsEveryField`: 200 seeded random schedules, envelope included; other keys unchanged.
- `testAdoptingScheduleKeepsAnOutOfRangeCheckpointValue`.
- P4: `testAGuiSaveRecordsTheExportedScheduleNotALaterEdit`. Using `GuiSaveHarness`, change the cycle in the singleton between the export and the record (the harness's pause hooks) and assert the record's schedule keys equal the export's.
- P4: `testAPromotionRecordsTheArenaStartScheduleNotAnEditDuringTheArena` (N2). Using `GuiSaveHarness`, start an arena, change the cycle in the singleton before promotion, promote, and assert the promotion record's schedule and parameters equal the arena-start capture, with the mid-arena edit absent from the promotion record and present in the next save's `parameter_changes` at step S with `restamped_from` equal to its original step (P4-1). It also asserts that the post-promotion session save **succeeds**, and that its `trainer.safetensors` flat `trainer_*` keys equal its record's schedule keys (P3-1). If the harness cannot drive an arena without Metal, the extracted functions that compose the promotion record and the post-promotion trainer metadata from the arena-start capture are tested directly, and the end-to-end case is validated in V3.
- P4 (moved from P2, N1): `testEncodeRefusesARecordWhoseScheduleDisagreesWithTheTrainerSchedule`.

**Validation.** V2 (see Validation): segment 1's snapshot schedule keys equal segment 0's, and match the flat `trainer_*` keys.

## Gap 6 — `git_dirty` is always true; uncommitted code has no identity (High)

**Problem.**
- `DrewsChessMachine/generate-build-info.sh:30` writes the tracked counter before the dirty test (`:36`). The tracked `App/BuildInfo.swift` is regenerated every build. The test runs at the repo root (`:18`).
- `LineageRecord.Build` (`Persistence/LineageRecord.swift:396-416`) records the flag. `ResumeGap.environmentGaps` compares it (`Training/ResumeExactness.swift:106-108`), so it feeds the `build` decision, which only the behavior-fingerprint check (`:110-139`) rescues.
- With no diff identity, the code constants are unrecoverable for any dirty build:
  - BN EMA 0.99 and ε 1e-5 (`Network/ChessNetwork.swift:2592-2593`, `:2626`);
  - `sqrtScaleBaseBatchSize = 4096` (`Training/ChessTrainer.swift:1230`);
  - `splitWorkingWeightSync` (`:1647`).

**How it misleads.** cont-step6093 says `git_hash 1ab52554, git_dirty true`. That would read the same for a pristine checkout.

**Decision.** Fix in code: the flag and the hash constant in P1; recording in P4.

**Design.**
1. P1, `generate-build-info.sh` (scope per review A1):
   - Compute `GIT_DIRTY` **before** writing the counter, over the compiled project only: `git -C "$REPO_ROOT" diff --quiet HEAD -- DrewsChessMachine ':(exclude)DrewsChessMachine/build_counter.txt' ':(exclude)DrewsChessMachine/DrewsChessMachine/App/BuildInfo.swift'`. `diff HEAD` covers staged and unstaged changes.
   - `DrewsChessMachine/` holds the Xcode project, the scheme, the test plan, the app and test sources, and the local packages.
   - Untracked, non-ignored files under the same path (`git ls-files --others --exclude-standard -- DrewsChessMachine`) also make it dirty. The project uses `PBXFileSystemSynchronizedRootGroup` (`DrewsChessMachine/DrewsChessMachine.xcodeproj/project.pbxproj:30-37`), so an untracked `.swift` file there is compiled.
   - `GIT_DIFF_SHA256` is the SHA-256 of `git diff --binary HEAD` over that pathspec, followed by each untracked file framed as `<path byte length>\n<path>\n<content byte length>\n<content>`, in sorted path order. It is empty when clean.
   - `BuildInfo` gains `static let gitDiffSHA256: String?` (`nil` exactly when `gitDirty == false`).
2. P1, optional (O-7): copy the same byte stream to `$BUILT_PRODUCTS_DIR/$CONTENTS_FOLDER_PATH/Resources/BuildDiff.patch`, a build-phase output with no runtime write.
3. P4: `LineageRecord.Build` gains `gitDiffSHA256: Recorded<String?>`.
   - Schema 3 requires it; value `null` is allowed only with `git_dirty false` (checked on decode and on `Build.current`).
   - Schema 2 decodes it as `.unrecorded`. The schema-2 `git_dirty` is kept as recorded, and P5 documents that it counted generated files.
   - `ResumeGap.environmentGaps` also compares the diff hash (`.unrecorded` counts as changed), still subject to the fingerprint escape.

**Tests.**
- `documentation/dashboards/tests/test_build_info_script.py` (new; kept in the one Python suite the repo runs, see review D3). Each test copies the script into a temporary git repo laid out like this one, commits, runs it and parses the generated Swift:
  - `test_a_clean_tree_is_not_dirty_after_the_counter_bump` (regression: `gitDirty = true` today);
  - `test_a_change_outside_DrewsChessMachine_is_not_dirty`;
  - `test_a_tracked_source_edit_is_dirty_and_hashed`;
  - `test_an_untracked_source_file_is_dirty_and_changes_the_hash`;
  - `test_the_generated_files_never_change_the_hash`;
  - `test_the_same_diff_hashes_the_same_twice`;
  - `test_framing_distinguishes_path_and_content_boundaries`.
- `DrewsChessMachineTests/BuildInfoConsistencyTests.swift` (new): `testDiffHashIsNilExactlyWhenClean`.
- P4, `DrewsChessMachineTests/LineageBuildRecordTests.swift` (new):
  - `testASchemaThreeBuildWithoutTheDiffKeyIsRefused`;
  - `testASchemaTwoBuildDecodesItsDiffAsUnrecorded`;
  - `testADifferentDiffHashIsABuildChangeUnlessTheFingerprintMatches`.

**Validation.** V1: after committing, a clean build gives `gitDirty == false` and `gitDiffSHA256 == nil`; editing a README under `experiments/` changes neither. Editing one Swift comment gives `true` and a 64-hex hash.

## Gap 7 — mixed corpora: only the first is named (Medium)

**Problem.**
- The runner feeds every corpus's sealed shards in argv order, but stores the first corpus's ID and path only (`CLI/CorpusReplayRunner.swift:1154-1158`) in `CorpusPosition` (`Persistence/LineageRecord.swift:264-296`).
- `shard_sha256` covers all shards, unlabelled.

**How it misleads.** A run over `A B` reads as a run over `A`, and `fed.corpus.shard` indexes a concatenation the file never names.

**Decision.** Fix in code (P4), **for provenance only**. Resume semantics are unchanged (review C3, modified):
- every shard's seal hash already covers its front header, which holds the corpus ID (`Persistence/GameCorpusShard.swift:92-102`, hash started over the header at `:320-322` and verified from the file start at `:492`);
- the ordered `shard_sha256` equality (`CLI/CorpusReplayRunner.swift:1258-1265`) therefore already pins every corpus's identity and order.

A name-list comparison would add nothing, so none is added. The existing first-ID check (`:1245`) stays as it is.

**Design.**
- `CorpusPosition` replaces `corpusID` / `corpusPath` with `corpus_identity` (S2). Swift's `enum CorpusIdentity { case listed([CorpusEntry]); case firstOnly(id: String, path: String) }` is the one stored value; there are no parallel fields (review A19).
- Encode and decode check that `Σ shard_count == shard_sha256.count` for `.listed`.
- Schema 2 decodes to `.firstOnly`.
- The runner builds the list in its existing loop (`:1154-1170`).
- `fed.corpus.segment_start` (review D7) records the `(epoch, next_game_index)` this segment's feed began at: the resolved start, after `--start-shard` / `--start-game-index` / resume.
- `SafetensorsModelIO.replayResumePoint` (`Persistence/SafetensorsModelIO.swift:502-526`) exposes `firstCorpusID`, derived from either case, for the unchanged check.
- `ckpt_inventory.py:71` reads both shapes.

**Tests** (new `DrewsChessMachineTests/MixedCorpusRecordTests.swift`):
- `testAReplayOverTwoCorporaRecordsBothInFeedOrder` (regression; two `writeCorpus(in:)` folders give two IDs).
- `testASchemaTwoCorpusPositionDecodesAsFirstOnly`.
- `testShardCountsMustSumToTheShardHashCount`.
- `testSegmentStartIsTheResolvedStartGameIndex`.

**Validation.** A V4 variant with `--replay-corpus X --replay-corpus Y`: `corpus_identity.listed` has two entries, and their shard counts sum to the `shard_sha256` length.

## Gap 8 — policy-tail precision missing from the record (Medium)

**Problem.**
- The process-wide precision (`Network/ChessNetwork.swift:3303-3321`) is written only as the flat `trainer_policy_tail_precision`, on trainer-state files (`Persistence/SafetensorsModelIO.swift:94`, `:166-167`; `Persistence/ModelCheckpointFile.swift:124-139`).
- Champion files and `SegmentSummary` carry none. The behavior fingerprint (`Training/BehaviorFingerprint.swift:63-86`) is a hash, not the value.

**Decision.** Fix in code (P4).

**Design.**
- `configuration.policy_tail_precision` is the process value, passed once per runner:
  - corpus replay `config.policyTailPrecision` (`CLI/CorpusReplayRunner.swift:145`);
  - train-vs-UCI `.process` (`CLI/TrainVsUciRunner.swift:225`);
  - GUI `trainer.policyTailPrecision` (`App/SessionController+Lineage.swift:207`).
- The encode-time agreement check is described in S3.
- Champion files get it through `configuration`.
- Resume keeps reading the flat key (`CLI/CorpusReplayRunner.swift:1253-1254`, `CLI/TrainVsUciRunner.swift:226-227`, `App/SessionController+Lineage.swift:282-283`).
- The `[RUN]` line (`Logging/RunProvenanceLine.swift`) gains `policy_tail=…` (review D4).

**Tests** (in `SegmentConfigurationRecordTests.swift`):
- `testAReplayRecordCarriesThePolicyTailPrecision` (regression).
- `testEncodeRefusesAPolicyTailThatDisagreesWithTheFlatKey`.
- `testAChampionFileRecordsThePromotedTrainersPolicyTail`.
- `PolicyTailPrecisionProvenanceTests` is unchanged: its fixture has `parameters: nil`, so `configuration` is nil.
- The `[RUN]` tests (`DrewsChessMachineTests/LineageProvenanceTests.swift:170-190`) check fragments, so the added fields need no test edits.

**Validation.** A V2 run with `--policy-tail-precision fp32_from_pre_bn`: `configuration.value.policy_tail_precision` equals the flat key; after the resume, segment 0's summary carries it.

## Gap 9 — seed settings contradict the run's seed (Low–Medium)

**Problem.**
- The snapshot includes the *settings* `random_seed_mode` and `random_seed` (`Training/TrainingParameters.swift:1337-1353`).
- The seed used is `rng.streams` (`Persistence/LineageRecord.swift:603-653`), and `--seed` is folded into `configured` (`Training/RandomSeedMode.swift:107-113`).
- Evidence: seg-resume-check (`--seed 777`) has snapshot `"0"/0` against streams `777 configured`; cont-step6093 has `"0"/0` against `18019510007828584227 drawn`.

**Decision.** Fix in code (P4); the alternative is O-9.

**Design.**
- `LineageRecord.Parameters.excludedParameterIDs = [RandomSeedModeParameter.id, RandomSeed.id]`, next to `Parameters.init(values:)` (`Persistence/LineageRecord.swift:357-375`). `init(values:)` throws if handed either key. No test passes them today: the five test call sites use only `learning_rate`, `training_batch_size`, `replay_ratio_auto_adjust`.
- `TrainingParametersSnapshot.lineageValues()` drops exactly those IDs; every composing site calls it.
- Decoding never checks this (S1).
- `RunStreams.SeedOrigin` gains `commandLine = "command_line"` (schema 3 only). Then:
  - `recordedOrigin` maps `.commandLine` to it (`Training/RandomSeedMode.swift:107-113`);
  - `effectiveMode` adds `.inherited(firstSegment: .commandLine)` → `.seeded` (`:98-103`), which the compiler requires once the case exists (review A9).
- `params_sha` on `[RUN]` (`Logging/RunProvenanceLine.swift:42`) changes value (R2).

**Tests** (new `DrewsChessMachineTests/LineageSeedRecordTests.swift`):
- `testTheLineageSnapshotOmitsTheSeedSettings` (regression).
- `testParametersInitRefusesTheSeedKeys`.
- `testACommandLineSeedIsRecordedAsCommandLine`.
- `testASchemaTwoSeedOriginStillDecodes`.
- `testAnInheritedSeedKeepsTheFirstSegmentsOriginAndMode` (covers `.commandLine` in both switches).

**Validation.** V2 with `--seed 777`: a composed snapshot of 83 keys, and `rng.streams.seed_origin "command_line"`.

## Gap 10 — keys that do nothing on a path look meaningful (Low; Low–Medium for train-vs-UCI)

**Problem.** The snapshot is all keys on every path, and nothing says which ones a path reads. Train-vs-UCI is worse:
- its snapshot's `self_play_*` keys look as if they apply;
- its `session.json` writes the self-play ply cap and τ (B1).

**Decision.** Document now (P5). The per-key `appliesTo:` attribute on `@TrainingParameter` is deferred (O-12): it touches every declaration and the macro, and `configuration.value.path_kind` (S2) already lets a reader apply the table.

**Applicability table (P5).** Built from each path's readers (`ReplayParams.init`, `CLI/CorpusReplayRunner.swift:41-57`; `TrainerHyperparameters.init`, `Training/TrainerHyperparameters.swift:64-86`; `ReplayBuffer.SamplingConstraints`; and each runner):
- **replay:**
  - the 18 Optimizer and 20 LR/Momentum Cycling keys;
  - `training_batch_size`, `replay_buffer_capacity`, `replay_buffer_min_positions_before_training` (`:1664`), `replay_ratio_target` (feed per step, `:1212`);
  - `max_plies_from_any_one_game`, `target_sampled_game_length_plies`, `max_draw_percent_per_batch`, `replay_buffer_stratify_by_material`;
  - `batch_stats_interval`, `kl_probe_interval`.
- **vsuci:** the same **except** `replay_ratio_target` (the driver passes `replayRatioTarget: nil`, `CLI/TrainVsUciRunner.swift:841`), plus `periodic_autosave_interval_sec` (read once at start, per CLAUDE.md). It reads nothing from Self-Play Sampling; its ply cap and move selection are `configuration.vsuci` (B1).
- **gui:** every key. Some act only under `--train` (`legal_mass_collapse_*`, `App/SessionController+Training.swift`).
- **all paths:** the seed keys are recorded in `rng.streams`, not the snapshot (gap 9).

The table is maintained by hand until O-12 is decided. P5's document says so.

## Gap 11 — budgets and train-vs-UCI opponents not recorded (Low / Medium)

**Problem.**
- **Budgets.**
  - Step and time limits from a `--parameters` file (`CLI/CliTrainingConfig.swift:27-34`) are applied (`App/DrewsChessMachineApp.swift:1114`, `:1417-1418`) but appear in no record.
  - `--epochs` is argv only.
  - When neither `--epochs` nor a step limit is given, the run enforces one epoch: `let epochLimit: Int? = config.epochs ?? (stepLimit == nil ? 1 : nil)` (`CLI/CorpusReplayRunner.swift:1218-1219`).
- **Opponents.** The spec (`CLI/TrainVsUciRunner.swift:15-26`) is argv only. Engine identity is never captured: `UCIArbiter.handshake` skips the `id` lines (`App/UCI/UCIArbiter.swift:168-176`).

**Decision.** Fix in code (P4).

**Design.**
- `configuration.budget = {training_step_limit, training_time_limit_sec, epoch_limit}` holds the **resolved** limits each runner enforces (review A6). `null` means the path enforces no such limit.
  - The corpus-replay resolution moves into one function, `CorpusReplayConfig.resolvedBudget`, which the runner (`:1218-1219`) and the record both call.
  - Train-vs-UCI uses `stepLimit` / `timeLimitSec` (`CLI/TrainVsUciRunner.swift:31-32`).
  - GUI `--train` uses its resolved limits (`App/DrewsChessMachineApp.swift:318-356`).
  - Interactive GUI: all `null`.
- `configuration.vsuci.opponents` (S2):
  - `executable_sha256` of the resolved executable, computed once at run start;
  - `id_name` / `id_author` from the first instance's handshake: `UCIArbiter.handshake()` returns `UCIEngineIdentity`, collecting `id name` / `id author` before `uciok`. A missing `id name` is recorded `null` and logged `[VS-UCI] engine … sent no id name`;
  - option values pass the same secret-marker redaction as argv (`Persistence/LineageRecord.swift:972-1011`).

**Tests.**
- `DrewsChessMachineTests/UCIEngineIdentityTests.swift` (new): `testHandshakeReturnsTheEnginesIdLines`, with a fake engine in the style of `DrewsChessMachineTests/UCIArbiterTests.swift:219-230`, which prints `id name Fake` / `id author test`. Also `testAnEngineWithoutIdNameRecordsNull`.
- In `SegmentConfigurationRecordTests.swift`:
  - `testAReplayRecordCarriesItsResolvedBudget` (regression);
  - `testAReplayWithNeitherEpochsNorAStepLimitRecordsEpochLimitOne`;
  - `testAStepLimitFromTheParametersFileIsRecorded`.
- `DrewsChessMachineTests/TrainVsUciLineageTests.swift` (new): `testAVsUciSaveRecordsItsGenerationConfig`. It uses the in-process setup of `TrainVsUciRefusalTests` if a save is reachable there; otherwise it is pinned at the record builder with a fake opponent list.

**Validation.**
- V2: `budget.training_step_limit 200`, `epoch_limit null`.
- V2b (segment 0 without a step limit, on a tiny corpus): `epoch_limit 1`.
- V5: `opponents[0].id_name` equals the engine's output, and `executable_sha256` equals `shasum -a 256 <binary>`.

## Gap 12 — effective LR/momentum at save not stored (Low)

**Problem.**
- The fed LR is the cycle value (or static LR) × sqrt-batch factor × warmup multiplier. Only the inputs are stored, so a reader must re-implement `LRMomentumCycle` math, and that copy can drift.
- The app itself already has two copies (review N6, verified):
  - `buildFeeds` (`Training/ChessTrainer.swift:6102`) computes the fed values on the host: warmup `:6154-6158`, base LR `:6173`, √batch `:6174-6182`, warmup applied `:6183`, momentum `:6203`;
  - `effectiveLearningRate` / `effectiveMomentum` (`:4437-4462`, `:4473-4479`) recompute them for the status bar and say they "mirror" `buildFeeds`.

**Decision.** Fix in code (P4), owner-gated twice:
- O-16: keep the field at all, since the value is derivable;
- O-18: route `buildFeeds` through the same function.

Recommended: both yes. That also removes the existing duplicate.

**Design.**
- **One static function,** `LRMomentumCycleReadout.values(schedule:staticLearningRate:staticMomentum:batchSize:sqrtBatchScaling:) -> (cycleStep, learningRate, momentum)`, in `Training/LRMomentumCycle.swift` beside the cycle math.
  - It evaluates the cycle **once** for both channels, as `buildFeeds` does (`Training/ChessTrainer.swift:6166-6172`, "One schedule evaluation feeds both channels").
  - Today the readouts call `.learningRate(…)` and `.momentum(…)` separately (`:4449`, `:4475`); the pinned grid proves they are unchanged by the switch (review pass 3, NB-b).
- **With O-18:** `buildFeeds`' host-side scalar computation (`:6154-6183`, `:6203`) and both readouts call it.
  - This is CPU arithmetic producing feed scalars, not graph-builder code. The feed tensors, placeholders and graph are unchanged.
  - The fields are `learning_rate_fed` / `momentum_fed`, and they are the fed values by construction (same function, same inputs).
- **Without O-18:** only the readouts call it; `buildFeeds` keeps its own copy. The fields are named `learning_rate_readout` / `momentum_readout`, and P5 documents them as the status-bar readout that mirrors `buildFeeds`.
- `configuration.schedule_at_save` = the function applied to the *exported* schedule and the record's own in-force static LR, momentum, batch size and √batch flag. It never reads live trainer properties.
- `null` for records without a trainer.

**Tests** (in `SegmentConfigurationRecordTests.swift`, plus `DrewsChessMachineTests/LRMomentumCycleReadoutTests.swift`, new):
- `testScheduleAtSaveMatchesTheTrainersReadout`.
- `testScheduleAtSaveIsNullWithoutATrainer`.
- `testReadoutRefactorKeepsEffectiveLearningRateBitIdentical`: a grid of steps × batch sizes × cycle on/off × warmup, pinned against values computed by the current functions *before* the refactor. The pins are generated once from the pre-refactor build and never modified.
- With O-18: `testBuildFeedsLearningRateAndMomentumAreBitIdenticalAfterTheRefactor`, the same pinned grid for the feed scalars (via the trainer's existing step-feed readback where available; otherwise the extracted pure function before/after). `ResumeEquivalenceTests` must pass unchanged (bit-exact resume end to end).

## Gap 13 — resume parameter changes leave no trace (Medium)

**Problem.**
- **CLI:** see Corrections. A changed weight decay is reported `[RESUME] EXACT`.
- **GUI** (review B7): the three "saved but not applied" values (batch size, promote threshold, arena games) take the current settings with only a `[RESUME-PARAM] … DIFFERS` log (`App/SessionParameterResume.swift:253-269`). `guiResumeGaps` adds no gap (`App/SessionController+Lineage.swift:270-300`).

**Decision.**
- P2: on the CLI paths, log one `[RESUME-DIFF] <id>: parent=… this_run=…` line per differing key, in the same format as the GUI's (`App/SessionParameterResume.swift:102`).
- Gap 1 makes every change visible across segments in the file.
- Whether a difference should be a `params` gap (on either path) is O-4. Default: log only.

**Design.**
- `TrainingParametersSnapshot.differences(fromLineage: LineageRecord.Parameters) throws -> [ParameterDifference]` decodes the parent's `snapshot_json` through each declaration's *type* (`K.decode`), **not** through `validate` (review NB1).
  - A parent value outside today's declared range (a range narrowed since) is reported as a difference, marked `out of today's range`, and never aborts the resume.
  - An ID only the parent has is reported `parent only`, never dropped.
  - It throws only when `snapshot_json` is not a JSON object, or a value has the wrong JSON type for its declaration. The record's sha256 check (`Persistence/LineageRecord.swift:377-386`) already guarantees the text is the one that was written.
- It is called where `parentRecord` is in scope (`CLI/CorpusReplayRunner.swift:1255`; `CLI/TrainVsUciRunner.swift:~250`), against the in-force snapshot *after* schedule adoption.
- Seed keys are compared only if the parent snapshot has them (schema 2), and are then reported informationally.

**Tests** (in `ReplayResumeRecordedParametersTests.swift`):
- `testDifferencesListsEveryChangedKeyAndNothingElse`.
- `testAnAdoptedScheduleIsNotADifference`.
- `testAParentOnlyKeyIsReportedNotDropped`.
- `testAnOutOfRangeParentValueIsReportedNotThrown` (NB1).

## B1 — train-vs-UCI game generation unrecorded, and `session.json` misrecords it (Medium–High)

**Problem.**
- The DCM side plays `.argmax` (`CLI/TrainVsUciRunner.swift:497`; `Network/MPSChessPlayer.swift:153-160`).
- `--max-plies` defaults to 400 and `--eval-sync-steps` to 10 (`App/DrewsChessMachineApp.swift:1253-1254`). Neither appears in argv when defaulted.
- The lineage snapshot shows self-play τ and ply-cap keys this path never reads.
- `session.json` writes `maxPliesPerGame: p.selfPlayMaxPliesPerGame` (`CLI/TrainVsUciSession.swift:196`) rather than the run's `config.maxPliesPerGame`, plus an unused self-play τ (`:141-144`).

**Decision.**
- Fix the `session.json` cap bug in P2 (no schema change).
- Record `configuration.vsuci` in P4.
- The self-play τ in a train-vs-UCI `session.json` stays: it is the GUI-format settings record. It is documented as not applying (P5); changing the session format is out of scope.

**Design.**
- P2: `TrainVsUciSession.sessionState` takes `maxPliesPerGame` from `TrainVsUciConfig` as a new required argument.
- P4: `configuration.vsuci` (S2). `trainer_move_selection` is read from the `SamplingSchedule` handed to the driver, the single source.

**Tests.**
- `DrewsChessMachineTests/TrainVsUciSessionStateTests.swift` (new): `testSessionStateRecordsTheRunsPlyCap` (regression).
  - It calls `sessionState(…, maxPliesPerGame: 123)` and asserts 123.
  - Written after P2's API-first sub-step (O-2): with the argument added but still ignored, it fails; after the fix it passes.
- `TrainVsUciLineageTests.testAVsUciSaveRecordsItsGenerationConfig` (gap 11).

**Validation.** V5: `configuration.vsuci.max_plies_per_game` equals `--max-plies` (400 when omitted), `trainer_move_selection` is the argmax schedule, and `session.json` `maxPliesPerGame` equals the same value.

## B2 — self-play Dirichlet noise (Medium, GUI)

**Problem.**
- α 0.3, ε 0.25, first 30 plies (`Network/MPSChessPlayer.swift:52-56`), used by every GUI self-play game (`App/SessionController.swift:1040`). It is not a parameter.
- The behavior fingerprint uses its own literal (`Training/BehaviorFingerprint.swift:209-210`), so a change to the app's constant moves neither the fingerprint nor, while gap 6 stands, any recorded build identity.

**Decision.** Fix in code (P4).

**Design.** `configuration.self_play_dirichlet`, read from `SamplingSchedule.selfPlay.dirichletNoise` at save. Non-null exactly when `configuration.value.path_kind == "gui"`.

**Tests.**
- `testAGuiRecordCarriesTheSelfPlayDirichletConfig` (`GuiSaveHarness`).
- `testACLIRecordHasNoSelfPlayDirichlet`.

## B3 — the init seed is lost across a branch (Medium)

**Problem.** `.branch` sets `initialization = nil` (`Persistence/LineageTracker.swift:181-183`). lrA (`20261005-23-TD5N`) branches from mint `20261005-22-yRzB` (`init_seed 20261005`, `dcm-init-1`), and records `init_seed: null`.

**Decision and design.** Fix in P4 via `AncestorRun.initialization` (gap 1b). The test is in gap 1b's list.

## B4 — what the `cum_*` totals mean (Medium)

**Problem.**
- The doc says `cum_trainer_step` is "Total trainer steps behind these weights" (`Persistence/LineageRecord.swift:177-179`).
- A branch restarts all totals at the new trainer's clock (`Persistence/LineageTracker.swift:181-187`), while a derive or graft continues the source's (`:386-401`). A branch from a 100k-step model reports e.g. `cum_trainer_step 3000`.

**Decision (O-15).**
- Recommended (a): correct the doc to "this run's totals". With `ancestry` (gap 1b), add a `scripts/dcm_lineage.py` helper `weights_totals(record)` that sums this run's totals with each `AncestorRun`'s, stopping at an unrecorded total. No stored copy is added.
- (b): store `steps.weights_cum_trainer_step` (and games/time) as `Recorded<Int?>`.

The plan implements (a) in P4 unless the owner picks (b).

**Tests.** `documentation/dashboards/tests/test_lineage_schema3.py::test_weights_totals_sum_along_ancestry_and_stop_at_unrecorded`.

## B5 — GUI promotion chain (Medium, GUI)

**Problem.** Self-play games come only from the current champion, which changes at each arena promotion or Promote Trainee Now. The trainer file records only `rng.streams.arenas_started` (`Persistence/LineageRecord.swift:626-628`). The history is in `session.json`'s `arenaHistory`, which is not in the model file.

**Decision.** Fix in code (P4), modified from the review.
- A new `noteChampionChange(championID:trainerCompletedSteps:trigger:)` is called next to the two existing promotion calls (`App/SessionController+Arena.swift:509`, `App/SessionController+ManualPromote.swift:175`).
- It does not go inside `recordPromotedChampionOrigin`, so `ChampionLineageRecordTests`' three direct calls (`DrewsChessMachineTests/ChampionLineageRecordTests.swift:99`, `:113`, `:117`) stay unedited.

**Design.**
- `configuration.champion_changes`, entries `{trainer_step, recorded_unix, champion_model_id, trigger: "arena" | "manual"}`.
- It is held by the segment journal (gap 4) and is `[]` on CLI paths.
- `trainer_step` is the clock the promoted weights carry: S, the arena-start step the trainer was rewound to (`trainerSnapshotCompletedSteps`), for an arena promotion; the trainer's clock under the pause for Promote Trainee Now. It is never the pre-rewind clock (P4-1).
- **Ordering (review NB7):** `noteChampionChange` runs *after* the promotion's lineage record has been built (`App/SessionController+Arena.swift:501-504` before `:509`; `App/SessionController+ManualPromote.swift:169-171` before `:175`). The champion's own origin record therefore never lists itself, and the trainer's next save is the first record with the entry.

**Tests.** `testAPromotionIsJournalledWithTheTrainerStep`, `testPromoteTraineeNowIsJournalledAsManual`, `testThePromotionRecordPrecedesItsChampionChangeEntry`.

## B6 — GUI auto replay-ratio: delays and effective target in force (Low–Medium, GUI)

**Problem.** With `replay_ratio_auto_adjust` on:
- the controller computes training-step and self-play delays and an integral-compensated effective target (`SessionController.effectiveReplayRatioTarget`, `App/SessionController.swift:448`; `ReplayRatioController.computedDelayMs`, `Training/ReplayRatioController.swift:792`; `smoothedSelfPlayDelayMs`, `:852`);
- they are written back to the singleton only when auto is switched off (`App/UpperContentView/ControlSideEffectsProbe.swift:202-248`).

**Decision.**
- Document in P5.
- Add the derived, never-read-back `configuration.replay_ratio_at_save` (GUI only) in P4.
- The measured average reuse stays derivable as `cum_trainer_step × batch / cum_positions`.

**Test.** `testAGuiRecordCarriesTheReplayRatioStateAtSave`.

## B8 — toolchain and build configuration (Low–Medium)

**Problem.**
- `LineageRecord.Build` has no Xcode, SDK or configuration fields (`Persistence/LineageRecord.swift:396-414`).
- The bf16 working-weight stomp is tied to a beta toolchain/OS (`Training/ChessTrainer.swift:1595-1646`).
- Frozen binaries are identified only by their path in argv (e.g. `FrozenBuilds/DCM-2320-1ab52554.app`).

**Decision.**
- P1: the script writes `BuildInfo.xcodeBuild`, `sdkBuild` and `configuration` from the build-setting environment variables `XCODE_PRODUCT_BUILD_VERSION`, `SDK_PRODUCT_BUILD_VERSION` and `CONFIGURATION`.
  - Not verified here: that the run-script phase sees these names. The script refuses to generate (non-zero exit, build fails) if any is empty, so a wrong name cannot silently produce blank fields.
- P4: record them as `Recorded<String>`.
- The executable's `LC_UUID` is O-17.

**Tests.** `test_build_info_script.py::test_toolchain_fields_are_written_and_empty_refuses` (the test sets the variables), and `LineageBuildRecordTests.testToolchainFieldsRoundTrip`.

## B9 — value-head recentering at load (Low)

**Problem.** Decoding an unmarked W/D/L head removes its shared offset (`Persistence/ValueHeadRecentering.swift:20-33`, report `:44-56`). The result is on `ModelCheckpointFile.valueHeadCentering` (`Persistence/ModelCheckpointFile.swift:315-317`) and logged under `[NUMERICS]`. The child's starting weights then differ from the bytes its `parent.content_sha256` names.

**Decision.** Fix in P4, modified from the review: the fact goes in the segment's `configuration.start_value_head_recentered` rather than in `parent`. `LineageTracker.ParentFile` (constructed in 23 test sites) is not touched.

**Design.** Value rule (review N5):
- **CLI.** The runner that loads `--start-model` has the decoded file's `valueHeadCentering`: `.recentered` → `true`; `.alreadyCentered` / `.notApplicable` → `false`. `.keptAsStored` is never used to train (`Persistence/ValueHeadRecentering.swift:30-32`), so it is an error here.
- **GUI branch from the champion.** The champion's start weights came from an *earlier* load, when the model or session was loaded, not from anything at segment start.
  - `SessionController.ChampionOrigin` (`App/SessionController+Lineage.swift:31-38`) changes its `.file` case to `case file(LineageTracker.ParentFile, startWeights: ChampionStartWeights)`, with `enum ChampionStartWeights { case loaded(ValueHeadCentering); case notLoaded }`.
  - The two load sites set `.loaded(file.valueHeadCentering)`: Load Model (`App/SessionController+Checkpoint.swift:811`) and Load Session (`:1002`). A missing `valueHeadCentering` on a decoded file is an error, never `.notLoaded`.
  - The promotion site (`App/SessionController+Lineage.swift:473`) sets `.notLoaded`: promoted weights come from the trainer, not from a load.
  - A branch records `true` for `.loaded(.recentered)`, and `false` for any other `.loaded` and for `.notLoaded`.
- **Built champion** (`.built`): `false`.
- **CLI fresh run** (no `--start-model`): `false`, as for a built GUI champion (review pass 3, P3-2).
- **Continue after Stop and New Session, keep trainer:** `null`, because the segment has no new start weights.
- A CLI `--resume-exact` loads its start file, so it records that file's value under the CLI rule above (review pass 4 nit).
- **GUI session resume** (trainer loaded from `trainer.safetensors`): the trainer file's own `valueHeadCentering`, as for the CLI.

**Tests.**
- `testARecenteredStartFileIsRecorded`, using an unmarked file built as `ValueHeadRecentering`'s existing tests build it.
- `testAGuiBranchFromARecenteredLoadedChampionRecordsTrue` (N5).
- `testAGuiBranchFromAPromotedChampionRecordsFalse`.
- `testAContinuedSegmentRecordsNull`.
- `testAFreshCLIRunRecordsFalse` (P3-2).

`ChampionLineageRecordTests` constructs `.file(…)` at `:50`, `:63`, `:81` and pattern-matches it at `:101`. These gain the `startWeights:` argument and binding; this is listed in O-14.

---

# Part P — Phasing

Each phase is its own commit: build, the phase's tests, commit (the owner's standing order for approved multi-phase plans). Builds go through drews-xcode-mcp only. No phase runs the app or tests while a training run is live.

### P1 — Build identity (gaps 6-flag, B8-constants)
- Touch:
  - `DrewsChessMachine/generate-build-info.sh`;
  - the generated `App/BuildInfo.swift`;
  - new `documentation/dashboards/tests/test_build_info_script.py`;
  - new `DrewsChessMachineTests/BuildInfoConsistencyTests.swift`;
  - `CHANGELOG.md`.
- Order: the Python regression test fails against the current script; fix; it passes.
- Lowest risk: no Swift logic changes.

### P2 — CLI resume records what it trains (gaps 5-CLI, 13; B1 `session.json` fix)
- Touch:
  - `Training/LRMomentumCycle.swift`;
  - `CLI/CorpusReplayRunner.swift` (`ReplayParams`, `:1046-1052`, `:1255`);
  - `CLI/TrainVsUciRunner.swift` (`:344`, `:~250`);
  - `CLI/TrainVsUciSession.swift` (`:196`);
  - new `ReplayResumeRecordedParametersTests.swift` and `TrainVsUciSessionStateTests.swift`;
  - the O-3 test edits;
  - `CHANGELOG.md`.
- **The schedule backstop in `SafetensorsModelIO.encode` is not in P2** (review N1). It lands in P4 together with the GUI composition rule, so no intermediate commit can fail a GUI session save. `testEncodeRefusesARecordWhoseScheduleDisagreesWithTheTrainerSchedule` moves to P4 with it.
- **Order** (review NB6; matches O-2):
  1. API-first: `TrainVsUciSession.sessionState` gains the required `maxPliesPerGame` argument, still ignored. Its only caller is `CLI/TrainVsUciRunner.swift:596`; there are no test callers.
  2. `testSessionStateRecordsTheRunsPlyCap` and `testAnExactResumeRecordsTheScheduleItTrainsUnder` are run and seen to fail.
  3. The fixes; both pass unmodified.
- No schema change.

### P3 — GUI values in force (gap 3)
- Touch:
  - new `Training/RunStartParameterCapture.swift`;
  - `App/SessionController.swift`;
  - `App/SessionController+Training.swift` (`:192-205`, `:675-676`, `:2021-2024`);
  - `App/SessionController+Lineage.swift:385`;
  - `App/SessionController+Checkpoint.swift:1219-1220`;
  - `App/SessionController+Heartbeat.swift:161-162`, `:582`;
  - `App/SessionController+Arena.swift:889`, `:975`;
  - `App/UpperContentView/TrainingSettingsPopoverModel.swift`, `TrainingSettingsPopover.swift`;
  - new `RunStartParameterCaptureTests.swift`;
  - the O-14 test-support edits;
  - `CHANGELOG.md`.
- Order: API-first, then regression tests (they fail), then switch the readers.
- No schema change.

### P4 — Lineage schema 3 (gaps 1, 1b, 4, 5-GUI, 6-record, 7, 8, 9, 11, 12; B1–B6, B8, B9)
One commit, because a schema-3 shape must never change after a build has written it. Internal order, each step building:

1. **API-first.**
   - Add the schema-3 types: `Recorded`, `SegmentSummary` fields, `TrainingConfiguration`, `AncestorRun`, `CorpusIdentity`, `Build` fields, `SeedOrigin.commandLine`.
   - Add the schema-aware decode, `lineageRecordForSave`'s `schedule:`, `UCIArbiter.handshake`'s identity return, and the required tracker/record arguments (forcing O-1/O-14).
   - `currentSchema` stays 2; nothing writes schema 3.
2. **Regression tests.** Add the regression tests (JSON-level); set `currentSchema = 3`; see them fail (values not yet filled).
3. **Recording.**
   - `LineageTracker` fills `configuration`, the summaries and `ancestry`.
   - `ParameterChangeJournal` + the `commitAssignment` hook (`Training/TrainingParameters.swift:1533-1657`, `:2449-2474`).
   - Recaptures (gap 3.5), champion changes.
   - The runners: corpora, `segment_start`, budget, `vsuci`, policy tail, Dirichlet, value-head recentering, `schedule_at_save` via the shared readout, `replay_ratio_at_save`.
   - `configuration.value.path_kind` and the invariants keyed to it (N4).
   - The promotion-time journal re-stamp, `restamped_from`, and the journal-clock encode check (P4-1).
   - The arena-start capture of schedule and in-force snapshot (N2), and the inline post-promotion save's trainer metadata built from it (`App/SessionController+Arena.swift:713-723`; P3-1).
   - `ChampionOrigin.file`'s `startWeights` (N5).
   - With O-18, `buildFeeds` calling the shared readout (N6).
   - The seed exclusion; the encode backstops (the schedule backstop moves here from P2, N1; the policy-tail backstop).
   - `withoutTrainerState()` and `untrainedCopyRecord` writing schema 3 from older sources.
   - `CLI/CliTrainingRecorder.swift`: `ResultsLineage` (`:306-331`) also encodes `configuration`, but not `segments` / `ancestry` / `derivation_history` (review A10).
   - The doc correction for `cum_*` (B4).

   Every regression test passes unmodified.
4. **Python.**
   - `scripts/dcm_lineage.py`: schema range, schema-3 required keys (`:119-132`, `:174-177`), the `weights_totals` helper.
   - `documentation/dashboards/ckpt_inventory.py`.
   - New `documentation/dashboards/tests/test_lineage_schema3.py`: `test_oldest_supported_schema_matches_the_app`, `test_schema_three_summary_keys_are_required`, `test_weights_totals_sum_along_ancestry_and_stop_at_unrecorded`.
   - The existing `test_lineage.py` is unchanged.
5. **Fixtures.** `DrewsChessMachineTests/LineageSchemaTwoFixtures.swift` holds the two real `dcm_lineage` texts (cont-step6093, seg-resume-check seg1) as Swift raw string literals, copied byte for byte and never edited. `testFixtureLiteralsAreUnedited` pins each literal's SHA-256. The test target has no resource-bundle mechanism today (review A16).

Also in P4: the `LineageRecord.swift` file comment (house style: multi-paragraph *why*), the CLAUDE.md "File lineage" paragraph (with the owner's approval) and the `[RUN]` line's `policy_tail=` / `git_diff=<12>` fields.

### P5 — Documentation (gaps 10, B6, B1's `session.json` τ note, schema-2 `git_dirty` meaning)
- `documentation/lineage-record.md`: a schema-3 field reference with schema-2 differences and the applicability table. Created only with the owner's approval (CLAUDE.md: don't silently invent a new markdown doc); otherwise the content goes into CLAUDE.md "File lineage".

### P6 — Optional: pre-lineage reconstruction report (gap 2)
- Only if O-8 is approved. Python only.

---

# Part V — Validation

Every end-to-end run below needs an idle GPU: confirm no live training session first (owner rule). Commands use the built binary `$BIN`, never `xcodebuild`. `$S` is a scratch folder.

**Corpus and parameter files for V2/V4.** The real corpus `20260624-192615-w3aA5b`. A.json is from `--create-parameters-file` into `$S`, then edited:
- `training_batch_size 256`, `replay_buffer_capacity 20000`, `replay_buffer_min_positions_before_training 5000`;
- `lr_warmup_steps 20`, `lr_cycle_enabled true`, `lr_cycle_period_steps 100`.

B.json differs only in `weight_decay 0.0005`, `lr_warmup_steps 40` and `lr_cycle_period_steps 200`.

**V1 (P1).** See gap 6. Also confirm that `git status` after a build still shows `build_counter.txt` and `BuildInfo.swift` modified, while `BuildInfo.gitDirty == false` on an otherwise clean `DrewsChessMachine/`.

**V2 (P2; repeated after P4).** The second command starts from the step-enumerated file, never from the rolling file it writes. The rolling file then holds the start model's `model_id` at its `training_step`, so replacing it is permitted (review A13).

```
$BIN --replay-corpus 20260624-192615-w3aA5b --parameters $S/A.json --seed 777 --training-step-limit 200 \
     --out-model $S/v2-replay-latest.safetensors --enumerate-checkpoints
$BIN --replay-corpus 20260624-192615-w3aA5b --parameters $S/B.json --seed 777 \
     --start-model $S/v2-replay-step200.safetensors --resume-exact --training-step-limit 200 \
     --out-model $S/v2-replay-latest.safetensors --enumerate-checkpoints
```

Expected in `$S/v2-replay-seg1-step200.safetensors` (dumper above):
- snapshot `lr_warmup_steps 20` and `lr_cycle_period_steps 100` (adopted), `weight_decay 0.0005` (B);
- flat `trainer_lr_warmup_steps 20`, and `trainer_lr_momentum_cycle` containing `"lrPeriodSteps":100`;
- log lines `[REPLAY-RESUME] WARNING lr_warmup_steps … restoring the checkpoint's value` and `[RESUME-DIFF] weight_decay: parent=0.0003 this_run=0.0005`.

After P4, additionally:
- `schema 3`;
- `configuration.value.policy_tail_precision` equals the flat key;
- `budget = {training_step_limit: 200, training_time_limit_sec: null, epoch_limit: null}`;
- `segments[0].parameters.value.sha256` equals the step-200 file's `parameters.sha256`;
- `rng.streams.seed_origin "command_line"`, and the composed snapshot has 83 keys;
- `fed.corpus.corpus_identity.listed` has one entry, and `segment_start` equals the step-200 file's `next_game_index` / `epoch`.

**V2b (P4).** `budget.epoch_limit 1` when neither `--epochs` nor a step limit is given. This is checked by the XCTest `testAReplayWithNeitherEpochsNorAStepLimitRecordsEpochLimitOne`, on the synthetic corpus in a temporary folder, not end to end (review NB5). `--import-pgn` has no output-folder flag (`CLI/PGNImporter.swift:21`, `:92`; `outputParentDirectory` is set by no CLI argument), so an import would write into the production `Corpora/` folder, and a real-corpus single-epoch run is far too long for a validation step.

**V3 (P3, P4; GUI).** See gaps 3 and 4. Then resume the saved session and save again: `segments[0].configuration.value.parameter_changes` holds the first segment's LR change, and the new segment's is `[]`. After an arena promotion, `champion_changes` has one `"arena"` entry.

**V4 (P4; branch, derive, mixed corpora).**
- `--start-model $S/v2-replay-seg1-step200.safetensors` **without** `--resume-exact` (new `--out-model`): `ancestry[0].segments` has two summaries, and `ancestry[0].initialization.value` equals the V2 files' `rng.init_seed` / `init_scheme`. V2's segment 0 had no `--start-model`, so it is a fresh run whose init seed is recorded and carried by the exact resume (`Persistence/LineageTracker.swift:193`) (review N3).
- A branch from a mint (Build/`--new-model` with a seed): `ancestry[0].initialization.value.init_seed` equals the mint's.
- `--derive-model --from <that file> <an operation valid for its arch> --out $S/d.safetensors`: `ancestry` grows, and `derivation_history` gains one record.
- `--replay-corpus X --replay-corpus Y`: `corpus_identity.listed` has two entries.

**V5 (P4; train-vs-UCI, owner machine with an engine).** `--train-vs-uci "cmd=<engine>;n=1;go=nodes 1" --training-step-limit 20`. Check `configuration.value.vsuci` (B1, gap 11), and that `session.json` `maxPliesPerGame` is 400.

**Old files still decode (all phases).** With the new build:
- `--analyze-numerics <file> --numerics-static-only` (read-only; writes its report under the analyses folder) succeeds on cont-step6093, `seg-resume-check-replay-seg1-step1000`, `b2275-step33000` and one v3 file.
- `scripts/dcm_lineage.py` reads every lineage file and derives the same runs as before.
- **Baseline for an exact resume of a schema-2 file** (review A14; build 2320 never resumed cont-step6093, so there is no earlier token set to compare with): a `--resume-exact` of cont-step6093 (`--policy-tail-precision fp32_from_pre_bn --epochs 12 --training-step-limit 10`, out-model in `$S`) must log:
  - `[RESUME] build changed (2320 1ab52554+dirty → …), behavior fingerprint matches|differs`;
  - then `[RESUME] EXACT` when the fingerprint matches, or `[RESUME] NOT EXACT: build` (refused unless `--accept-inexact build`) when it differs (`Training/ResumeExactness.swift:106-139`).

  No other token may appear. This resume writes schema 3, and its new segment's summary of the cont segment carries `parameters` recorded and `configuration` unrecorded.

**No change to training math.**
- **Two builds, same run:** the pre-phase and post-phase builds each run V2's segment 0 with `--seed 777`.
  - Where the `ResumeEquivalenceTests` determinism probe reports `bitExact`, the two files' `content_sha256` (tensor data only, `Persistence/SafetensorsFile.swift:27-29`) must be equal.
  - Otherwise, compare per tensor with `scripts/safetensors_tensor_hash.py` within the probe's tolerance.
- **Existing suites that must pass unchanged** (targeted runs; full suite once before merging, per CLAUDE.md):
  - `ResumeEquivalenceTests`, `ExactResumeTests`, `ExactResumeCompletionTests`, `CheckpointManagerSafetensorsTests`, `TrainerHyperparametersTests`;
  - `LineageRecordTests`, `LineageProvenanceTests`, `ChampionLineageRecordTests`, `UntrainedCopyRecordTests`, `DeriveLineageTrainedSourceTests`, `PolicyTailPrecisionProvenanceTests`;
  - `GuiResumeGapsTests`, `GuiLineageLifecycleTests`, `SessionSaveConsistentCutTests`, `LineageFedCountsTests`, `SessionParameterResumeTests`;
  - `TrainingParametersTests`, `TrainingParametersRunHoldTests`, `RunSeedParameterTests`, `UCIArbiterTests`, `BehaviorFingerprintTests`;
  - the Python suites in `documentation/dashboards/tests/`.

  Tests that need edits are only those listed in O-1, O-3 and O-14.
- **No graph-builder, trainer-step or sampler edits in any phase.** Each phase's diff is reviewed for changes under `Network/ChessNetwork.swift` graph code, `Training/ChessTrainer.swift` step code and `Training/ReplayBuffer.swift` sampling. The allowed trainer edits are gap 12's readout refactor of `effectiveLearningRate` / `effectiveMomentum` and, only with O-18, `buildFeeds`' host-side LR/momentum scalars calling the same function. Both are guarded by pinned bit-identical tests, and the latter also by `ResumeEquivalenceTests`.

---

# Owner decisions needed

- **O-1 Mechanical test call-site edits (P4, required).** New required arguments with no defaults (no silent defaults). Only arguments are added; no assertion changes.
  - `.record(` on a tracker: `LineageRecordTests.swift` (10), `ExactResumeCompletionTests.swift` (3), `LineageProvenanceTests.swift` (3), `BehaviorFingerprintTests.swift` (1), `LineageTestSupport.swift` (1), `GuiLineageLifecycleTests.swift` (1), `GuiResumeGapsTests.swift:38`, `GuiResumeContinuationGapsTests.swift:35`, `DropoutRNGStateTests.swift:179`, `TrainVsUciSessionTests.swift:180`, `SegmentIndexedCheckpointNamingTests.swift:88`;
  - `LineageRecord(`: `ChampionLineageRecordTests.swift` (6), `InitSeedRecordingTests.swift` (2), `LineageRecordTests.swift` (1), `LineageTestSupport.swift` (1);
  - `CorpusPosition(`: `ExactResumeTests.swift` (2), `LineageRecordTests.swift` (2);
  - `Build(buildNumber:`: `BehaviorFingerprintTests.swift`, `ExactResumeCompletionTests.swift`, `LineageTestSupport.swift` (1 each).

  Counts are from `grep` at `a7d3b9ca`, including calls whose arguments start on the next line. They are re-counted with a multi-line pattern (`record\(\s*\n?\s*at:`) before P4.
- **O-2 Regression-test sequencing.** In P2 (B1), P3 and P4, regression tests are written after an API-first step so they compile. They fail at that point and pass after the fix, unmodified. Is that acceptable as "write the test first"?
- **O-3 `ReplayParams` becomes immutable and derived (P2).**
  - `ResumeEquivalenceTests.swift:196-203`, `CorpusReplayRefusalTests.swift` and `FinalTrainerSaveFailureTests.swift` build their snapshot with `declaredDefaults(overriding:)` instead of mutating fields afterwards.
  - One value changes: `CorpusReplayRefusalTests.testAnExactResumeRefusedForAGapThrowsNamingTheGaps` passes batch 16 (`:116`), below the declared range `32...65536` (`Training/TrainingParameters.swift:757`). It becomes 64, still different from the first run's 32, so the feed per step still changes; the assertions stay the same.
  - All other overrides are in range.
  - If declined, P2 keeps `var`s and drops the schedule backstop (those tests would trip it). The adoption fix and its regression test stand either way.
- **O-4 Resume parameter changes as gaps.** Should a changed training-math key on any exact resume be a `params` gap: a CLI `--resume-exact`, or a GUI resume where a saved value is replaced by the current setting (`App/SessionParameterResume.swift:253-269`)? Default: log only, on both paths.
- **O-5 Gap 3.** Record the in-force value (planned). Also disable the three fields during a run?
- **O-6 Gap 8.** Should summaries of schema-2 parents carry the parent file's flat `trainer_policy_tail_precision`? It is a recorded value, but carrying it means adding it to `LineageTracker.ParentFile` (constructed in 23 test sites). Default: unrecorded.
- **O-7 Gap 6.**
  - Embed `BuildDiff.patch` in the app bundle?
  - Widen the dirty/diff scope beyond `DrewsChessMachine/` (e.g. `scripts/`)? Default: no; `documentation/` and `experiments/` never count.
- **O-8 Gap 2.** Build the reconstruction report (P6)?
- **O-9 Gap 9.** Exclude the seed settings from composed snapshots (planned), or keep them and document that `rng.streams` is the authority?
- **O-10 Journal granularity (gap 4).** The Replay-tab fields propagate live as the user types (`App/UpperContentView/TrainingSettingsPopoverModel.swift:1587-1590`), so each valid intermediate value is a real in-force change and is journalled. Keep (honest; several entries per edit), or journal at popover Save only (fewer entries, but an in-force interval goes unrecorded)?
- **O-11 ROADMAP.** Add this plan to ROADMAP.md (needs permission per the global rules)?
- **O-12 Gap 10.** Add `appliesTo:` to `@TrainingParameter` later?
- **O-13 Opponent executable.** `TrainVsUciOpponentSpec.command` is documented as an executable path (`CLI/TrainVsUciRunner.swift:16-17`). If a bare name resolved through `PATH` is ever allowed, record and hash the resolved path.
- **O-14 GUI test-support edits (P3/P4, required).** Only setup and arguments are added; no assertion changes.
  - `DrewsChessMachineTests/GuiSaveHarness.swift:119` and `GuiLineageLifecycleTests.swift:109`, `:132`, `:167`: a `controller.beginRunStartCapture(buffer:)` call next to each tracker assignment.
  - `LineageFedCountsTests.swift:70`, `:97`: the new `schedule:` and parameter-snapshot arguments of `lineageRecordForSave` (P4; with review N1's resolution the GUI rule and this edit are both in P4).
  - `ChampionLineageRecordTests.swift:50`, `:63`, `:81` (constructions of `ChampionOrigin.file`) and `:101` (pattern match): the new `startWeights:` associated value (N5, P4).
- **O-15 `cum_*` meaning (B4).** (a) Correct the doc to "this run's totals" + an ancestry-summing helper (recommended, planned), or (b) store weights-totals.
- **O-16 Gap 12.** Keep `schedule_at_save` (with the readout refactor) or defer it? It is derivable from the file.
- **O-17 B8.** Also record the executable's `LC_UUID` (read at runtime via `_dyld_get_image_header(0)`)?
- **O-18 One implementation of the fed LR/momentum (N6).** Allow `buildFeeds`' host-side LR/momentum scalar computation (`Training/ChessTrainer.swift:6154-6183`, `:6203`) to call the shared `LRMomentumCycleReadout` function that the status-bar readouts also call?
  - It is guarded by a pinned bit-identical test and `ResumeEquivalenceTests`.
  - It is the one trainer step-path edit in this plan, and it removes an existing duplicate.
  - If declined, `schedule_at_save`'s fields are named `learning_rate_readout` / `momentum_readout`.

# Risks

- **R1 Older builds cannot read schema-3 files.**
  - `FrozenBuilds/` binaries and any pre-P4 build refuse schema-3 models and `session.json` (`Persistence/LineageRecord.swift:835`).
  - Live runs on an old build keep writing schema 2, which the new build reads.
  - The P4 CHANGELOG entry names the first build that writes schema 3.
- **R2 `params_sha` changes.** Removing the seed keys changes `[RUN] params_sha=` for identical settings. No code compares parameter hashes across files (only `parameters == nil` checks at `App/SessionController+Lineage.swift:292`, `CLI/CorpusReplayRunner.swift:1286`, `CLI/TrainVsUciRunner.swift:254`), and the Python tools do not read `parameters`.
- **R3 The `commitAssignment` hook touches every stored property's `didSet`** (`Training/TrainingParameters.swift:1533-1657`). The edit is mechanical. It is covered by `TrainingParametersTests`, `TrainingParametersRunHoldTests` and the new journal tests.
- **R4 Header growth.** About 3.6 KB per summary, never truncated. The 64 MB header guard (`Persistence/ModelFileCatalog.swift:252`) is the backstop. A long line's header size is measured before merge.
- **R5 Encode guards can halt a save.** The schedule and policy-tail checks throw from `SafetensorsModelIO.encode`.
  - With gap 5's rule (lineage schedule composed from the same export the flat keys come from), no production writer can trip the schedule check; it only catches a writer that bypasses the rule. `testAGuiSaveRecordsTheExportedScheduleNotALaterEdit` and `testAPromotionRecordsTheArenaStartScheduleNotAnEditDuringTheArena` pin the GUI race and arena cases.
  - The schedule check ships in the same commit as the GUI rule (P4), never before it (N1).
  - Corpus replay tolerates one failed save and halts on the second (`CLI/CorpusReplayRunner.swift:164-206`).
- **R6 `git_dirty` meaning shifts at P1.** Schema-2 records written after P1 carry the corrected flag; earlier ones the always-true flag. The build number tells them apart (the P1 CHANGELOG names it).
- **R7 Journal volume.** Keystroke-level live edits (O-10) add entries. They are never truncated; R4's measurement covers them.

# Non-goals

- Rewriting, re-saving or annotating any existing model or session file, or migrating schema-2 records in place. A *new* write is always schema 3 (S1).
- Recording code constants that change the *training math* as parameters: BN momentum and ε, the √batch base, the optimizer form, `splitWorkingWeightSync`. They are covered by the build identity (`git_hash` + `git_diff_sha256` + toolchain) and, for numerics, by the behavior fingerprint.
  - The self-play Dirichlet noise is the exception: it shapes the *data*, the fingerprint does not cover it, and it is recorded (B2).
- Turning `--policy-tail-precision` into a `@TrainingParameter`. It stays process-wide by design (`App/DrewsChessMachineApp.swift:128-133`).
- Forward compatibility (older builds reading schema 3).
- Changing training math, resume refusal semantics (unless O-4 says so), sampling, or the `[STATS]` / `results.json` formats beyond carrying the record they already carry (`results.json` gains `configuration`, A10).
- Changing the train-vs-UCI `session.json` format beyond the B1 ply-cap fix.
- A full time series of effective LR/momentum. `schedule_at_save` is one derived point per file; the series stays in `[STATS]` / `results.json`.

---

# Review reconciliation

An independent reviewer audited the question from scratch and then reviewed this plan. Every item is listed below. "Accepted" means folded in as proposed; "modified" means folded in with the change noted; "rejected" gives the reason.

| Item | Verdict | Where / why |
|---|---|---|
| A1 dirty scope is the repo root | Accepted | Gap 6 design 1; O-7 rewritten. Verified `REPO_ROOT="$SCRIPT_DIR/.."` (`generate-build-info.sh:18`) and the synchronized root group (`project.pbxproj:30-37`). |
| A2 GUI save can race the schedule guard | Accepted | Gap 5 design 2 (one rule; GUI `schedule:` argument), gap 12 (shared readout), R5. Verified the awaits (`App/SessionController+Checkpoint.swift:432-469`) and the ungated `apply(to:)` (`TrainingSettingsPopoverModel.swift:1584`). |
| A3 captured keys change inside a segment | Accepted | Gap 3 point 5; tests renamed. Verified the kept tracker for New Session, keep trainer (`App/SessionController+Lineage.swift:182-185`) and the re-read at every start. |
| A4 ordering claim wrong; hook placement | Accepted | Gap 4 design 1 and 3, tests. Verified `releaseRunHolds` (`App/SessionController+Training.swift:38-45`; `Training/TrainingParameters.swift:2310-2319`) and the early return (`:2471`). |
| A5 P3 breaks harness-based tests | Accepted | P3 order; O-14. Verified `GuiSaveHarness.swift:119`, `GuiLineageLifecycleTests.swift:109`, `:132`, `:167`. |
| A6 wrong epoch budget | Accepted | Gap 11 (`resolvedBudget`, `epoch_limit`); V2/V2b. Verified `CLI/CorpusReplayRunner.swift:1218-1219`. |
| A7 per-category table wrong | Modified | Gap 10 table rebuilt per key. Also corrected the review's own list: train-vs-UCI does **not** read `replay_ratio_target` (`CLI/TrainVsUciRunner.swift:841` passes nil). The proposed doc-generator test is replaced by "maintained by hand until O-12". |
| A8 old-schema records rewritten | Accepted | S1 (every write is schema 3; three-state `configuration`; carried snapshots unchecked), S2 invariant, gap 1/1b tests. Verified `App/SessionController+Lineage.swift:440-441`, `Persistence/LineageRecord.swift:899-913`. |
| A9 seed-mode switch | Accepted | Gap 9 design. Verified `Training/RandomSeedMode.swift:98-113`. |
| A10 `results.json` lineage keys hand-listed | Accepted | P4 step 3. Verified `CLI/CliTrainingRecorder.swift:306-331`. |
| A11 O-1 incomplete | Accepted | O-1 list extended. Verified each of the five sites. |
| A12 O-3 not purely mechanical | Accepted | O-3. Verified `CorpusReplayRefusalTests.swift:116` (batch 16) against range `32...65536`. |
| A13 V2 refused at launch | Accepted | V2 commands. |
| A14 no baseline for the cont-step6093 resume | Accepted | "Old files still decode" baseline. |
| A15 tests would write real settings | Accepted | Gap 3/4 test notes. Verified the persistence path (`Training/TrainingParameters.swift:2471-2472`) and the pattern (`TrainerHyperparametersTests.swift:31-41`). |
| A16 no fixture mechanism | Accepted | P4 step 5 (raw string literals + SHA pin). |
| A17 stale census | Accepted | Census recounted 2026-10-05 03:15 CDT: 4,334 files, v8 = 23. |
| A18 Dirichlet not covered by the fingerprint | Accepted | Non-goals; B2. Verified `Training/BehaviorFingerprint.swift:209-210`, `Network/MPSChessPlayer.swift:52-56`. |
| A19 `corpora` + `first_corpus` double copy | Accepted | One `corpus_identity` (S2, gap 7). |
| A20 line-level corrections | Mostly accepted | Flag declarations (`:2270`, `:2276`, `:2291`), GUI tail-gap line `:282-283` and the arena/stats wording are applied. **Rejected:** the label-smoothing `didSet` range "`:1542-1548`". The property spans `:1541-1547` (declaration at 1541, `didSet` 1542-1546, closing brace 1547); line 1548 is the next property. |
| B1 train-vs-UCI generation unrecorded, `session.json` bug | Accepted | New section B1; P2 fix; S2 `configuration.vsuci` replaces top-level `opponents`. Verified `CLI/TrainVsUciRunner.swift:497`, `App/DrewsChessMachineApp.swift:1253-1254`, `CLI/TrainVsUciSession.swift:141-144`, `:196`. |
| B2 self-play Dirichlet | Accepted | Section B2. |
| B3 init seed lost on branch | Accepted | `AncestorRun.initialization`. Verified on lrA vs r7b24-fresh headers. |
| B4 totals meaning | Accepted as owner decision | O-15, recommendation (a). |
| B5 promotion chain | Modified | Journalled via a new `noteChampionChange` at the two call sites instead of inside `recordPromotedChampionOrigin`, so `ChampionLineageRecordTests` (`:99`, `:113`, `:117`) stays unedited. |
| B6 auto replay-ratio state | Accepted | Section B6 (doc + derived GUI field). |
| B7 GUI resume parameter replacement | Accepted | Folded into gap 13 and O-4. Verified `App/SessionParameterResume.swift:253-269`. |
| B8 toolchain identity | Accepted | Section B8; `LC_UUID` is O-17. Env-var names unverified; the script refuses empty values. |
| B9 value-head recentering at load | Modified | Recorded as `configuration.start_value_head_recentered`, not as a `parent` field: the loading runner has the value, and `ParentFile` (23 test constructions) stays untouched. |
| C1 gap 1b Critical | Accepted | Summary, gap 1b. |
| C2 gap 10 Low–Medium for train-vs-UCI | Accepted | Summary, gap 10. |
| C3 gap 7's new refusal | Modified (no refusal added; no O-16 needed for it) | The review's scenario (identical shards re-imported under another ID) cannot occur: each shard's seal hash covers its front header, which holds the corpus ID (`Persistence/GameCorpusShard.swift:92-102`, `:320-322`, `:492`). The ordered hash check already pins every corpus. A name check adds nothing, so none is added and resume semantics are unchanged. |
| C4 gap 12 optional | Accepted | O-16; built on A2's single function if kept. |
| C5 R5 understated | Accepted | R5 rewritten. |
| C6 O-4 covers the GUI | Accepted | O-4. |
| D1 `adoptingSchedule` must not clamp | Accepted | Gap 5 design 1, test. |
| D2 capture lifetime through Stop | Accepted | Gap 3 design 2. |
| D3 build-script test beside the script | Rejected | `documentation/dashboards/tests/` is the repo's one discovered Python suite, and it already tests `scripts/` modules (`test_tooling.py` header: "Run: python3 -m unittest discover -s documentation/dashboards/tests"). A second location would be a second test command nobody runs. |
| D4 `[RUN]` fields | Accepted | Gap 8; P4. The `[RUN]` tests check fragments, so no edits are needed. |
| D5 stats/chart sample | Accepted | Gap 3 design 3. |
| D6 Python per-summary required keys | Accepted | P4 step 4. |
| D7 segment start corpus position | Accepted | `fed.corpus.segment_start`, summaries' `segment_start_corpus`. |

## Pass 2 (N1–N6, NB1–NB7)

The reviewer agreed with every rejected or modified item from pass 1 (A20, D3, C3, A7, B5, B9).

| Item | Verdict | Where / why |
|---|---|---|
| N1 backstop in P2 before the GUI rule | Accepted (option a) | The schedule backstop and its test move to P4, in the same commit as the GUI rule (gap 5 design 3, P2 and P4 blocks, R5). Verified the GUI still reads the singleton after the awaited export (`App/SessionController+Checkpoint.swift:445` → `:469`). |
| N2 arena promotion has no captured schedule | Accepted | Gap 5 design 2 (arena-start capture of schedule and in-force snapshot; `lineageRecordForSave` takes the snapshot explicitly); new test. Verified the tuple has no schedule (`App/SessionController+Arena.swift:170-181`) and the rewind restores weights, velocity, clock and dropout only (`:455-467`). |
| N3 null `initialization` ambiguous; V4 wrong | Accepted, refined | S2 table. Refinement: the ancestor run's *first-segment* start (`segments.first?.start ?? run.start`) decides, because a resumed segment's own `run.start` is `resume` even when the run began as a branch. A derive of a pre-lineage file (`continues_unrecorded_history true`, `Persistence/LineageTracker.swift:381`) is still `recorded(null)`. V4 corrected (V2's segment 0 is fresh). Three new tests. Verified the doc (`Persistence/LineageRecord.swift:508-516`) and the resume carry (`Persistence/LineageTracker.swift:193`). |
| N4 derive of a schema-3 source; path-kind invariants | Accepted | `configuration.value.path_kind`; invariants keyed to it (S2); verbatim carry (S1); two tests. Verified the copy writes its own `pathKind` (`Persistence/LineageTracker.swift:405`). |
| N5 GUI branch from a loaded champion records null | Accepted | B9 design: `ChampionOrigin.file(_, startWeights:)` set at the load sites (`App/SessionController+Checkpoint.swift:811`, `:1002`) and the promotion site (`App/SessionController+Lineage.swift:473`). Adds `ChampionLineageRecordTests` call-site edits to O-14 (`:50`, `:63`, `:81`, `:101`). |
| N6 two implementations of the fed LR | Accepted (option a, owner-gated) | S3 row, gap 12, O-18. Verified `buildFeeds` (`Training/ChessTrainer.swift:6102`, `:6154-6183`, `:6203`) against the readouts (`:4437-4462`, `:4473-4479`). Option (b) (readout names) applies if O-18 is declined. |
| NB1 out-of-range parent values must not throw | Accepted | Gap 13 design (decode by type, not `validate`); new test. |
| NB2 seed edits in the journal | Accepted | Gap 4 design 2: `excludedParameterIDs` are skipped; new test. |
| NB3 non-schedule keys can race in a GUI save | Accepted (documented, no code) | Gap 5 design 4: the journal entry at the paused clock makes the record consistent; the snapshot is not claimed cut-consistent for those keys. |
| NB4 observer closure type and captures | Accepted | Gap 4 design 2 (`@Sendable`, weak captures, main-actor install/remove); new test. |
| NB5 V2b would write into production `Corpora/` | Accepted | Verified that `--import-pgn` has no output-folder flag (`CLI/PGNImporter.swift:21`, `:92`). V2b is now the synthetic-corpus XCTest only. |
| NB6 P2 order | Accepted | P2 block. |
| NB7 B5 ordering | Accepted | B5 design (record built at `Arena.swift:501-504` / `ManualPromote.swift:169-171` before the champion-change call); new test. |

## Pass 3 (P3-1, P3-2, NB-a–NB-c)

The reviewer verified N1–N6 and NB1–NB7 as correct.

| Item | Verdict | Where / why |
|---|---|---|
| P3-1 the inline post-promotion save would fail the P4 backstop | Accepted | Gap 5 design 2 (new bullet), the P4 step-3 touch list and the extended N2 test. Verified the save builds its trainer metadata schedule from the trainer's current `lrWarmupSteps` / `lrMomentumCycle` (`App/SessionController+Arena.swift:713-723`) while its lineage is `promotionLineage` (`:739`, `:790`). |
| P3-2 S2/B2 wording contradicts the N4 invariants and B9 | Accepted | Every S2 / B2 / B6 / gap 10 reference now says `configuration.value.path_kind`. `start_value_head_recentered`'s comment matches B9: `false` for weights built in-process, promoted, or a fresh CLI run; `null` only for continuation. The CLI fresh case is stated in B9, with test `testAFreshCLIRunRecordsFalse`. |
| NB-a duplicate item 4 in gap 5 | Accepted | Renumbered to 5. |
| NB-b single cycle evaluation | Accepted | Gap 12 design. Verified that `buildFeeds` evaluates once (`Training/ChessTrainer.swift:6166-6172`), while the readouts evaluate per channel (`:4449`, `:4475`). |
| NB-c capture after the training pause | Accepted | Gap 5 design 2. Verified the pause is acquired at `App/SessionController+Arena.swift:134`, before the export at `:170-181`. |

## Pass 4 (P4-1 and nits)

The reviewer verified P3-1, P3-2 and NB-a–NB-c.

| Item | Verdict | Where / why |
|---|---|---|
| P4-1 training continues through the arena; a promotion rewinds the clock but not the settings | Accepted, extended | Gap 5 design 2 (re-stamp under the promotion pauses), gap 4 (`restamped_from`, journal-clock encode check, three tests), B5 (`noteChampionChange` uses S), N2 test corrected. Verified: training is paused only around the snapshot (comment `App/SessionController+Arena.swift:126-133`, pause `:134`–`:188`); both gates pause at promotion (`:423-425`); the clock rewinds at `:463`. Extension beyond the review: the promotion record includes only the journal entries that existed at the arena-start capture, so its `parameters` (arena-start snapshot) and `parameter_changes` stay undo-consistent. |
| Nit: `configuration.path_kind` in P4 step 3 | Accepted | Now `configuration.value.path_kind`. |
| Nit: B9 null bullet too broad | Accepted | `null` only for Continue after Stop and New Session, keep trainer. A CLI `--resume-exact` records its start file's value. |

## Pass 5 (concurrence; final nits)

The reviewer concurred with the plan, with no blockers, on 2026-10-05. Four non-blocking nits were applied:

| Item | Verdict | Where / why |
|---|---|---|
| Arena line ranges inconsistent | Accepted | Now read consistently as comment `:126-133`, pause `:134`–`:188` (verified: comment 126-133, `pauseAndWait` 134, `resume` 188). |
| Encode check with a null step count | Accepted | Gap 4: skipped when the record's or summary's step count is `null`. |
| Separate journal arrays at arena-start capture | Accepted | Gap 5 design 2: the length of `parameter_changes` and of `champion_changes` is each captured. |
| `restamped_from` in the Python mirror | Accepted | S1 Python mirrors: the schema-3 entry keys, `restamped_from` included, are required. |

**Could not verify here** (no app, build or test runs while training is live):
- the build-setting variable names for B8;
- whether `GuiSaveHarness` can run without Metal, or drive an arena (the N2 and N5 tests fall back to the extracted composition functions if not);
- whether the trainer exposes the fed LR scalar for direct readback (O-18's bit-identical test otherwise compares the extracted pure function before and after);
- the multi-line O-1 counts;
- the log counts quoted in gap 2;
- whether the behavior fingerprint covers the 4096 √batch base.
