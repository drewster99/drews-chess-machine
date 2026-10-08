# Run labels — plan

Status: **proposed** (owner request 2026-10-08). Decisions §2 are the owner's (2026-10-08) unless marked
open. Nothing is implemented; implementation starts on the owner's go.

## 1. Goal

A model name (`model_naming.name`, lineage schema 4) names where a model **started**. Many runs can
train from one start (R-replay and R-fixedlr both carry `zlra-ab-fresh`), and every checkpoint of a
run shares one `ModelID` (`20261006-43-a89C` names all 40 of B-leakyall's files). Today the only
readable handle for "which training produced this file" is a dashboard registry or a folder name.

A **run label** names one lineage run (one `lineage_run_id`): "B-leakyall", "v5 seg 4",
"R-fixedlr". It lives in the file, is set once when the run begins, and is shown wherever a
model is identified: `<run label> @ step N (from <model name>)`.

`ROADMAP.md:62` already proposes an optional `run_label`; this plan supersedes it with a required
one.

## 2. Decisions

1. **Required when a training run begins**, i.e. whenever a training path mints a new
   `lineage_run_id`:
   - fresh start;
   - branch (non-exact `--start-model`, GUI start from a loaded model, GUI "reset trainer from
     champion");
   - resume of a pre-lineage file (mints a run ID today, `LineageTracker.swift:306`).
   Missing → refused at launch (CLI) or Start disabled (GUI). No pre-filled value: a pre-filled
   label is a silent default that gets clicked through.
2. **Not accepted where the run continues**: exact resume, GUI continue after Stop, GUI "new session
   keep trainer", GUI session resume, promotions. The label belongs to the run; one source of truth.
3. **Derive and graft require a label too**: they mint a run ID (`untrainedCopyRecord`,
   `ModelDerivation.swift:446`, `ModelGraft.swift:392`) that later training continues or branches
   from.
4. **A branch or derive must not reuse its parent run's label** (compared after trimming, case-
   insensitive). Global uniqueness is not checked: there is no reliable registry, and building one
   would be a side channel.
5. **Renaming (owner, 2026-10-08): live run only, recorded as an event, saved files never edited.**
   - CLI: `--rename-run <label>` with an exact resume (`--resume-exact`) only; a plain `--run-label`
     there stays refused, so a rename is never accidental.
   - GUI: Train ▸ Rename Run… on the live run.
   - The record keeps `changes: [{from, to, trainer_step, recorded_unix}]`. A file shows the label it
     was written with, plus "renamed to X" when its own history records a later rename. No side
     table mapping run IDs to their newest label.
6. **Records with no training run get no label**: `--new-model`, GUI Build, and the automatic
   untrained copy of a pre-lineage champion (`championFileLineageRecord`, `untrainedCopyRecord`).
   Their field is "does not apply", never a made-up label.
7. **Old files are never backfilled** (owner rule): pre-schema-5 records read as unrecorded and
   display the short run ID.

Open:

- **O-1. Schema number.** This adds lineage schema 5. `SELF_PLAY_FROM_TRAINER_PLAN.md` also
  proposes schema 5; whichever lands first takes 5 and the other takes the next number.

## 3. Design

### 3.1 Type and validation

- `RunLabel` (`Persistence/RunLabel.swift`): the label text plus `changes`.
- Validation uses the **same text rule as model names**: trimmed, 1–120 characters, no control
  characters. Move the rule out of `ModelNaming.validatedName` (`ModelNaming.swift:102`) into one
  shared validator that both call, so the two can't drift.

### 3.2 Lineage record (schema 5)

- New top-level field `run_label`, holding one of three states:
  - recorded `{label, changes}`;
  - recorded "does not apply" (`null`; no training run, decision 6);
  - unrecorded (schema ≤ 4).
- It follows the `model_naming` pattern (`LineageRecord.swift:1306–1312`):
  - decoded at `schema >= 5`;
  - `refuseKeys` below that;
  - written only at `currentSchema`;
  - passed through `withoutTrainerState()` (:1442), so champion files of promoted weights carry the
    trainer's label.
- `AncestorRun` gains the parent run's label, so a branch's history stays readable.
  - `AncestorRun.init(from:)` (`LineageRecordSchema3.swift:887`) decodes its segments with
    `currentSchema` and has no schema parameter, so the new key gets a schema-aware decode: absent in
    ancestry written at schema ≤ 4 reads as unrecorded.
- Flat mirror `lineage_run_label` in `MirrorKey` / `metadataEntries()` (`LineageRecord.swift:1412`,
  :1476), next to `lineage_run_id`. Header-only readers (bot, pickers, Python) can then show it
  without decoding the full record.
  - It is write-only like the other mirrors and never read back as truth.
  - `model_naming` has no mirror; adding one for it too is out of scope.
- `scripts/dcm_lineage.py`: `SUPPORTED_SCHEMA = 5`, `_SCHEMA_5_TOP = ("run_label",)`, and a reader
  `run_label(record)` mirroring `model_naming` (:264).

### 3.3 Tracker (`Persistence/LineageTracker.swift`)

- `LineageTracker.Start` cases that mint a run ID take a validated `RunLabel`:
  - `.fresh` and `.branch`;
  - `.resume` of a pre-lineage parent. That case is chosen inside the tracker, so the caller passes
    an optional label and the tracker refuses when the parent is pre-lineage and none was given.
- `.resume` of a recorded parent refuses a label (decision 2) and inherits the parent's.
- `.branch` refuses a label equal to the parent's (decision 4).
- `mintRecord` (:648) and `untrainedCopyRecord` (:675) record "does not apply", except that derive
  and graft pass their required label through `untrainedCopyRecord`.
- One function decides whether a start needs a label, takes one, or refuses one. The GUI, both CLI
  runners, derive and graft all call it, so each mode's rule is defined in one place. It is pure and
  unit-tested.

### 3.4 CLI

| Path | Parser | Flag |
|---|---|---|
| GUI `--train` | `DrewsChessMachineApp.swift` init, beside `--seed` (:404) | `--run-label <text>`, required (a `--train` start is always a new run) |
| `--replay-corpus` | `handleReplayCorpusIfPresent`, `switch` at :1000 | `--run-label` required unless `--resume-exact`; `--rename-run` only with `--resume-exact` |
| `--train-vs-uci` | `handleTrainVsUciIfPresent`, `switch` at :1294 | same as replay |
| `--derive-model`, graft | `DeriveModelCLI.swift`, beside `--name` (:133–141) | `--run-label` required |
| `--new-model` | unchanged | `--run-label` refused (no run) |

- Each refusal names the rule, e.g. "a fresh run needs --run-label", "an exact resume continues run
  'B-leakyall'; use --rename-run to rename it", "branch label equals the parent run's".
- Exit codes follow each path's existing refusal code.
- Help text goes in `CommandLineHelp.swift`.
- `[RUN]` line (`RunProvenanceLine.swift:35`): `label="<text>"` after `run=`; `label=none` for
  unrecorded or does-not-apply records.
- `results.json` `lineage` (`CliTrainingRecorder.ResultsLineage`, :333) adds `run_label` to its
  hand-written CodingKeys, as `modelNaming` was.

### 3.5 GUI

- **Start gate:** `startTrainingFromMenu()` (`UpperContentView.swift:2798`) and the three-way dialog
  (:1347) decide the `TrainingStartMode`. Before `startRealTraining(mode:)` (`SessionController+
  Training.swift:31`), when the mode begins a new run, a sheet asks for the label and Start stays
  disabled until it validates. New-run modes:
  - `.freshOrFromLoadedSession` with no pending session, or with a pre-lineage session;
  - `.newSessionResetTrainerFromChampion`.

  The label is passed into `beginLineageSegment` (`SessionController+Lineage.swift:240`) and from
  there to the tracker. One View struct in a new file under `App/UpperContentView/`.
- **Rename Run…** in the Train menu, enabled only while a run is live. It records a change event at
  the current trainer step, and the next save carries it.
- **Display sites** (show `label @ step` with the model name as origin; short run ID when
  unlabelled):
  - title bar / `ModelNameplate` (`ModelNameplate.swift:26`, `TitleBarView.swift:84`);
  - About (`AboutNameplateRows.swift`);
  - session picker group header and "Run" section (`SessionPickerSheet.swift:82`, :125), from
    `session.json` lineage via `SessionManifest`;
  - auto-resume block (`AutoResumeModelBlockView.swift:22`);
  - Lichess bot:
    - model record table: `LichessBotModelRowLabel.swift`, today the bare model ID;
    - model file picker row: `LichessBotModelLinePicker.swift:261`, adding `runLabel` to
      `ModelFileEntry` (`ModelFileCatalog.swift:7`) from the flat mirror;
    - followed-lineage row: `LichessBotFollowedLineageRow.swift:144`, today `run <8 chars>`;
    - ready log line: `LichessBotModelSlots.swift:156`, `label=`.

## 4. Phases

Each phase: build, then commit. Tests in the phase that needs them.

1. **Type, validator, schema 5 field, ancestry label, flat mirror, Python reader.**
   Tests:
   - validator cases (trim, empty, 120/121 characters, control characters);
   - a schema-4 record decodes with `run_label` unrecorded;
   - a schema-5 round trip in all three states;
   - ancestry written at schema 4 decodes;
   - Python reader against a schema-5 fixture.
2. **Tracker rules.** Tests for the pure "needs / takes / refuses a label" function across every
   start mode, the parent-equality refusal, inheritance on resume, does-not-apply for mint and
   untrained copies, and a rename event recorded at its step.
3. **CLI flags, `[RUN]`, `results.json`, help.** Tests: each path refuses a missing label,
   `--run-label` on an exact resume, `--rename-run` without one, and a branch label equal to the
   parent's; accepted labels land in the record and on the `[RUN]` line.
4. **GUI start gate, Rename Run…, display sites.** Tests: the gate's mode-to-requirement mapping
   (pure); the session manifest and model catalog expose the label; display strings for labelled,
   unlabelled and renamed cases.
5. **Docs:** CLAUDE.md (lineage schema 5, `[RUN]` field, the CLI flags), CHANGELOG, ROADMAP (`:62`
   item marked superseded by this plan).

### Known blast radius

- Test helpers that start runs or build records need explicit labels:
  - `LineageTestSupport.swift` (`forTests`, `sessionTestFixtureRecord`);
  - `ModelLineageTestSupport.swift`;
  - `GuiSaveHarness.swift`;
  - the local `config(...)` helpers in `ResumeEquivalenceTests`, `ReplayResumeRecordedParametersTests`,
    `ReplayResumeDiffLogTests` and `TrainingSideReviewRegressionTests`.
- About 20 files construct `LineageTracker` directly.
- Existing experiment launch scripts stay as written (they are the record of what ran). New launches
  pass `--run-label`.

## 5. Validation

- Full test suite passes, slow forensic suites included (persistence changes).
- A short `--replay-corpus` with `--run-label T1` and `--enumerate-checkpoints`:
  - every step file carries `run_label.label == "T1"` and `lineage_run_label`;
  - `[RUN]` shows `label="T1"`;
  - `dcm_lineage.py` prints it.
- An exact resume of it:
  - without a label: carries T1;
  - with `--run-label`: refused;
  - with `--rename-run T2`: later files show T2 with `changes` recording T1 → T2 at the resume step,
    and earlier files still read T1.
- A branch from it with `--run-label T1`: refused. With `T3`: its ancestry names T1.
- GUI:
  - a fresh Play-and-Train can't start without a label;
  - continue after Stop shows the label with no prompt;
  - the Lichess bot table, the model picker and the title bar show `label @ step`.
- An old file (for example `20260713-v5cont-resume-replay-step270000`) shows its short run ID and no
  label.
