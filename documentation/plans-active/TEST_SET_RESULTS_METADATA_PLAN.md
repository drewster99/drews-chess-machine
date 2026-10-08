# Test-set results in model-file metadata — plan

**Status: decisions settled (2026-10-07, owner); awaiting "start".** Nothing
here is implemented yet.

## Goal

Every model file DCM writes carries the puzzle test-set results **of the
weights in that file**, so a picker, a script or a person can see what a file
is without re-probing it. That covers corpus replay, train-vs-UCI, GUI
self-play session saves (periodic, promote, Promote Trainee Now, SIGUSR2,
manual) and File menu saves.

Per test set:
- id, title, short description, and a fingerprint of the puzzle list
- pElo and NLL
- total positions, top-1 correct count, top-5 correct count
- average probability on the right move, average rank of the right move
- per theme: correct count and total

Several sets (an array) go in **one** JSON metadata key. Percentages are
derived by the reader and are not stored.

## What exists today (research, 2026-10-07)

- **Probe math.** It is the same code for the GUI watcher and `--probe-model`:
  - `TacticalProbeRunner.runBatch` runs one batched forward.
  - `buildProbeResult` derives top-5, rank, probability and NLL from the
    legal-masked policy.
  - `LichessProbeHistory.aggregates` folds them per theme;
    `LichessProbeOverallSummary` folds them overall.
  - `LichessProbeHistory.mlePuzzleElo` computes pElo (Bradley-Terry MLE;
    ±inf when every answer is right or every answer is wrong).
  - `ProbeModelCLI.summary` already emits every field above as one JSON line.
- **Sets.** Both are bundled:
  - `Resources/lichess_probes_200.json` (200)
  - `Resources/lichess_probes_wide.json` (4,435)

  The only labels in code are `"200"` / `"wide"`. Each file's `metadata` block
  (source, snapshot, filters) is ignored by the loader. Theme display names
  live in two view-local switches (`LichessProbeMonitorView`,
  `LichessProbeDetailView`).
- **Cost.**
  - Both sets in one batch: 0.2–0.6 s of GPU time (GUI ticks, M4 Pro).
  - Wide only through the CLI: 1.1–1.5 s, plus the network build.
- **What's already saved.** No probe data is in safetensors metadata.
  - `session.json` holds the GUI probe histories.
  - `manifest.json` holds `latestPElo200` / `latestPEloWide`. These come from
    the last probe tick of the *trainer snapshot then*, not necessarily the
    saved weights.
- **Metadata.**
  - `SafetensorsModelIO.encode` writes every key. JSON blobs already in use:
    `architecture`, `dcm_lineage`, `trainer_lr_momentum_cycle`,
    `derivation_history`.
  - Readers ignore unknown keys.
  - `content_sha256` hashes only the tensor data, so adding a key changes no
    hash.

## Design

### D1 — One evaluator, run on the saved weights

`ModelTestSetEvaluator` (new, `Training/` or `Persistence/`) takes the
architecture plus the exact weight arrays being saved. It returns a
`ModelTestSetResults` record. It owns one inference `ChessMPSNetwork` per
architecture, built on first use and reused: the weights are loaded fresh for
each save, and the network is rebuilt only when the architecture changes. It
calls the same `TacticalProbeRunner.runBatch` + aggregation code the watcher
and the CLI use. There is no second copy of the math.

`ProbeModelCLI` and `LichessProbeWatcher` keep their outputs but take their
numbers from the same aggregation, so the three can't drift apart.

**Probe isolation (CLAUDE.md).** The evaluator reads only the exported
weights. It uses no RNG (argmax and softmax only), and it touches no trainer,
optimizer, buffer or stream state.

### D2 — When it runs, relative to the save's consistency cut

A save already exports its weights under its pauses (the GUI's one consistent
cut). The evaluator runs **after** those pauses are released, on the exported
arrays, and **before** the file is encoded: the results go into the header of
the one and only write. There is no save → read back → probe → rewrite pass
(the owner's preference). Encoding and writing the file follow. So self-play and training pause
no longer than they do today, and a save takes about 0.5–1.5 s longer per file
probed.

| Save path | Files probed |
|---|---|
| GUI session save (all triggers) | `champion.safetensors` and `trainer.safetensors`, each on its own weights |
| File ▸ Save Champion | the champion |
| Corpus replay | once per save: the rolling out-model and the enumerated checkpoint are the same bytes |
| Train-vs-UCI session / enumerated save | trainer (and champion if the session has one) |
| `--new-model`, `--derive-model`, graft | the new file (cheap; keeps the rule "every file carries results") |

### D3 — The encoder requires it

`SafetensorsModelIO.encode` gets a **required** `testSetResults:` parameter of
type `ModelTestSetResultsField`:

```swift
enum ModelTestSetResultsField {
    case evaluated(ModelTestSetResults)
    case failed(reason: String)
}
```

The parameter is required (no default), so the compiler finds every writer
and none can forget it. A probe failure never fails the save: the key records
the failure, and the save logs it.

`SafetensorsFile.encode` direct callers (`ModelDerivation`, `ModelGraft`) move
to `SafetensorsModelIO.encode`, or pass the field the same way.

### D4 — The metadata key and its JSON

Key: `dcm_test_set_results`. It is one JSON string, sorted keys, written by
one `Codable` type:

```json
{
  "schema": 1,
  "status": "evaluated",
  "evaluated_at_unix": 1791414697,
  "build": 2425,
  "policy_tail_precision": "mixed_final_projection",
  "sets": [
    {
      "id": "lichess_wide_2026_06",
      "title": "Lichess puzzles, wide",
      "description": "4,435 puzzles rated 400–3200, flat per-100 density, mate-weighted (Lichess DB 2026-06).",
      "fingerprint_sha256": "…",
      "positions": 4435,
      "top1_correct": 2212,
      "top5_correct": 3733,
      "avg_correct_probability": 0.1923,
      "avg_correct_rank": 3.667,
      "nll": 2.0710,
      "pelo": 1630.4,
      "pelo_bound": null,
      "errored": 0,
      "themes": [
        {"id": "hangingPiece", "title": "Hanging piece", "correct": 253, "total": 309}
      ]
    }
  ]
}
```

- **`pelo`**: JSON has no infinity. When every answer is wrong or every
  answer is right, `pelo` is `null` and `pelo_bound` is `"all_wrong"` /
  `"all_correct"`.
- **`fingerprint_sha256`**: a hash of the set's puzzle list (ids + FENs +
  moves). Results are only comparable when it matches, which guards against a
  regenerated set reusing an id.
- **`failed`**: `{"schema": 1, "status": "failed", "reason": "…"}`.
- **Older files**: no key means "not recorded". The decode type has three
  states: not recorded, failed, evaluated.
- **No format-version bump.** The key is optional and older builds ignore it.
  This is unlike architecture fields, which are required.

### D5 — Test-set identity has one source

Each bundled set's own JSON `metadata` block gains `id`, `title` and
`description`. `LichessProbeData` decodes them, and the fingerprint is
computed at load. The theme display names move into one place
(`ProbeCategory.title`). Both views and the metadata use it, which removes the
two view-local switches. `scripts/curate_lichess_probes.py` writes the new
fields so a regenerated set keeps them.

### D6 — Reading and display

- `SafetensorsModelIO.decode` and the header-only
  `ModelFileCatalog.headerMetadata` decode the key into
  `ModelTestSetResultsField?`.
  - A malformed blob is reported as unreadable results, with the reason. It
    does **not** refuse the model: the results are informational, not
    training state.
- **Summary shown in pickers**: from the set with the most positions, show
  the set title, pElo, NLL, top-1 %, top-5 %. When the results are failed or
  absent, the picker says so ("results not recorded", or "results failed:
  …"), never blank.
  - Lichess bot model picker rows (`LichessBotModelFileRow`, through
    `ModelFolderHeaderCache`).
  - Session picker (`SessionPickerSheet`). Its pElo column and detail move
    from `manifest.json`'s last-tick values to the saved files' own results
    (see O-2). The manifest gains the same summary, so the list doesn't open
    every safetensors header.
- `scripts/dcm_lineage.py` gets a reader (`test_set_results(path)`), so the
  dashboards can use embedded results instead of re-probing.

## Decisions (owner, 2026-10-07)

- **Both session files.** A session save evaluates and writes results into
  **both** `champion.safetensors` and `trainer.safetensors`, each from its own
  weights.
- **Probe before writing** (preference, met by D2): the results go into the
  only write of each file.
- **O-1 Probe failure:** as recommended — save anyway, with the failure
  recorded and logged.
- **O-2 Session picker:** must show at least the summary elements (set
  id/title, pElo, NLL, top-1 %, top-5 % of the largest set). Implementation
  choice: the list row shows the trainer's (what today's pElo column
  describes); the detail shows the champion's and the trainer's side by side.
- **O-3:** every model write carries the results (derive, graft and
  `--new-model` included).
- **O-4:** no backfill of existing files now.
- **O-5:** the 9 hand-built tactical positions are left out.

## Open decisions as first proposed

- **O-1 Probe failure.** Recommend: save anyway, record `failed` with the
  reason, log it, and raise the existing checkpoint-status warning in the
  GUI. The alternative, failing the save, risks losing training state over a
  display feature.
- **O-2 Which file the session picker summarizes.** Recommend the trainer:
  that's what today's manifest pElo describes, since the probe watcher's
  default target is the candidate. The champion's results are shown beside
  it in the detail view.
- **O-3 Derive / graft / `--new-model`.** Recommend including them, so every
  file DCM writes carries results. A derived file's weights differ from its
  source's.
- **O-4 Existing files.** Recommend no backfill in this plan. Filling them in
  would rewrite headers, which is migration-like work. A later
  `--annotate-test-results <file|dir>` could add the key in place, in the
  same way as the policy-tail header edit, if asked.
- **O-5 The 9 hand-built tactical positions.** Recommend leaving them out:
  they're a smoke test, not a test set.

## Validation

Unit tests (new class `ModelTestSetResultsTests`, plus additions where
noted):

1. **JSON round trip.** Encode → decode is identical for the evaluated,
   failed and not-recorded cases. `pelo` null + `pelo_bound` for all-right and
   all-wrong sets. Sorted keys. Every number finite.
2. **Same math as the CLI.** For a seeded fixture network, the evaluator's
   numbers equal `ProbeModelCLI.summary`'s for both sets, field by field.
3. **Results describe the saved weights.** Save a session whose champion and
   trainer weights differ. Each file's results equal a fresh probe of that
   file's own weights, and the two differ.
4. **Every writer carries the key.** Covers the GUI session save (both files),
   File ▸ Save Champion, a corpus-replay save (rolling and enumerated bytes
   equal), a train-vs-UCI save, `--new-model`, derive and graft. Uses the
   existing harnesses (`GuiSaveHarness`, corpus-replay and train-vs-UCI
   session tests).
5. **Failure path.** An evaluator forced to throw still writes the file, with
   `status: failed` and the reason. The save reports success with a warning.
6. **Probe isolation.** A save with results leaves the trainer's dropout
   state, the RNG streams, the optimizer and the replay buffer
   byte-identical (pattern of `DropoutRNGStateTests`).
7. **Reading.** Old files (no key) decode as not recorded. A malformed key is
   reported but the model still loads. Header-only and full decode agree.
8. **Set identity.** Ids, titles and descriptions come from the bundled JSON.
   The fingerprint is stable, and changes when one puzzle changes.
   `ProbeCategory.title` covers every theme.
9. **Picker summary.** Picks the largest set. Formatting covers both pElo
   bounds and the failed and not-recorded cases.

Run-time checks:

10. A short corpus-replay run (`--training-step-limit 2000
    --enumerate-checkpoints`). `scripts/dcm_lineage.py` prints the results
    from both enumerated files, and they match `--probe-model` on the same
    files. The run log shows the extra time per save.
11. A GUI session save: the session picker and the Lichess bot picker show
    the summary, matching the files.

Full test suite before merge: this changes persistence and serialization
(CLAUDE.md).

## Phases (one commit each, build + targeted tests per phase)

1. Set identity: JSON fields, loader, fingerprint, `ProbeCategory.title`,
   views moved to it; curate script.
2. `ModelTestSetResults` types and JSON, plus the evaluator, which shares
   `ProbeModelCLI`'s aggregation. Tests 1, 2, 8.
3. Required encoder parameter, and every writer wired in (D2/D3). Tests 3–6.
4. Reading, the manifest summary, both pickers, `dcm_lineage.py`. Tests 7, 9.
5. Docs: CLAUDE.md "Saved model state" (the new key), CHANGELOG, and this
   plan marked as built. Run-time checks 10–11 and the full suite.
