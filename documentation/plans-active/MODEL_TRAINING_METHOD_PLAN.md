# Model training method in model pickers

Owner request 2026-10-08: "This picker should indicate if the model is from a self play, uci play, corpus replay — if we know. also every place else we pick a model."

## Owner decisions (2026-10-08)

- **Mixed history: a chain, oldest first.** A model trained by more than one method shows every method its weights went through, oldest first, consecutive repeats collapsed: `corpus replay → self-play`.
- **Past Lichess games: read the played file's header.** New games record the method. A game recorded before that reads it from the header of the file it played, when that file still holds the same `model_id`. Game records are never rewritten. Past games played from the live trainer or champion stay unknown.

## Decisions made while planning

- **Labels:** `self-play` (GUI Play-and-Train), `corpus replay` (`--replay-corpus`), `UCI play` (`--train-vs-uci`). Unknown or untrained shows nothing ("if we know").
- **Unknown earlier history is not marked.** A chain lists what the records state. No "…" marker for a history before the oldest run that no record covers; it would sit on nearly every older file.
- **Past games match by `model_id`, not file hash.** A rolling `-latest` file is rewritten in place by its own run (one `model_id` per run), so the played file's method is still the file's method after a later save; a whole-file hash would call every rolling file unknown.
- **Pre-lineage GUI files are unknown.** A file without a lineage record is known only from its `creator`: `replay` → corpus replay, `train-vs-uci` → UCI play. GUI writers (`manual`, `periodic`, `promote`, `sigusr2`) also saved untrained or loaded models, so they say nothing about training.
- **A pre-lineage champion file stays unknown** for the bot's champion source: `ChampionOrigin` carries the file's lineage, not its `creator`.

## Where the method comes from (one derivation: `ModelTrainingHistory`)

From a lineage record, oldest first:
1. each ancestor run's segments (`ancestry.runs[].segments[].configuration.path_kind`), skipping segments whose configuration is unrecorded or nil (not trained);
2. the record's earlier segments (`segments[].configuration`); an unrecorded one (a schema-2 summary, which names no path) adds nothing, and loses nothing: a run never changes path, so the record's own segment names that run's method and the repeat would collapse;
3. the record's own segment (`SegmentSummary(of:)`): `configuration.path_kind` when recorded; for a schema-2 record (configuration unrecorded) that trained (`parameters` present), `invocation.path_kind`; nothing when not trained.

A segment summary uses the same rule (its own `parameters` and `path_kind`).

`path_kind` maps `gui` → self-play, `replay` → corpus replay, `vsuci` → UCI play; `derive` / `new_model` never appear in a configuration and map to nothing.

Without a record: the `creator` rule above. An unreadable record: unknown.

## Where it shows

1. **Lichess Record card Model menu and Models pane** — `LichessBotModelCheckpointStatistics.label` (shared by both) appends the chain.
   - `LichessBotWeightsSnapshot.trainingHistory`: the file loader (lineage + `creator`), the champion (`championOrigin`: built → none, file → its lineage), the trainer (`LineageTracker.trainingHistory`; none without a tracker).
   - `LichessBotGenerationInfo.trainingHistory` (optional; nil in records written before it) → `LichessBotGenerationFacts.trainingHistory`; index schema 5.
   - Index rebuild resolves a nil history of a file generation from the recorded file path's header (`model_id` must match), once per path per rebuild; a file that's gone, unreadable or another model logs one `[LICHESS-BOT]` line and stays unknown.
2. **Lichess bot model file / lineage picker** (`LichessBotModelLinePicker`) — `ModelFileEntry.trainingHistory`, from the catalog's header read.
3. **Load Session picker** — champion and trainer methods from the session's model file headers (safetensors only; a legacy `.dcmmodel` session reads unknown, never the whole file), in the Run section and the run's group header.
4. File ▸ Load Model and the probe comparison loader use macOS open panels: nothing to annotate.

## Validation

- `ModelTrainingHistoryTests`: every rule above (fresh self-play, replay, vsuci, ancestry chain with repeats collapsed, schema-2 invocation fallback, untrained, unrecorded segment, creator fallback, unreadable).
- Lichess: generation → facts → choice label carries the chain; the index resolves a past file game from its file's header and refuses another `model_id`.
- Catalog entry and session picker read the chain from a written file.
- Build with no new warnings; targeted test classes pass.
