# Model naming plan: a model keeps its name, its preset and shows its file format

Status (2026-10-07): approved by the owner (design questions answered 2026-10-07); implemented; the session picker shows the name too (`SessionManifest.modelNaming`; a manifest or index-cache entry written before it shows none). Deviation: a graft records the target preset, unedited (`ModelNaming.ofGraft`), not the source's preset marked edited.

Paths are relative to `DrewsChessMachine/DrewsChessMachine/` unless they start with `DrewsChessMachine/`, `DrewsChessMachineTests/`, `documentation/` or `scripts/`.

## Problem

- The title bar and About popover start every summary with `v3` / `v4` / `v5`: `NetworkArchitecture.architectureVersionLabel`, a display-only "family" worked out from the block style (output norm → 5, pre-activation → 4, else 3). Any network with an output norm reads "v5", so the label says almost nothing about the model and collides with the "v5" training line (`documentation/v5-lineage.md`).
- The New Network screen's Name field and preset picker are dropped at Build: `BuildNewModelRequest` carries only the architecture and init seed, and no file stores a name. This is the leftover in `RUNTIME_ARCHITECTURE_CONFIG_PLAN.md` ("Preset `label` in `__metadata__` … not done").

## Owner decisions (2026-10-07)

- **D1 — the name rides with the model.** It survives saves, resumes, branches, promotions and trainer forks. `--new-model` and `--derive-model` take `--name`; a derive keeps the source's name unless `--name` is given.
- **D2 — "preset it came from" = the preset last chosen in Start from, plus whether the built topology was edited away from it.**
- **D3 — the `v3/v4/v5` label is replaced by the format of the file the weights came from.** Weights made in this process (a build, a promotion) show the format this build writes.

## Design

### Where the name lives: the lineage record (schema 4)

D1's behavior is exactly the inheritance the lineage record already has. Every model file written since format v7 carries a `LineageRecord`, built in one place (`LineageTracker`), and its run-level facts are carried to the next file by the documented rules: fresh / branch / resume / derive / champion file / promotion. So the name is a record field, not a second copy carried beside each network's `ModelID`:

- `LineageRecord.modelNaming: Recorded<ModelNaming>`, JSON key `model_naming`. **Schema 4** adds it: required in a schema-4 record, refused in an older one, which reads back as `unrecorded` (never filled in). `currentSchema` becomes 4; `oldestDecodableSchema` stays 2.
- `ModelNaming` (`Persistence/ModelNaming.swift`):
  - `name: String?` — the name given at build (`--name`, the Name field); nil when none was given. Validated: trimmed, non-empty, at most 120 characters, no control characters.
  - `presetStart: Recorded<PresetStart?>` — `recorded(PresetStart(preset:edited:))` for a model whose topology started from a preset; `recorded(nil)` for one built from no preset (Start from "Custom"); `unrecorded` only when a `--name` is given to a derive whose source records no naming.
- Inheritance (`LineageTracker`):
  - fresh → the naming the start states (GUI Build, `--new-model`, a fresh `--replay-corpus` / `--train-vs-uci`);
  - branch / resume → the parent record's naming, or `unrecorded` for a parent without a record;
  - `--derive-model` → the source's naming. When the derive changes the architecture, a recorded unedited preset start becomes `edited: true`. `--name` replaces the name and keeps the preset start (`ModelNaming.ofDerive`).
  - graft (`--graft-to`) → the target preset, unedited (none for a target given as a file), and the source's name unless `--name`; unrecorded when neither name is known (`ModelNaming.ofGraft`).
  - GUI save of a pre-lineage champion → unrecorded (the source has no record).
  - A champion file written from a record (`withoutTrainerState`) keeps the record's naming, so a promoted champion carries the trainer run's naming.
- The CLI's fresh runs: `--preset <name>` → `PresetStart(preset: name, edited: false)`; no `--preset` → the default `NetworkArchitecture.current` preset (`v4_5block_7x7`), unedited. `--new-model --architecture`: a built-in or user preset name → that preset, unedited; a path to an architecture file → `recorded(nil)` (no preset; `--name` names it).
- The GUI Build: `BuildNewModelModel.startedFromPreset` is set by the Start from picker (and by `init` to the preset the initial fields equal, nil for `.newModelDefault`, which the picker shows as "Custom"); `edited` = the built architecture differs from that preset's. The Name field (`labelOverride`) is the name.
- `scripts/dcm_lineage.py` reads schema 4 (`model_naming` required at 4, refused before).

### Display

- `NetworkArchitecture.architectureVersionLabel` is removed. `shortLabel` and `architectureSummary` lose the `vN` prefix (topology only). `ArchitectureMetadata` (session.json) and `AnalysisExportMetadata.Architecture` stop writing `architectureVersion`; older session.json files still decode (an unknown key is ignored by the synthesized decoder).
- `ModelNameplate` (`App/ModelNameplate.swift`): the champion's naming + where its weights came from (`.file(formatVersion:)` / `.thisProcess`). Derived from `SessionController.championOrigin` in its `didSet` (single source), published as the observable `championNameplate`.
  - `ChampionOrigin.built(initialization:naming:)` → `.thisProcess`.
  - `ChampionStartWeights.loaded(_:fileFormatVersion:)` → `.file(formatVersion:)`; `.notLoaded` (promotion) → `.thisProcess`.
- Title bar: `<name> · preset <p> (edited) · format v<N> · 3-block 9×9 · 128ch · 8,271,279 params`; an absent name / preset part is omitted.
- About popover: Name, Preset and File format rows that also say "none given", "none (custom)", "not recorded (file predates names)", and "weights made in this process; saves write v<N>".
- Auto-resume Models block and session picker: the name / preset from the session's record.
- Logs: `[BUTTON] Build Network`, `[BUILD]` and `[NEW-MODEL]` lines state the naming.

## Phases (build + commit per phase)

1. **Schema 4 + tracker + writers.** `ModelNaming`, the record field and codec, tracker starts and copies, every caller (GUI build, CLI fresh runs, `--new-model --name`, `--derive-model --name`, graft), the Python reader. Tests: `ModelNamingTests` (validation; JSON round trip; schema-3 record → unrecorded; schema-3 record with `model_naming` refused; schema-4 record without it refused; fresh/branch/resume/copy inheritance; edited flag on an architecture-changing copy; `--name` override keeps the preset start); existing tests updated only where the schema number or a removed API forces it.
2. **Display.** Remove `architectureVersionLabel`; nameplate; title bar, About, auto-resume, session picker, log lines. Tests: nameplate text for every case; summary strings without the prefix.
3. **Docs.** CLAUDE.md (lineage schema 4, the owner decisions above), `documentation/deriving-models.md` (`--name`), `--help` text, CHANGELOG, `RUNTIME_ARCHITECTURE_CONFIG_PLAN.md` leftover marked resolved.

## Validation

- Build succeeds with no new warnings.
- New and touched test classes pass; the full suite passes before the last commit (schema and persistence change → full run per CLAUDE.md).
- Manual: build a network named "naming-test" from preset `v4_5block_7x7` with one edit → title bar `naming-test · preset v4_5block_7x7 (edited) · format v12 · …`; save a session; relaunch and resume → same nameplate, format v12; load an old model → no name/preset, its own format version; `--new-model --architecture nt8y --name x` → header `dcm_lineage.model_naming` = `{"name":"x","preset_start":{"recorded":true,"value":{"preset":"nt8y","edited":false}}}` (read with `scripts/dcm_lineage.py`).
