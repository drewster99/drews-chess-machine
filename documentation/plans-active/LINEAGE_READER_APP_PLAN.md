# Python model-file readers read the app — `dcm_lineage.py` (2026-10-21)

**Status: done 2026-10-21** (L1–L4). Deviations: instead of a separate full-loader comparison, `decode` itself calls `inspect`, so the two cannot disagree on a header rule; `--describe-model` over all 5,468 model and session files described 5,466 and refused the 2 hand-made seeds the loader has refused since 2026-08-12; on the other 5,466 the old Python reading and the app's were identical (record, step reading with its note, test-set status); the Python tests' real files are copies of one minted tiny model with rewritten headers (`documentation/dashboards/tests/app_files.py`), which also gives pre-v7 and unversioned files the app can no longer write; `sessions_summary.py` still checks `session.json`'s own fields (format version, required keys) in Python, since no app command describes a session folder.

Owner request (2026-10-20/21): "Yes do dcm_lineage.py also" — after `scripts/dcm_arch.py` became a file-based wrapper over `--describe-model` (f114f95e). Same decision applies: the app is the one implementation of every model-file rule; Python asks it, on files; no input path in the app exists only for tests; Python tests use real files the app writes, and rule tests live in Swift.

## What `dcm_lineage.py` re-implements today (moves to the app)

| Python | App rule it mirrors |
|---|---|
| `format_version_of`, `lineage_of` (presence by format version, JSON parse) | `SafetensorsModelIO.lineage(fromMetadata:formatVersion:)` |
| `validated_record` + `_REQUIRED*`, `_SCHEMA_*`, `_check_*` | `LineageRecord` decoding (schemas 2–5) |
| `step_reading`, `_header_int`, `_trainer_clock`, `StepReading` table | `ModelFileStepReading`, `TrainerScheduleState.decode` |
| `test_set_results`, `_checked_test_set_results` | `ModelTestSetResultsField.reading` |
| `dcm_arch.parsed_format_version` / `checked_format_version` (kept for these) | `ArchitectureFormat.safetensorsFormatVersion` |

Stays in Python (analysis over app-validated records, not app rules): `model_naming`, `corpus_ids`, `weights_totals`, `device_label`, `derive_runs`, `segment_table`, `display_name`, `model_paths`, `scan`/`scan_files`, `checkpoint_facts`.

## Constraint: scans need header-only reads

`replay.py` scans every model file (`Models/` is 130 GB, `Sessions/` 140 GB). `--describe-model` today runs the full loader (reads every tensor), which is correct but far too slow for scans, and one process per file adds launch cost. So:

## Phases

### L1 — one header-level check path in the app
- `SafetensorsFile`: split `decode` into a header parse (`parseHeader`: index entries + metadata, every structural check: dtype, shape, offsets, contiguous coverage, shape vs byte count — needs only the header bytes and the data-region size) and the data read. `decode` = `parseHeader` + read floats + content-hash check (unchanged behavior). New `readHeader(at:)` reads only the header bytes of a file.
- `SafetensorsModelIO`: move every header-level rule of `decode` into `inspect(header:source:)` (architecture, tensor names / element counts / dims against the plan, velocity presence, trainer schedule, gradient-norm history, lineage presence and record, model ID, step reading). `decode` calls it, then does the data work (un-transpose, value-head recentering). One implementation of each rule.
- `--describe-model <file>...`: header-only, through `inspect`; one JSON object per line (`{"path", "description"}` or `{"path", "error"}`), so one process describes thousands of files. The description adds `lineage` (`recorded` + the record as stored, or `unrecorded` + format version), `step_reading`, `test_set_results` (`ModelTestSetResultsField.reading`), `creator`. Exit 0 when every file described, 1 when any was refused (each refusal is its own line), 2 on a usage error.
- Validation: full Swift suite (persistence change); `--describe-model` on all 5,308 model files agrees with the full loader (same refusals).

### L2 — Python readers over the app
- `dcm_arch.describe_files(paths)` (batched, cached per file version, copies) and `describe(path)`; `dcm_arch` drops its format-version parsing (no remaining user).
- `dcm_lineage`: `lineage_of(path)`, `step_reading(path)`, `test_set_results(path)`, `checkpoint_facts(path)`, `scan_files(paths)` read the description; `FileLineage` carries the description (model ID from it). The re-implemented rules and their constants are removed.
- `sessions_summary.py`: the session's lineage from its `trainer.safetensors` description (the same record `session.json` carries) instead of validating `session.json`'s copy in Python.
- Every consumer passes paths (ckpt_inventory, replay, selfplay, vsuci, _lineage_registry, bn_liveness, probe_record, reconstruct_pre_lineage_params, replay_supervisor, resume_equivalence_compare, policy_tail_* where they use the removed functions).
- Validation: old vs new reading on every model file (record identical, step reading identical, test-set status identical); all dashboards Python tests.

### L3 — tests
- Map each Python rule test (test_lineage.py, test_lineage_schema3.py, test_test_set_results.py, parts of test_replay_probe.py / test_reconstruct_pre_lineage_params.py / test_tooling.py) to its Swift test; add Swift tests for gaps; then remove the Python rule tests (owner approved for dcm_arch; same rule).
- Python analysis tests (derive_runs, weights_totals, …) stay: they test Python code that exists, on record dicts the app has validated.
- New Python tests: the readers on real files the app writes (`--new-model`, `--derive-model`), including a refused file.

### L4 — docs, CLAUDE.md, ROADMAP, CHANGELOG; full recheck; commit and push per phase.
