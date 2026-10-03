"""Reconcile the run registry with the lineage records in the model files.

`replay.py derive-registry` (and the equivalent self-play / vs-UCI commands)
use this module. It is pure — no file access, no registry loaded at import — so
the reconciliation rules can be tested on synthetic inputs.

What is derived, per registry segment, from `scripts/dcm_lineage.py`:
`lineage_run_id`, `segment_id`, `model_id`, `date`, `cumstep_base`,
`games_base`, `elapsed_base_sec`, `wall_base_sec`, `device`.

Rules:

- **Matching.** A lineage run belongs to the registry run whose segments name
  its `lineage_run_id`, or — before that has been written — whose segments name
  one of its files' `model_id`s. A lineage run matching two registry runs is a
  contradiction and is reported, never resolved by picking one. A lineage
  segment matches the registry segment naming its `segment_id`, else its
  `model_id`.
- **Fill only.** A derived value is written only into a field the registry
  segment lacks. A field the registry already holds is compared: equal is
  "same"; different is a conflict, reported and left as is. A pinned
  hand-entered base is never overwritten by this command.
- **The step axis.** `cumstep_base` is the segment's start on the lineage run's
  own trainer clock, measured from the run's segment 0. When the run's segment 0
  is the registry run's first segment and the run did not continue unrecorded
  history, that clock *is* the registry's axis. Otherwise the registry axis is
  anchored at the `cumstep_base` the registry already holds for the segment
  matching the run's segment 0; with no such anchor the base stays unrecorded.
- **New runs and segments are not created.** A registry run carries a label,
  color and session log that no file states; a lineage run or segment with no
  registry counterpart is reported as a proposal for the owner to add.
"""
import copy

# The fields reconciled, in report order. `segment_id` and `lineage_run_id` are
# identity; the rest are the measured bases.
RECONCILED_FIELDS = ("lineage_run_id", "segment_id", "model_id", "date", "cumstep_base",
                     "games_base", "elapsed_base_sec", "wall_base_sec", "device")
# Float fields compare equal within this absolute tolerance: they are sums of
# measured seconds, and a hand-entered value written with a few decimals is the
# same value.
FLOAT_TOLERANCE = 1e-6


class RegistryContradiction(ValueError):
    """Registry entries and lineage records that cannot both be right."""


def _equal(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= FLOAT_TOLERANCE
        except (TypeError, ValueError):
            return False
    return a == b


def _registry_run_for(reg_runs, derived_run):
    run_id = derived_run.lineage_run_id
    model_ids = {s.fields["model_id"] for s in derived_run.segments if "model_id" in s.fields}
    hits = []
    for name, cfg in reg_runs.items():
        segments = cfg.get("segments", [])
        if any(sg.get("lineage_run_id") == run_id for sg in segments) or \
                any(sg.get("model_id") in model_ids for sg in segments if sg.get("model_id")):
            hits.append(name)
    if len(hits) > 1:
        raise RegistryContradiction(f"lineage run {run_id} matches registry runs {hits}")
    return hits[0] if hits else None


def _registry_segment_for(segments, derived_segment):
    hits = []
    for index, sg in enumerate(segments):
        if sg.get("segment_id") and sg.get("segment_id") == derived_segment.fields.get("segment_id"):
            hits.append(index)
        elif sg.get("model_id") and sg.get("model_id") == derived_segment.fields.get("model_id"):
            hits.append(index)
    if len(hits) > 1:
        raise RegistryContradiction(
            f"lineage segment {derived_segment.fields.get('segment_id')} matches registry segments {hits}")
    return hits[0] if hits else None


def plan(reg, derived_runs):
    """Work out what `derive-registry` would change.

    `reg` is the parsed registry and `derived_runs` the result of
    `dcm_lineage.derive_runs`.

    Returns (updated_registry, report). `updated_registry` is a deep copy with
    every fill applied; `report` is a dict with `fills`, `same`, `conflicts`
    (lists of (registry run, segment index, field, registry value, derived
    value)), `unmatched_runs` (lineage run id -> derived segment table) and
    `unmatched_segments` ((registry run, lineage segment fields) pairs), and
    `unrecorded` ((registry run, segment index, field)) for derivable fields
    the records hold as null."""
    updated = copy.deepcopy(reg)
    runs = updated["runs"]
    report = dict(fills=[], same=[], conflicts=[], unmatched_runs={}, unmatched_segments=[], unrecorded=[])
    for run_id, derived in sorted(derived_runs.items()):
        name = _registry_run_for(runs, derived)
        if name is None:
            report["unmatched_runs"][run_id] = [dict(segment_index=s.segment_index, **s.fields)
                                                for s in derived.segments]
            continue
        segments = runs[name]["segments"]
        mapping = {}
        for derived_segment in derived.segments:
            index = _registry_segment_for(segments, derived_segment)
            if index is None:
                report["unmatched_segments"].append((name, dict(segment_index=derived_segment.segment_index,
                                                                **derived_segment.fields)))
            else:
                mapping[derived_segment.segment_index] = index
        # Registry axis offset for this lineage run's trainer clock.
        offset = None
        if 0 in mapping:
            anchor_index = mapping[0]
            if anchor_index == 0 and not derived.continues_unrecorded_history:
                offset = 0
            elif "cumstep_base" in segments[anchor_index]:
                offset = segments[anchor_index]["cumstep_base"]
        for derived_segment in derived.segments:
            if derived_segment.segment_index not in mapping:
                continue
            index = mapping[derived_segment.segment_index]
            target = segments[index]
            values = dict(derived_segment.fields)
            unrecorded = set(derived_segment.unrecorded)
            if "cumstep_base" in values:
                if offset is None:
                    del values["cumstep_base"]
                    unrecorded.add("cumstep_base")
                else:
                    values["cumstep_base"] = offset + values["cumstep_base"]
            for field in RECONCILED_FIELDS:
                if field not in values:
                    if field in unrecorded:
                        report["unrecorded"].append((name, index, field))
                    continue
                if field not in target:
                    target[field] = values[field]
                    report["fills"].append((name, index, field, None, values[field]))
                elif _equal(target[field], values[field]):
                    report["same"].append((name, index, field, target[field], values[field]))
                else:
                    report["conflicts"].append((name, index, field, target[field], values[field]))
    return updated, report
