# Lichess Probe Detail — auto-compare to session-start + wide-set comparison

**Status: DONE** (shipped and in use as of 2026-06-23; the original build-verified
note is superseded). Implements an auto-comparison that, by
default, diffs the live probe against **this session's start-of-training
snapshot** (the `training-start-*` auto-exports), for **both** sets, with a
header toggle and UserDefaults persistence. Also adds the OVERALL summary band
to the wide set and lights up the wide charts' comparison reference lines.

> **Refinement since drafting:** manual `Compare…` no longer shows a friendly
> mismatch alert on a probe-count that doesn't match a set exactly. It now routes
> the picked file to the **nearest** set size — comparing `probeCount` against the
> live `LichessProbeData.wideSet.count` / `largeSet.count` and pinning to whichever
> is closer (`routeManualComparison` in `LichessProbeDetailView.swift`). Landed via
> follow-up commits `524d59f` / `d593c2b` / `b585198`. Read the "mismatch → friendly
> alert" passages below as superseded.

## Goal / behavior
When the Detail window opens, default to comparing live-vs-session-start so the
"how much has the model improved since training started" delta is visible with
no clicks — but never clobber a comparison the user deliberately chose, and
never silently compare against a *previous* session's file.

## State
- **In-memory, on `SessionController` (`@Observable`):** the path(s) of the
  start-of-training exports written **this launch** —
  `currentSessionStartSet200URL: URL?`, `currentSessionStartWideURL: URL?`.
  Set when `scheduleStartOfTrainingProbeExport` fires (point E). nil until then.
  Because we *wrote* them this launch, "is it the current session?" is answered
  by construction — no filesystem scan, no heuristic.
- **UserDefaults (per set):** `lichessCompareLastSet200Path`,
  `lichessCompareLastWidePath` — the last *manually* chosen comparison file for
  each set. Absent ⇒ that set is in auto mode.
- **In-memory, on the Detail view:** `autoUpdateSelectedComparison: Bool` (one,
  window-level) + the two loaded comparisons `comparison200`, `comparisonWide`.

## State machine

**Window opens (`.onAppear`):**
- If *either* UserDefaults per-set path is set → load those file(s) into their
  set(s); `autoUpdateSelectedComparison = false`.
- Else → `autoUpdateSelectedComparison = true`; for each set, if its in-memory
  current-session export URL is non-nil, load it; if nil, leave that set
  uncompared (point E will fill it when the export fires).

**`Compare…` (manual pick):** open the panel, parse the file, **route by
`probe_count`** (≈200 → `comparison200`; ≈4,435 → `comparisonWide`; mismatch →
friendly alert). Set `autoUpdateSelectedComparison = false`. Save that file's
path to the matching per-set UserDefaults key. The *other* set's comparison is
left untouched (so each set can be pinned to a different file independently).

**`Clear compare`:** clear `comparison200` + `comparisonWide`, clear *both*
UserDefaults keys, set `autoUpdateSelectedComparison = false`. (One button,
clears everything — matches today's single Clear.)

**Toggle `autoUpdateSelectedComparison`:**
- **false → true:** load the current-session export(s) for whichever sets have
  fired (same as the auto branch of window-open); **clear both UserDefaults
  keys** (so next launch starts in auto mode again).
- **true → false:** do nothing (leave the loaded comparisons as they are).

**New start-of-training export written (point E, observed via the `@Observable`
`SessionController`):** if `autoUpdateSelectedComparison == true`, load the
newly-written file into its set. (Both files are written ~together; each routes
to its own set.)

## Wide-set UI additions (this change)
- **OVERALL summary band** for the wide set — the same band the 200 set has
  (argmax / top-5 / avg-prob / avg-rank / NLL / pElo, plus the `cmp` row and
  `Δ live − cmp` row when `comparisonWide` is loaded). Parameterize the existing
  `overallSummaryBand` to take `(history, comparison)` and render it once per set.
- **Wide charts' comparison reference lines:** pass
  `comparisonWide?.overallSummary.meanNegLogProb` / `…mlePuzzleElo` into the wide
  `LichessProbeOverallTrendChart` (currently hard-nil).
- **Header `cmp:` label + model:** show the wide comparison's filename/model
  next to (or below) the 200's, or switch the header `cmp:` line to reflect the
  set the user is looking at. (Detail in implementation; simplest: show both.)
- *Deferred (not urgent, per request):* wide per-category section headers + the
  4,435 individual puzzle rows with CMP columns.

## Implementation touchpoints
1. **`LichessProbeExporter.exportLatest`** → return the written `URL?` (nil on
   skip/failure). Manual callers ignore it; the auto-export captures it.
2. **`SessionController.scheduleStartOfTrainingProbeExport`** → store the two
   returned URLs into the new `@Observable` properties (so the open window
   reacts via `.onChange`).
3. **`LichessProbeComparisonLoader`** → factor a `load(from url: URL) ->
   LichessProbeComparison?` (no panel; same parse + schema-version path); the
   existing `loadFromFile()` calls it after the panel. Add a `probeCount`
   accessor on the parsed result for routing.
4. **`LichessProbeDetailView`** → second comparison state + the toggle in the
   header + the `.onAppear` / `.onChange(currentSessionStart*URL)` logic +
   route-by-`probe_count` in Compare + per-set UserDefaults read/write. Render
   the wide OVERALL band; feed wide chart cmp lines.
5. **UserDefaults keys** — two string keys above.

## Proposed sub-decisions (flag if you'd change any)
- **Per-set UserDefaults** (vs one global): lets you pin each set to a different
  file; auto-mode is "neither key set."
- **One `Clear compare`** clears both sets (not per-set).
- **Manual `Compare…` routes by `probe_count`** rather than adding a second
  picker button.
- **One window-level auto toggle** governs both sets together.

## Validation
- Build compile-only (no relaunch during training).
- Manual: open window cold (no UserDefaults) with a session-start export present
  → both sets auto-compare to it, toggle shows ON. Pick a file via Compare →
  toggle flips OFF, routes to the right set, persists. Relaunch → reloads the
  pinned file, toggle OFF. Toggle ON → reloads session-start, clears the pin.
  Clear → both gone, pins cleared. Open window *before* the export fires → no
  comparison, then it appears when the export lands.
