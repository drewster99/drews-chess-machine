# full10ply200 input encoding + history-dropout — implementation plan

Status: **Phases 1–4 SHIPPED (2026-06-23 audit) — tests passing.** The
`Full10Ply200EncodingTests` (in `DrewsChessMachineTests`) have since RUN and PASS, closing
the validation gate that was previously deferred behind a live training session. Phase 5
(history dropout) remains correctly **DEFERRED → ROADMAP.md**. Branch: `safetensors-storage`.
Created 2026-06-06.

Phase 3 scope note: in addition to self-play / inference / BN-warmup, the **arena**
(`TickTournamentDriver`) also encodes per-ply and was threaded with
`g.engine.recentStates` — otherwise a full10ply200 arena would evaluate on
history-less inputs while the nets were trained with history. The
probe/diagnostic/FEN encode paths (lichess probe, candidate probe, tactical probe,
UCI, Engine Diagnostics) are intentionally left history-less: they evaluate
isolated positions that genuinely have no game trajectory.

## Goal

Add a third `InputEncoding` case, `full10ply200`, alongside the existing `basic20` /
`basic30`. It stacks the last 10 game positions into a **200-plane** input tensor: the
current 20-plane `basic20` block repeated 10× for plies N, N-1, … N-9. No new "marker"
or presence planes. This is an *additional* option, not a replacement, and is selectable
when building a new network (not a named preset).

A companion training-time augmentation, `historyDropoutProbability` (feed the net
only frame N for some fraction of sampled positions, to regularize against
over-reliance on history), is **DEFERRED to future work** — tracked in
`ROADMAP.md` under "Future improvements". The current scope is **Phases 1–4
only**; Phase 5 below is retained for when it's picked up.

## Locked design decisions

- **200 planes = the `basic20` block (planes 0–19) × 10 frames.** Frame N → planes 0–19,
  N-1 → 20–39, …, N-9 → 180–199. Each frame carries the full basic20 set, including its
  own repetition planes 18/19. No temporal-repetition block (basic30's 20–29) inside any
  frame; no marker/presence planes.
- **Every frame is encoded from the ply-N mover's perspective** — orientation (flip),
  my-vs-opp piece assignment (planes 0–5 vs 6–11), and my-vs-opp castling (12–15) are all
  keyed to N's mover, *not* each frame's own `currentPlayer`. The invariant is uniform,
  not "never flip odd frames": flip iff N's mover is Black, applied identically to all 10
  frames. So with White at N the odd frames (N-1, N-3…) are not flipped; with Black at N
  they are. EP/clock/repetition (planes 16–19) carry each frame's own values.
- **Absent frames (game start) are all-zero.** No marker; a real frame always has two
  kings (planes 5 & 11), so all-zero piece planes are an unambiguous "no history here"
  signal. The encoder's leading zero-clear handles this for free.
- **No retroactive clearing.** History frames show the positions exactly as they were at
  the time, across irreversible moves. The per-frame planes 18/19 reflect each ply's
  own as-of-then repetition state (already baked into the stored `GameState`).
- **History is baked in at encode time, never at record time.** The replay buffer stores
  the encoded 200-plane tensor; the raw full 200 planes (real history) are what's stored.

## Architecture-independence note

The self-play *corpus* (see `SELFPLAY_CORPUS_PLAN.md`) is encoding-agnostic and serves
every architecture — re-encoding produces basic20/basic30/full10ply200 alike. This plan
is only the encoding + its consumers. Nothing here changes what self-play must *emit*.

---

## Phase 1 — Engine non-clearing state window — DONE

`ChessGameEngine` today retains `recentPositionKeys: [PositionKey]` (window 10, **cleared
on irreversible move**) — hashes only, not full states, and wrong clearing semantics for
this feature.

- Add `private(set) var recentStates: [GameState]`, cap 9, index 0 = position 1 ply ago
  (N-1), index 8 = N-9.
- In `applyMoveAndAdvance`, prepend the pre-move `state` and truncate to 9 — but **skip
  the irreversible-clear branch** that `recentPositionKeys` uses. The stored `GameState`
  already carries its as-of-then `repetitionCount`/`recentRepetitionMask`.
- Document the deliberate clearing difference vs `recentPositionKeys` in a comment.
- Window size 9 is coupled to the deepest history-stacking encoding (full10ply200 needs
  current + 9 prior). Document that coupling at the declaration.

## Phase 2 — Encoder refactor + `InputEncoding.full10ply200` — DONE

In `BoardEncoder.swift`, factor the current single-frame body into a private writer:

```
writeBasicBlock(_ state: GameState, perspective: PieceColor, planeBase: Int,
                includeTemporalRepetition: Bool, into base: UnsafeMutablePointer<Float>)
```

Replace every use of `state.currentPlayer` in the flip / my-vs-opp / castling / EP logic
with `perspective`. EP/clock/repetition use the frame's own values.

New entry point:

```
encode(_ current: GameState, history: [GameState], perspective: PieceColor,
       encoding: InputEncoding, into: UnsafeMutableBufferPointer<Float>)
```

- `basic20`: one call, `perspective = current.currentPlayer`, planeBase 0,
  `includeTemporalRepetition: false`.
- `basic30`: one call, planeBase 0, `includeTemporalRepetition: true`.
- `full10ply200`: 10 calls, **all** with `perspective = current.currentPlayer`,
  planeBase `f*20`; frame 0 = `current`, frames 1…9 = `history[f-1]` when present;
  `includeTemporalRepetition: false`. Absent frames are simply not written (stay zero).

Keep a single-`GameState` convenience overload (empty history, `perspective =
current.currentPlayer`) for non-hot-path callers (tests, Forward-Pass demo). It is valid
only for single-frame encodings; for full10ply200 it would yield only frame N — multi-frame
encodings MUST go through the history-aware path. Assert/guard this.

In `NetworkArchitecture.swift`, add `case full10ply200` with `planeGroups` rendering the
10 frame-blocks (basic20 groups offset by `f*20`, labeled by frame). `planeCount` derives
to 200; everything downstream (stem depth, buffer stride, weight plan, summaries) follows
automatically.

## Phase 3 — Thread session encoding + history through the hot paths — DONE

Finishes the "Phase C" debt the BoardEncoder comments flag (call sites currently hardcode
the `basic30` default stride):

- **Self-play** — `BatchedSelfPlayDriver.swift:343`: `boardFloats =
  BoardEncoder.tensorLength(for: arch.inputEncoding)`; encode each game with
  `g.engine.recentStates`. Record path (`ActiveGame.recordPly`/`flush`) is otherwise
  unchanged — it copies the wider tensor.
- **Inference** — `MPSChessPlayer.swift:333`: add a `recentStates` argument to the
  choose callback, sourced from `engine.recentStates` in `ChessMachine.runGameLoop`; size
  the encode scratch from the session encoding; encode with history. The
  `MoveEvaluationSource` is unchanged (it takes already-encoded floats). Note: a player
  only sees its own turns, so the consecutive-ply history can only come from the engine.
- **BN warmup** — `ChessMPSNetwork.swift:135` `calibrateBNRunningStats`: the random walk
  must maintain a rolling 9-state window so it feeds 200-plane inputs (else stem-shape
  mismatch).
- Remove `BoardEncoder.tensorLength` (static basic30 default) from hot-path use; thread
  the actual session encoding everywhere.

`decodeSynthetic` needs **no change** — frame N's pieces/castling/EP/clock stay at planes
0–17 (used by `ChessTrainer.legalMassSnapshot`).

## Phase 4 — ReplayBuffer stride + Build-New-Network UI — DONE

- Buffer stride already derives from `arch.inputPlanes` (`ChessTrainer.swift:4111`); verify
  200 planes (12800 floats/position) flows end-to-end. Surface replay-capacity guidance
  given the RAM cost below.
- Add `full10ply200` to the input-encoding picker on the Build-New-Network screen (a
  selectable option, not a preset).

## Phase 5 — `historyDropoutProbability` augmentation — DEFERRED (see ROADMAP.md)

> **Deferred to future work.** Not part of the current full10ply200 scope
> (Phases 1–4). Spec retained below for when it's picked up.


Sometimes train on frame-N-only (history zeroed) to regularize against history
over-reliance. Binary (not random-depth): with probability *p*, zero frames N-1…N-9.

- New `@TrainingParameter historyDropoutProbability` ∈ [0,1], default 0, `liveTunable:
  true` (re-read each step, not snapshotted). Walk the full CLAUDE.md add-a-parameter
  checklist (declare + `allKeys`; singleton wiring; parameters.json round-trip;
  `.dcmsession` save/load with `[RESUME-PARAM]` block; results.json/recorder if relevant;
  `[STATS]`/`[SAMPLER]` log visibility; UI in `TrainingSettingsPopover`; live-tunability).
- **Applied at training sample time, not record time.** In `ChessTrainer`, right after
  `replayBuffer.sample(...)` fills the staging boards and **before** GPU upload: for each
  of the `batchSize` positions, draw uniform [0,1); if `< p`, `vDSP_vclr` that position's
  history suffix — floats `[20*64 ..< planeCount*64]` within its slice (one contiguous
  clear). Real full history stays in the buffer; same position varies across epochs.
- **No-op for single-frame encodings**: with basic20/basic30 the suffix range is empty,
  so the parameter does nothing — harmless when full10ply200 is not in use.
- Self-play, arena, and Play Game always feed real full history. Dropout is
  training-augmentation only ("train with dropout, play clean").

---

## Validation

- **Encoder-fill unit test for full10ply200** (`DrewsChessMachineTests`): asserts the
  encoder writes exactly the 200-plane ranges per `planeGroups`; odd-frame perspective
  correctness (a Black-to-move history frame shows *our* pieces in planes 0–5, correctly
  oriented); absent frames are all-zero; a hand-built repetition sequence lights the right
  per-frame planes 18/19.
- **basic20 / basic30 are behavior-preserving** across the encoder refactor — existing
  encoder tests stay green without modification (regression guard).
- **Round-trip**: build a full10ply200 net → run a self-play game (Engine Diagnostics) →
  confirm stem accepts 200 planes, buffer stride = 12800, a training step runs.
- **Parameter round-trips** for `historyDropoutProbability`: `--show-default-parameters`,
  `--create-parameters-file` reload, and `.dcmsession` save/load.
- All existing tests pass at the end of each phase, unmodified.

## Costs / risks (accepted, on the record)

- Replay stride 1920 → **12800 floats = 51.2 KB/position** (6.65×). At a 500k-position
  buffer that's ~25.6 GB just for boards (vs ~3.8 GB now). Replay capacity will likely
  need to drop ~6.6× to hold the same RAM.
- Stem conv input depth 30 → 200: ~1.25M weights on the 7×7 preset (vs 188K), ~230K on
  3×3 (vs 34.5K).
- Per-ply encode cost is ~10× (10 frames). Cheap vDSP fills, but on the hot path.
