# Plan: `full10Ply10Reps210` input encoding

Status: **SHIPPED** (2026-06-13 audit). `InputEncoding.full10Ply10Reps210` (210
planes, 10-plane repetition tail) is implemented in `NetworkArchitecture.swift`
and threaded through the encoder / replay / build paths.

Branch: `safetensors-storage`. Follow-up to Experiment 2 (`full10ply200`). The Experiment-2
finding: dropping the explicit repetition planes is one of two suspects for the slower bootstrap.
This adds them back **without** changing `full10ply200`.

## 1. Goal

New `InputEncoding` case: **`full10Ply10Reps210`** = `full10ply200`'s 200 planes, plus the
**10 temporal-repetition planes appended at 200–209**, identical in meaning to `basic30`'s
planes 20–29. 210 planes total. `full10ply200` is left untouched.

## 2. Exact basic30 rep semantics (verified in code, must match EXACTLY)

Source: `ChessGameEngine.applyMove` + `PositionKey` + `BoardEncoder.writeBasicBlock`.

- **`PositionKey`** = `board` + `currentPlayer` (STM) + 4 castling flags + `enPassantSquareIndex`.
  **Not** halfmove clock.
- **Window** `recentPositionKeys`, capped at **10** (`recentPositionKeyWindow`). After a move,
  `recentPositionKeys[i]` = `PositionKey` of the position **(i+1) plies before** the new state.
- **Cleared on any irreversible move** (pawn move / capture → `halfmoveClock == 0`): window emptied,
  rebuilds from empty.
- **Mask:** bit `i` set iff `recentPositionKeys[i] == key(current)`. Encoder plane `20+i` = bit `i`.
  The encoder only **reads** `state.recentRepetitionMask` — it does not compute it.
- **Consequence:** a repeat needs the same STM, so only **even ply-distances** (k = i+1 even, i odd)
  can ever be set. Planes for odd ply-distances are always 0. (basic30 already behaves this way.)

## 3. Decision: do NOT store the reps; recompute on the consumer side

Storing reps in the buffer would change the per-position frame format and the gather hot path —
"a total mess" and not worth it. Instead the reps are produced at the two consumer sites:

- **Inference path** (self-play / arena move selection): `BoardEncoder.encode` has the live
  `GameState`, so it reads `state.recentRepetitionMask` and fills planes 200–209 directly —
  exactly as `basic30` does. Trivial, exact.
- **Training path** (replay buffer `sample` → GPU staging): no `GameState`. Recompute the mask from
  the stored prior frames (Section 5). The replay buffer's stored format stays **byte-identical to
  `full10ply200`** (20-plane frames) — no new array, no `.dcmsession` back-compat work.

## 4. Why recompute == basic30 exactly (equivalence argument)

- **Board equality subsumes the window-clear.** A pawn move or capture makes the prior board
  unreachable, so a true `PositionKey` repeat can never span an irreversible move. Comparing boards
  therefore yields the same result as the engine's cleared window — without tracking the clear.
- **Stored planes 0–16 at even distance ⟺ `PositionKey` equality.** planes 0–11 = board, 12–15 =
  castling, 16 = EP; STM is implicit via perspective + ply parity. Even-distance priors are stored in
  the **same perspective** as the current frame (same mover color), so it's a raw float compare — no
  flip needed. Exclude planes 17 (halfmove), 18–19 (rep counts) from the compare.
- **Window depth = 10, stack depth = 9.** `full10ply200` gathers plies N…N-9 (frames 0–9). The
  deepest rep (plane 209, i=9) is **10 plies ago = N-10**, one deeper than the stack. So the recompute
  must read priors **k = 1…10 directly from the buffer**, not only the reconstructed stack.
- **Ring eviction caveat.** If a prior slot was overwritten, the recompute treats it as absent (mask
  bit 0), under-counting vs the push-time mask. This matches `full10ply200`'s existing behavior for
  its history stack (evicted priors are zeroed), and is negligible: a position and its 10 immediate
  priors are pushed consecutively into a ~1M-slot ring, so if N is live its priors are live except in
  a ≤10-slot window at the eviction boundary. Accepted, by-design, same as the current encoding.

## 5. Recompute algorithm (training path), exact

For each sampled position at slot `srcIndex`, ply `basePly = plyIndex[srcIndex]`, game
`gameId = workerGameId[srcIndex]`, after `reconstructHistoryStack` has written planes 0–199:

1. Zero planes 200–209 of the reconstructed board (10 × 64 floats).
2. For `k in [2, 4, 6, 8, 10]` (even distances only):
   - `priorSlot = (srcIndex + k) % capacity`.
   - valid iff `workerGameId[priorSlot] == gameId` **and** `basePly - k >= 0` **and**
     `plyIndex[priorSlot] == basePly - k` (same validity test the gather uses).
   - if valid and `memcmp(boardStorage + srcIndex*stride, boardStorage + priorSlot*stride, 17*64*4) == 0`
     (planes 0–16 equal) → set plane `200 + (k-1)` to all-1 (64 floats).
   - odd-distance planes (200, 202, 204, 206, 208) stay 0.

`stride = floatsPerBoard = planesPerFrame*64 = 1280`. Cost: 5 memcmps (≤1088 floats each) per sampled
position. Cheap; the gather already touches this memory under the same lock.

## 6. Touchpoints

1. **`Network/NetworkArchitecture.swift` — `InputEncoding`:**
   - Add `case full10Ply10Reps210`.
   - `planeGroups`: the `full10ply200` groups (0–199) **+** `PlaneGroup(200...209, <basic30 plane-20–29 text>)`.
   - `historyFrameCount = 10`, `planesPerFrame = 20`, `planeCount = 210` (derived).
   - **Generalize the frame invariant.** Add `tailPlaneCount` (planes beyond the stack;
     `= 0` for all existing cases, `10` here) so `historyFrameCount*planesPerFrame + tailPlaneCount == planeCount`.
   - Fix `planeDescription` multi-frame branch to also render the appended rep tail.
2. **`Encoding/BoardEncoder.swift` — `encode(_:into:encoding:)` switch:** add `.full10Ply10Reps210`
   case = the `full10ply200` body, then write planes 200–209 from `state.recentRepetitionMask`
   (reuse the basic30 loop: `for i in 0..<10 where (mask>>i)&1==1 { fillPlane(base, plane: 200+i) }`).
   This is the inference path. (History frames carry no reps, same as full10ply200.)
3. **`Training/ReplayBuffer.swift`:** `storedStride` unchanged (planesPerFrame*64 = 1280);
   `reconstructedStride = planeCount*64 = 13440`. In `emit()`, after `reconstructHistoryStack`,
   for this encoding run the Section-5 rep pass. Gate it so other encodings are byte-identical.
4. **`Network/ChessNetwork.swift` / `ChessMPSNetwork.swift`:** stem input depth = `planeCount` (210) —
   verify it flows from `arch.inputEncoding.planeCount` (it does at `ChessNetwork.swift:149`); confirm
   no other site hardcodes 200/30. `historyDepth = historyFrameCount-1 = 9` (unchanged from full10ply200).
5. **Tests (`DrewsChessMachineTests`):**
   - Generalize `testFrameStructureInvariants` to `frames*perFrame + tail == planeCount`.
   - Add encoder-fills-exactly-these-planes coverage for `.full10Ply10Reps210` (extend the
     `BoardEncoderTests` plane-range assertion).
   - **Key correctness test:** build a short game containing a real repetition (and one across an
     irreversible move), push frames into a `ReplayBuffer(inputEncoding: .full10Ply10Reps210)`,
     `sample`, and assert the reconstructed planes 200–209 equal `basic30`'s planes 20–29 for the
     same positions (i.e. recompute == `state.recentRepetitionMask`). This is the EXACT-match gate.
   - (XCTest can only be **run** when no training session is live — build-compile during Exp 2.)
6. **`Views/Board/TensorChannelNames.swift` + launch precondition:** default arch is `basic30` (30
   planes), so the `DrewsChessMachineApp` precondition is unaffected. The board-channel **visualizer**
   for a 210-plane model is a separate concern — confirm it degrades gracefully or is out of scope.
7. **Build-New-Model UI / `Preset`:** confirm the encoding is independently selectable in the build
   UI; if it's only reachable via a `Preset`, add a preset that pairs the Exp-2 tower with
   `full10Ply10Reps210` so the next run is a clean one-variable change vs Experiment 2.

## 6a. Non-regression / isolation guarantees (hard requirement)

`basic20`, `basic30`, `full10ply200` must behave **byte-for-byte identically** after this change.
How each shared touch stays inert for them:

- **`InputEncoding`**: only an added `case` + new `switch` arms. `tailPlaneCount` returns `0` for every
  existing case (asserted in a test for all `allCases`), so derived counts and `planeDescription` for
  them are unchanged. No existing arm is edited.
- **`BoardEncoder.encode`**: a new `case` arm only. The `full10ply200` / `basic30` / `basic20` arms are
  not touched, so their emitted floats are identical.
- **`ReplayBuffer`**: the rep pass is gated on `inputEncoding == .full10Ply10Reps210` (or
  `tailPlaneCount > 0`). For every other encoding `reconstructedStride == historyFrameCount*storedStride`
  and the `emit()` path runs exactly as today — no extra reads, no extra writes.
- **Network/stem**: depth already flows from `planeCount`; no constant edited, so 30/200-plane models
  build and run unchanged.
- **Shared invariant generalization** is additive (`+ tailPlaneCount`, which is 0 for them).

**Verification of this bar requires running the existing `basic30` / `full10ply200` / encoder tests and
confirming they pass unchanged.** The XCTest suite cannot be *run* while Experiment 2 is training. So
full non-regression proof is gated on a training pause; until then I can only compile-verify (editing
source and building to DerivedData does **not** disturb the running Release process).

## 7. Validation — all satisfied (validated 2026-06-23; `Full10Ply10Reps210EncodingTests` pass)

- [x] Exact-match test (5e) passes: recompute-from-frames == basic30 mask, including the
      across-irreversible-move case and a true 4-/2-ply repeat.
- [x] Generalized frame invariant holds for all encodings.
- [x] Encoder-fills-planes test green for `.full10Ply10Reps210`.
- [x] Inference path (encode with GameState) and training path (recompute) produce identical
      planes 200–209 for the same position — assert in a test.
- [x] Build clean (compile-only while Exp 2 trains; full test run when paused).
- [x] No change to `full10ply200` / `basic30` behavior (regression: their tests still pass byte-for-byte).

## 8. Risks / open items

- Ring-eviction under-count (Section 4) — accepted, matches existing full10ply200 behavior.
- The `tailPlaneCount` generalization touches a shared invariant — keep the change minimal and assert
  `tailPlaneCount == 0` for every pre-existing case so nothing else shifts.
- Per-step cost of the rep pass is small but real; measure `[STATS]` step time vs Exp 2 after wiring.
