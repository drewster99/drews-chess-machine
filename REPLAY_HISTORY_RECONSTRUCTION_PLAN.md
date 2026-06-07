# Replay-buffer history reconstruction — store once, stack on read

Status: **PLANNED, not started.** Branch: `safetensors-storage`. Created 2026-06-06.
Do not implement until explicitly told to start.

## Goal

Stop storing the full 10-frame stack in every replay-buffer slot for `full10ply200`.
Store each ply **once** as a single 20-plane frame; **reconstruct** the 200-plane
perspective-relative network input at *trainer sample time* by stacking the sampled ply
with its 9 contiguous priors. Cuts replay RAM ~**10×** (51.2 KB/pos → 5.12 KB/pos, i.e.
back to basic20 size) because the redundancy — each ply is currently re-stored as a
history frame of ~10 neighboring slots — is removed.

## Scope / non-goals

- Touches ONLY the trainer's replay buffer **write** (self-play record) and **read**
  (sample) paths. The network input contract is unchanged: the trainer still feeds 200
  perspective-relative planes — it just *assembles* them from stored single frames.
- **Unchanged:** inference, arena, self-play *move selection*, BN warmup — they encode on
  the fly (bake-in) and never read the replay buffer.
- **Unchanged:** `basic20` / `basic30` storage — already single-frame, no reconstruction.
- Distinct from `SELFPLAY_CORPUS_PLAN.md` (that's the *persisted* corpus; this is the
  in-memory buffer). Same philosophy, different store.

## Key decision — perspective (DECIDED: relative storage + flip-odd-frames)

**The problem.** A frame must appear in **ply-N's mover's** perspective. N and N-1 have
*opposite* movers, so a ply stored in its own mover's perspective is the *wrong* perspective
for half the stacks that reference it. (Proven by our test case: `after-1.e4` stored as its
own position is Black-POV; position `after e4 e5` needs it as a White-POV history frame.) So
you cannot byte-copy stored frames into a stack unchanged.

**Decision.** Keep storing each ply exactly as the encoder already produces it — in **its own
mover's perspective** (the normal single-position encoding; literally frame 0 of the 200-plane
stack self-play already builds for inference, so we store that 1,280-float slice — no extra
encode, stored bytes unchanged from a normal encode). The buffer already stores positions this
way today (each position from its mover's POV, alternating White-POV / Black-POV).

At reconstruction, fix perspective with a **fixed pattern: flip the odd-indexed frames**
(1, 3, 5, 7, 9) — independent of N's color, because an odd number of plies back is always the
opposite mover from N. Even frames (including frame 0 = the sampled position) are already in
N's perspective and are copied as-is.

"Flip" = the encoder's full perspective transform, applied per odd frame:
- **vertical row-reversal** of every plane, and
- **swap the my/opp plane pairs**: pieces `0–5 ↔ 6–11`, castling `12–13 ↔ 14–15`.

EP (16) row-flips as a positional square (covered by the row-reversal); clock (17) and
repetition (18/19) are broadcast/scalar and perspective-independent (flip is a no-op); zeroed
frames flip to themselves. Applying this to a mover-relative frame yields **bit-exactly** the
opposite-mover-relative encoding — so the reconstructed stack equals the current bake-in
output, position for position.

**Rejected alternatives:** (a) store *absolute* (white-POV) + transform the whole block when N
is black — works, but needs an extra encode at write and diverges from the buffer's existing
relative storage. (b) flip inside the GPU graph — changes the network's input convention
*everywhere* (inference, arena, self-play, BN warmup), far outside this optimization's blast
radius. Relative storage + flip-odd-frames keeps the change local to the trainer and reuses
what's already stored.

## Storage format

- **Per-game contiguous, reverse-chronological blocks.** Within a game block, address
  ascends as ply descends: `[ply_K, ply_{K-1}, …, ply_0]`. A **forward** copy of 10 slots
  from the sampled ply's slot then yields `[N, N-1, …, N-9]` directly in stack order — no
  reordering.
- Each slot stores one **20-plane mover-relative frame** (the normal single-position encoding =
  frame 0 of the inference stack) + the existing metadata columns
  (`workerGameId`, `plyIndex`, move, outcome, tau, material, …).
- Ring buffer; games written as contiguous units; **wrap is fine** — a wrap-spanning
  reconstruction is just ≤2 memcpys.

## Write path

- Change `ActiveGame.flush` from **per-side** to **per-game ply-ordered** (reverse-chrono),
  one contiguous block per game. (Flush is already a bulk per-game op, so all plies are in
  hand — just emit them ply-descending.)
- Self-play records the single **mover-relative** frame per ply — the first 1,280 floats
  (frame 0) of the 200-plane inference stack it already built for move selection. No extra
  encode; it stores a slice instead of the full stack.

## Read / reconstruction (trainer sample)

For each sampled position N:
1. Forward-copy up to 10 slots from N's slot (≤2 memcpys if wrapping).
2. Validate each candidate slot `k` against expected `gameId == N.gameId` **and**
   `ply == N.ply - k`. On the first mismatch (game boundary / eviction / game-start
   underflow), **zero from that frame through frame 9**.
3. Flip the **odd-indexed frames** (1, 3, 5, 7, 9) — fixed pattern, independent of N's color:
   per odd frame, vertical row-reversal of every plane + swap the my/opp plane pairs
   (pieces 0–5↔6–11, castling 12–13↔14–15). Even frames (incl. frame 0) are used as stored.
4. Result is **bit-identical** to what the current bake-in produces for position N.

## Concurrency (DECIDED: reuse the existing buffer lock)

`ReplayBuffer` is already guarded by one `OSAllocatedUnfairLock` (line 52); `append()`
(self-play flush) and `sample()` (trainer) both take it, so they are mutually exclusive —
that's why there are no torn reads today. **Do the cross-slot gather inside the existing
`sample()` lock:** under the lock, copy the ≤10 raw frames into private staging (≤2 memcpys)
+ the `(gameId, ply)` validation reads; then **release the lock and apply the
flip-odd-frames transform** on the copied-out private data. No new mechanism, guaranteed
correct.

Lock-hold barely changes: today `sample()` copies one 200-plane stack (12,800 floats) per
position under the lock; after, it gathers ≤10 × 1,280 = ≤12,800 floats — the **same byte
volume**, just from 10 slots instead of 1.

*Only if* profiling later shows the gather's lock-hold starving self-play: revisit a
lock-free path (per-slot generation/epoch counter re-checked after the gather; treat a
changed generation as evicted → zero from there). Not needed for v1.

## Metadata / sampling

- Needs `(gameId, plyIndex)` per slot — **already stored** (`workerGameId`, `plyIndex`).
  Mover derived from ply parity (ply 0 = White to move).
- Stratified sampling is unchanged: the sampler still selects individual positions by its
  constraints; only the per-position *materialization* (one memcpy → gather+transform)
  changes.

## Validation / tests

- **Reconstruction == bake-in (bit-exact):** play a game, store single mover-relative frames,
  reconstruct each position (gather + flip-odd-frames), assert equality with
  `BoardEncoder.encode(state, history:…, encoding: .full10ply200)` for that ply — for both
  white- and black-to-move positions.
- **Boundary / eviction zeroing:** a position whose priors are absent/overwritten →
  correct zero-padding from the right frame onward.
- **Game-start zeroing:** first ≤9 plies reconstruct with the right number of zeroed frames.
- **Concurrency stress:** writes during gather never yield a torn frame (generation
  re-check catches it).
- **Regression:** `basic20` / `basic30` storage + sampling unchanged and green.

## Risks / cost

- Moves work from storage to the sample hot path: ≤2 memcpys + flipping the 5 odd frames
  per sampled position. Net win on RAM, small cost on sample throughput.
- The **per-side → per-game write-path rework** is the riskiest change — it touches
  `ActiveGame.flush` and the buffer's column layout; needs care to keep all the existing
  metadata columns consistent.

## Phases

1. **Single mover-relative frame storage** (store frame 0 of the inference stack — a 1,280-
   float slice) + the per-game block layout scaffolding — no behavior change yet (full10ply200
   still bakes in).
2. **Per-game ply-ordered write path** + reverse-chrono block emission.
3. **Trainer reconstruction**: gather + `(gameId, ply)` validate + zero + perspective
   transform; bit-exact test vs bake-in.
4. **Concurrency**: per-slot generation tags + post-gather re-check.
5. **Cut over** full10ply200 from bake-in to store-once; drop the 200-plane storage stride;
   full test pass + a smoke training run.

## Open items to confirm before implementing

- **Perspective approach** — RESOLVED (2026-06-06): relative storage (store the mover-relative
  frame 0 of the inference stack) + flip-odd-frames at read. See "Key decision".
- **Concurrency mechanism** — RESOLVED (2026-06-06): reuse the existing buffer
  `OSAllocatedUnfairLock` (gather under `sample()`'s lock, flip after release). Generation
  tags deferred to "only if profiling demands". See "Concurrency".
- **Cutover** — keep bake-in behind a flag during transition, or replace outright? *(OPEN —
  the one remaining decision)*
