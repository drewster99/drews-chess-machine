# Hybrid mixed precision: fp32 master weights + bf16 compute — implementation plan

Status: **IMPLEMENTED — compiles clean (0 errors). NOT runtime-tested** (training run
active; per workflow, no app/test runs during training). Runtime validation (the "money
test" + the momentum/round-trip tests) is pending the machine freeing up.
Branch: `bf16-trainer`.

**As-built note:** the checkpoint format/version bump in §5 turned out to be **unnecessary**.
Masters are parallel to `trainables + bnRunningStatsVariables` (same count as the existing
`base`), and the trainer payload is already fp32-on-disk — so `exportTrainerWeights` simply
emits the fp32 masters in place of `base` (identical count/layout), and `loadTrainerWeights`
restores the masters *and* rounds them into the bf16 working weights. No
`ModelCheckpointFile` / `trainerWeightCountV*` change; an old bf16-native trainer session
still loads (its bf16-rounded base seeds the master — no precision was there to lose). All
fp32 machinery is contained in `ChessTrainer` (plus one `syncMastersFromWorking()` call on
the promotion path in `SessionController+Arena`). New fp32 IO helpers
`ChessNetwork.readFloatsFP32` / `writeFloatsFP32`.

**Locked decisions:**
- **No toggle / no new enum.** This becomes the *canonical* meaning of a reduced-precision
  `ChessNetwork.dataType`: optimizer state is fp32, compute is `dataType`. Gated purely on
  `dataType != .float32` (for `.float32` the master ≡ the weights, today's path unchanged).
- **Persist the fp32 master** in the trainer session (correct resume).
- **Fully canonical (decision b):** the fp32-master treatment extends to **BN running
  mean/variance** as well, not just the gradient-updated trainables. Achieved trainer-side
  (see §2.1) so `ChessNetwork` needs no change.
- Consequence: the trainer-session format changes (velocity bf16→fp32, masters added), so
  the *current* run's autosaves won't resume under the new code. Champion `.dcmmodel` files
  are fp32-on-disk and unaffected. No migration (per project rules).

---

## 1. Why

The branch runs `ChessNetwork.dataType = .bFloat16` for trainer speed. The cost: **all
trainable weights are stored in bf16**, whose ULP is ~2⁻⁷ ≈ 0.8% of a weight's
magnitude. The SGD assign `weight_new = weight − lr·v_new` is computed and rounded in
bf16, so any per-step update smaller than ~0.4% of the weight **rounds away to a
bit-identical weight** — the weight is frozen. That's the same wall we hit in the
gradient-connectivity test and the reason we raised the default LR to 1e-3.

Standard mixed precision fixes this by keeping an **fp32 master copy** of the weights:
forward/backward run in bf16 (the speed win), but the optimizer accumulates updates
into the fp32 master, so sub-ULP updates add up over steps instead of vanishing. The
bf16 weights used by the forward pass are re-derived each step as `round(master)`.

Goal: a **toggle** so we can A/B "bf16-native" vs "fp32-master, bf16-compute" without
disturbing either.

---

## 2. Design: Option B (contained to `ChessTrainer`)

Two ways to do this:

- **Option A — fp32 weight variables, cast to bf16 inline at each use.** Touches every
  weight in `ChessNetwork`'s graph builders (stem, residualBlock, SE, heads, BN), and
  must be conditional so inference networks stay pure bf16. Wide blast radius.
- **Option B — keep `ChessNetwork`'s bf16 graph exactly as-is; add fp32 master + fp32
  velocity *in the trainer's optimizer subgraph*, and sync `bf16_weight = cast(master)`
  each step. (Chosen.)** All changes live in `ChessTrainer`; `ChessNetwork`, the
  inference networks, and the self-play path are untouched. The forward/backward still
  computes in bf16 (gradients are bf16-precision — exactly the standard recipe).

### Current update loop (`ChessTrainer.buildTrainingOps`, ~3416–3468), per trainable `W` (bf16):
```
clipped   = grad · clipScale
v_new     = μ · velocity + clipped                       // velocity: bf16
step      = lr · v_new  (+ lr · decayC · W   if shouldDecay)
W_new     = W − step                                      // bf16 assign — sub-ULP dies here
assign velocity = v_new ; assign W = W_new
```

### Hybrid update loop (when the toggle is on), per trainable `W` (bf16) with new master `M` (fp32):
```
g32       = cast(grad, fp32)
clipped32 = g32 · clipScale32                             // clip computed in fp32
v32_new   = μ32 · V32 + clipped32                         // velocity: fp32
step32    = lr32 · v32_new  (+ lr32 · decayC32 · M  if shouldDecay)   // decoupled WD on the MASTER
M_new     = M − step32                                    // fp32 accumulate — updates survive
assign V32 = v32_new
assign M   = M_new
assign W   = cast(M_new, bf16)                            // sync the bf16 working weight
```
**Scalar precision (important).** Today the optimizer scalars are fed bf16: their
placeholders are declared `dataType: dtype` and `buildFeeds` *narrows the Swift `Float`
to bf16 bits* (see `ChessTrainer.swift:1648`), so e.g. `lr = 1e-3` enters the graph as
`0.0009765625`. In the master regime the **update scalars `lr`, `weightDecayC`, `μ`,
`gradClipMaxNorm` must be fp32 placeholders fed the raw `Float`** — *not* a bf16
placeholder cast to fp32 (that only recovers the already-quantized bf16 value). The
gradient global-norm used for the clip stays fp32 end-to-end in this path (don't narrow
it back to bf16 before forming `clipScale`). The **loss-side** scalars (`entropyCoeff`,
policy/value loss weights, illegal-mass weight, label-smoothing, complementCE-enable)
stay bf16 — they shape the bf16 loss, whose gradient is bf16-precision regardless. This
means splitting the scalar feed path: fp32 descriptor + raw-`Float` write for the four
update scalars, bf16 for the rest. The `.float32` branch is today's path unchanged.

### 2.1 BN running stats (decision b) — same treatment, also trainer-side
BN running mean/var (`network.bnRunningStatsVariables`, bf16) are EMA-updated, not
gradient-updated, and their `0.99·old + 0.01·batch` EMA has the same (milder) bf16-ULP
accumulation problem. Treat them exactly like weights, entirely in the trainer:
- add an fp32 **running-stat master** per entry of `bnRunningStatsVariables`;
- build the EMA in fp32 using the per-layer batch stats `ChessNetwork` already exposes
  (`bnBatchMeanTensors` / `bnBatchVarTensors`, training mode):
  `runM_new = 0.99·runM + 0.01·cast(batchStat, fp32)` (fp32 constants);
- assign the fp32 running master, then sync `bf16_runningStat = cast(runM_new)`;
- **do not** append `network.bnRunningStatsAssignOps` (the network's bf16 EMA) in this
  mode — the trainer owns the EMA now; those ops become unused (harmless).
`ChessNetwork` is unchanged: running-stat *variables* stay bf16 (the inference-mode
normalize reads them directly, and they're what gets copied to champions); the fp32
running masters live only in the trainer, exactly parallel to the weight masters.
Indexing note: `bnRunningStatsVariables[2i]=mean(layer i)`, `[2i+1]=var(layer i)`;
`bnBatchMeanTensors[i]`/`bnBatchVarTensors[i]` are layer i — the trainer maps these.

### Master initialization / re-sync
The master must start equal to the current bf16 weights/stats (as fp32). The init data
isn't available at graph-build time, so add a one-time **`syncMastersFromWorking`** op set —
`assign(M[i], cast(W[i], .float32))` over both the trainable and running-stat masters —
run once:
- at trainer construction (after the graph builds),
- after any path that replaces working weights/stats wholesale where the master isn't
  itself being loaded: fresh fork (`loadBaseWeightsResetVelocity`), promotion's weight
  copy. (`loadTrainerWeights` loads the persisted masters directly — see §5 — then syncs
  working = cast(master), so it does not re-derive.)
This keeps masters and working copies consistent whenever state is replaced from outside
the optimizer.

---

## 3. No toggle — canonical, gated on `dataType`

No flag, no enum. The trainer's `buildTrainingOps` branches on
`ChessNetwork.dataType != .float32`:
- **reduced precision (`.bFloat16` / `.float16`):** build the fp32-master path (masters +
  fp32 velocity + fp32 EMA + sync-to-working). This is now the canonical bf16 trainer.
- **`.float32`:** today's path verbatim — working weights *are* fp32, so no separate
  master, no casts, no sync (the master would be a redundant copy).

So "bf16 mode" canonically means "bf16 compute + fp32 master." One switch (`dataType`),
no second axis.

---

## 4. Files / touchpoints

All in **`Training/ChessTrainer.swift`** (no toggle plumbing, no `ChessNetwork` change):

- `buildTrainingOps`, when `dataType != .float32`:
  - allocate fp32 **weight masters** parallel to `trainableVariables`, fp32 **running-stat
    masters** parallel to `bnRunningStatsVariables`, and make `velocities` fp32;
  - build the fp32 SGD update (§2) writing weight-master + velocity, then
    `bf16_weight = cast(weight-master)`;
  - build the fp32 EMA (§2.1) writing running-master, then
    `bf16_runningStat = cast(running-master)`; skip `network.bnRunningStatsAssignOps`;
  - build `syncMastersFromWorking` assigns (`master = cast(working, fp32)`).
  - **Split the scalar feed path** (§2): `lr`, `weightDecayC`, `μ`, `gradClipMaxNorm` get
    fp32 placeholders + fp32 NDArrays fed the raw `Float` (so they're exact); the loss-side
    scalars stay bf16. Keep the gradient global-norm fp32 through the clip computation.
- Velocity load/read/write infra (`readVelocityValues`/`writeVelocityValues`/`velocityLoad*`)
  becomes fp32; add the analogous master read/write/load infra.
- Run `syncMastersFromWorking` at end of `init` and after fresh-fork / promotion weight
  copies (not after `loadTrainerWeights`, which loads masters directly).
- Diagnostics already widen reductions to fp32 (landed) — grad/velocity norms stay correct.
- **Checkpoint** (persist the master — locked): `exportTrainerWeights` emits the fp32
  masters (weights + running stats) + fp32 velocity; `loadTrainerWeights` loads them and
  syncs working = cast(master). Bump the trainer payload version + `trainerWeightCountV*`
  guards. `ModelCheckpointFile` already speaks fp32 on disk, so no encoder change beyond
  the count/version.
- **No changes** to `ChessNetwork`, inference networks, self-play, arena, or the
  `.dcmmodel` champion format — masters are trainer-only; champions/candidates stay bf16
  (a champion is `cast(master)` = the working weights, copied as today).

---

## 5. Checkpoint handling (LOCKED: persist the master)

Trainer sessions currently save `base (bf16 weights+BN) + velocity`. The new trainer
session payload is the **fp32 masters + fp32 velocity**:
- weight masters (parallel to `trainableVariables`),
- running-stat masters (parallel to `bnRunningStatsVariables`),
- velocity (parallel to `trainableVariables`).

On resume, load the masters and sync working = `cast(master)`. This restores the exact
accumulated master — without it, every periodic / post-promotion autosave + resume would
discard the sub-ULP accumulation this feature exists for. `.dcmmodel` champion/candidate
saves are unaffected (always fp32 on disk; a champion is `cast(master)` = the working
weights, exported as today). Bump the trainer payload version + `trainerWeightCountV*`
guards; an old bf16-native trainer session (bf16 velocity, no masters) fails its count
guard and cleanly refuses to resume (no migration).

---

## 6. Memory / perf expectations (to measure once it runs)

- **Memory:** +1 fp32 master (2× a bf16 weight) and velocity goes bf16→fp32 (2×) for the
  trainable set (~3.9M params). Roughly +~30 MB of optimizer state — negligible vs the
  ~10–18 GB activation footprint at batch 4096.
- **Speed:** per step adds N elementwise casts (grad→fp32 once each, master→bf16 once
  each) and does the optimizer math in fp32. The matmuls/convs (the GPU cost) stay bf16,
  so step time should be ~unchanged; **this must be measured** (`gpu=`/`step=` in `[STATS]`)
  since measuring bf16 trainer speed is the branch's whole purpose.

---

## 7. Risks (why this is not safe to ship blind)

- **MPSGraph dtype matching is a runtime failure, not a compile error.** Every op in the
  master path must be consistently fp32; the only bf16↔fp32 boundaries are `cast(grad)→fp32`
  (in) and `cast(master)→bf16` (out). A stray bf16 scalar fed into an fp32 op, or assigning
  a bf16 tensor to an fp32 variable, crashes at `graph.run`. Needs a real run to shake out.
- **`graph.assign` ordering:** `W = cast(M_new)` must consume the freshly-computed `M_new`
  tensor (value-typed reference), matching the existing "assign doesn't invalidate the
  symbolic tensor" reasoning in the current loop.
- **Master/weight divergence** if any weight-replacing path forgets to re-sync the master
  (§2) — would silently train from a stale master.

---

## 8. Validation (deferred until training frees up — no app/test runs during training)

1. **Compile-build** (allowed now): clean build (the fp32-master path is built whenever
   `dataType != .float32`, i.e. the current bf16 setting, so it's exercised by default).
2. **The money test:** a connectivity-style probe at a *small* LR (e.g. 1e-5) where pure
   bf16 froze — assert weights actually move across a few steps (masters accumulate). The
   direct proof the feature works. (Also confirms the bf16-ULP gradient-connectivity test
   could drop its lr=10/wd=0 workaround — though leave that as-is unless we choose to.)
3. **`.float32` regression:** set `dataType = .float32` temporarily — must take the
   unchanged today's-path branch (no masters); existing tests green.
4. **Numerical:** one step vs a hand-computed fp32 update on a tiny net agree.
5. **Throughput:** compare `step=`/`gpu=` and RSS vs the pre-change bf16-native build over
   ~200 steps (the branch's whole point is bf16 trainer speed).
6. **Save/resume:** resume reproduces the masters bit-exactly; an old bf16-native session
   cleanly refuses (count guard).
7. **BN running stats:** confirm `pD`/value diagnostics and inference-vs-training BN
   agreement still behave (the EMA now accumulates in fp32, synced to bf16).

---

## 9. Decisions — RESOLVED

1. **No toggle, no enum** — canonical for reduced-precision `dataType` (gated on
   `dataType != .float32`). See §3.
2. **Persist the fp32 master** in the trainer session. See §5.
3. **Fully canonical (b):** fp32 masters cover BN running stats too, trainer-side. See §2.1.
