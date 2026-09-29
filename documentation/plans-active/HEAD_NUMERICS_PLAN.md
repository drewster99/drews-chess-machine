# Head numerics plan: stop the bf16 head-offset damage

Status (2026-09-28): Phase 0 implemented (`2194f1d`). Phases 1 and 2 implemented together in one commit (see CHANGELOG); their validation runs are still to be done. Phase 3 and later are not started.

Research behind this plan: `documentation/research/bf16-head-offset/` (the lineage survey, and `results/ejp0-681k/` for the model the Lichess bot plays). The inventory of every forward pass and logit consumer that this plan must touch is in the last section.

## The problem, briefly

- The two heads' outputs grow a large **shared offset**, a constant added to every logit of a position. Softmax ignores it, so nothing in the loss pulls it back.
  - Value logits on Ejp0 at 681k sit near +510.
  - Policy logits sink steadily with training: −42 at 681k, −138 at 1.397M, and down to −266 on v5.
- bf16 is coarse at those sizes (a step of 2–4 near 510, 0.25 near 42), so the real differences between logits are rounded away.
- **Measured on Ejp0 at 681k:**
  - Value: 55% of positions tie; value cross-entropy is +0.133 nats worse than fp64; the start position reads W/L 0.12/0.87 instead of 0.30/0.69.
  - Policy: top-2 moves tie in 12.7% of positions.
- **Cause:**
  - The cross-entropy targets are built in bf16 and don't sum to 1. This predicts 74% of the measured value-bias drift.
  - bf16 rounding in the backward pass.
  - No restoring force: softmax gives the offset zero gradient, and the head biases have no weight decay.
- The rest of the network measured fine: LayerNorm bounds the residual stream.

## Phases

Each phase is its own commit (build and commit per phase). The owner decides when to start.

### Phase 0: numerics audit (before any fix)

Extend the existing weight analyzer (`Network/NetworkWeightAnalyzer.swift`; Debug ▸ "Analyze Network Weights (Champion/Trainer)", and RunAllAnalyses) into a numerics audit. It measures today's problems and becomes the before/after gauge for every later phase.

**A. Static checks (weights only; fast; any checkpoint).** For every tensor, and for each of fp32, bf16 and fp16:
- **Overflow headroom:** max |w| against the format's maximum (fp16 overflows at 65504).
- **Underflow:** the fraction of values that become subnormal or flush to zero (fp16's smallest normal is 6.1e-5).
- **Rounding step against spread:** the format's step at the tensor's typical magnitude, divided by the tensor's standard deviation.

Known hot spots, checked by name:
- **Head shared-offset detectors:**
  - value `fc2`: the mean row's norm against the per-class residual norms, and the bias mean against its init ln6/3;
  - the policy final layer: the same statistics;
  - the policy bias mean against 0.
- **BatchNorm running statistics:**
  - variance near or below fp16's normal range;
  - |running mean| / √variance per channel. The known worst is 30.9 at `blocks.1.bn1`, which risks cancellation when the input is rounded before the mean is subtracted.
- **ReZero α:** whether tanh(α/C) rounds to exactly 1 in each format. Report it; don't flag it, since the cap is by design.
- **fp32 masters against the bf16 working copy:** when a trainer is analyzed, report the largest divergence per tensor.

**B. Dynamic checks (run the real MPSGraph network).**
- **Build:** the same weights are built as three networks, with the architecture's `computeDataType` set to fp32, bf16 and fp16 in turn.
- **Positions:** a fixed, reproducible set. The start position, plus a seeded sample from a corpus shard, plus the Lichess bot's recorded games when present. The set is configurable and its size is logged.
- **Head comparison against fp32:**
  - value: ties (two or more W/D/L logits equal), value CE change against game results where known, mean |Δv|, W/D/L argmax changes, and the start-position W/D/L;
  - policy: KL, top-1 changes, top-2 ties, and the legal-logit level (mean) and spread;
  - both heads: the per-position mean logit (the offset itself).
- **Per-layer checks** through optional analysis taps:
  - `ChessNetwork` gains an opt-in flag that exposes each block's output and each normalization's input as named targets. They are built only for audit networks, so production graphs are unchanged.
  - Per tap and per format: max |x|, the smallest nonzero |x|, per-channel |mean|/std, and headroom to each format's limits.

**Output**
- A fine / degraded / BAD verdict per check. The thresholds are named constants, starting from the survey's: value ΔCE ≥ 0.03 is BAD; ties ≥ 10% or ΔCE ≥ 0.005 is degraded.
- The existing JSON under `CheckpointPaths.analysesDir`, and a `[NUMERICS]` summary block in the session log.
- The existing alert.
- **CLI:** `--analyze-numerics <model file | folder>`. Given a folder, it covers every checkpoint, identified by `__metadata__` and not by filename, which replaces the Python survey.

**Also fix:** `ValueHeadAnalyzer` looks for `value_fc2_*`, but the variables are named `value_wdl_fc2_*` (ValueHeadAnalyzer.swift:64, :191-194; ChessNetwork.swift:3082), so its fc2 details are always nil.

**Validation**
- On Ejp0 at 681k, the audit reproduces the Python results within tolerance: value ties 55.3%, ΔCE +0.133, top-2 ties 12.7%, legal mean about −42.
- On an fp32 line (KbHZ or mUF5), everything is fine.
- A unit test pins each static check on synthetic tensors with known answers.

### Phase 1: inference fix — IMPLEMENTED (with Phase 2, one commit; validation pending)

**1a. fp32 head tails.**
- In `ChessNetwork.policyHead` and `valueHead`, cast the head's last hidden activation and its final weights and bias to fp32 before the final matmul or conv. Keep everything after that in fp32:
  - policy `policy_conv` (or `policy_fc`) + bias → `policyOutput` (fp32);
  - value `fc2` + bias → logits → `value_probs` softmax → scalar, including the scalar constant `[1, 0, −1]` (fp32).
- **For the policy, start the fp32 tail earlier: at the `policy_pre_bn` normalize** (`intermediate_conv` and `fc_bottleneck`; `simple_conv` has no pre-block, so its tail is the final conv alone). Measured on the lines with large policy offsets, h7vI and Xuub:
  - Starting at the final conv still leaves policy KL of 1.5–3.3e-3, with 3.5–5.3% of positions losing their fp64 top move.
  - The cause: the bf16-rounded policy features get multiplied by `policy.conv`'s large mean row, giving an error per square that softmax doesn't cancel.
  - Starting at the pre-BN normalize cuts this 7–14×, to KL 2.4–2.9e-4.
  - Source: `documentation/research/fp16-feasibility/`.
- The weights stay stored in their current dtype. `Models/` checkpoints are bf16-exact, so casting them up loses nothing.
- **Config D** (`bf16CastInForward`: fp32-stored variables cast to bf16 in the forward pass): the tail skips that cast, so it reads the fp32 variables directly.
- One builder serves every path: inference, batched self-play, arena, UCI, the bot, probes and the trainer. So this covers them all. See the inventory for the consumers to update.

**1b. Recenter the value head when a checkpoint is decoded** (one exact calculation, no gradual decay, no migration of saved files).
- **Where:** the checkpoint decoders, `SafetensorsModelIO.decode` (model and trainer files, including sessions) and the legacy `ModelCheckpointFile.decode`. Every file load passes through one of these.
  - Not `ChessNetwork.loadWeights`: recentering tensors that are already centered still flips low bits, which would break the bit-exact save/load round trips and the forward verification on save.
- **When:** only if the file's `__metadata__` lacks the marker `value_head_centered`. Every save writes the marker from now on, so each file is recentered at most once in its life, and new files round-trip bit-exactly.
- **What** (only for `.wdlSoftmax`): in `value_wdl_fc2` weights, native shape `[hidden, 3]`, subtract each hidden row's mean over the 3 classes. Subtract the mean of the 3 bias values. Softmax output is unchanged in exact math.
- **Trainer files:** apply the same row-mean removal to the `fc2` velocity tensors (`opt.<name>.velocity`). Otherwise the momentum puts the offset back. The decoded values are what seed the fp32 masters, so the masters come out centered too.
- **Reporting:** the decoded file reports what was removed (mean-row norm, bias mean). Loaders log it once under `[NUMERICS]`.
- **Lichess bot:** its generation info records that the head was recentered on load, since the weights played no longer match the file's SHA.
- **Policy:** it has no exact one-shot equivalent, because its offset varies by square. 1a makes it harmless, and Phase 2 stops it growing.
- **In-memory copies need nothing:** trainer → candidate, promotion, and champion copies all start from already-decoded (centered) weights. A freshly initialized network's small random offset is harmless, and Phase 2 keeps it from growing.

**1c. Readback and consumers.**
- Value and probability readback switch to fp32 (ChessNetwork.swift:1209, :1222, :1315, :1492-1493), along with the result buffers of the compiled batched executable. Leaving them as they are would silently misread fp32 bytes as bf16.
- Every in-graph op that mixes an fp32 head tensor with a compute-dtype constant, placeholder or mask gets an explicit cast (the list is in the inventory).

**Validation**
- The Phase 0 audit on Ejp0 at 681k: value ties ≤ 0.1%, value ΔCE ≤ 0.001, policy top-2 ties ≈ 0, policy KL ≤ 1e-4.
- Start-position W/D/L within 0.01 of fp64.
- Existing tests pass. Tests whose exact-value expectations change are listed below, and any change to them needs the owner's approval.
- Throughput: plies per hour in self-play and steps per second in training, before and after, with a regression under 2% expected.

### Phase 2: training fix (stop the offset growing) — IMPLEMENTED (with Phase 1, one commit; validation pending)

1. **fp32 targets that sum to exactly 1.** Build the policy and value targets in fp32 and renormalize them after smoothing (`y /= Σy`, in fp32).
   - Today's errors: bf16 `1/3` makes the value target sum to 1 + ε·0.00195, and bf16 `1/|legal|` is rounded.
   - Feed both ε values as fp32. They are bf16-narrowed on the host today (ChessTrainer.swift:5960-5961).
   - **Complement target:** when |legal| = 1 its main part is empty and the target sums to ε. Give that position zero complement weight instead of a malformed target.
2. **The fp32 tail in training.** This comes from Phase 1's builder change. The policy CE, complement CE and value CE run on fp32 logits against fp32 targets.
3. **Mean-center in fp32 before every softmax and CE on the loss path.**
   - Mean, not max: `reductionMaximum` has no gradient rule in this graph (ChessTrainer.swift:2691-2697).
   - Policy: center `policyOutput` over all 4864 logits **once, before masking**.
     - Policy CE and illegal mass use all 4864 logits.
     - Masking afterwards keeps the −1e9 cells out of the mean.
     - The legal-only softmaxes (entropy, played-move probability, the KL probe) are unaffected.
   - Value: center the 3 logits. Only for `.wdlSoftmax`: centering the single-logit `scalarTanh` head would force v = 0.
   - Effect: the gradient on the shared offset becomes exactly zero (1ᵀ(I − 11ᵀ/N)g = 0), whatever the targets. So the offset can no longer drift. Item 1 is still done for correctness of the loss values and the non-centered paths.
4. **Recenter on load:** already in Phase 1b, at checkpoint decode, for weights and velocity (the masters are seeded from the decoded values). It removes the offset existing lines carry. Nothing gradual is needed.
5. **Monitoring:** add the per-head mean logit to `[STATS]`, computed **before** centering: the policy mean over 4864 and the value mean over 3. It should stay near 0.
   - The touchpoints are listed in the inventory (trap 11).
   - New fields get default values, so the test helpers that build these structs still compile.
6. **Keep raw-magnitude diagnostics meaningful:** compute `pLogitAbsMax` and the probe's `policy_logit_abs_max` on the pre-centered tensor.
7. **Widen the bf16-only diagnostic reductions to fp32:** pLossWin/Loss, vMean/vAbs, pW/pD/pL, the non-negative counts and played-move probability. Update the trainer's scalar readbacks (ChessTrainer.swift:6707-6802) to fp32 to match.
8. **Value baseline:** keep the value-baseline forward fp32 through to the advantage. It is re-narrowed to bf16 at ChessTrainer.swift:2587 today.

**Validation**
- A short corpus-replay run on a fresh bf16 model: the head mean logits stay within ±1 after 20k steps (today the value bias mean alone climbs ~+2e-5 per step, with the shared value logit reaching ~510 by 100k).
- Resuming Ejp0: the value offset is gone at load (logged), and no drift over 20k steps.
- The Phase 0 audit on the resulting checkpoints is fine.
- The full test suite passes, per CLAUDE.md: `ChessTrainer` and `ChessNetwork` change.

### As implemented (Phases 1 and 2): deviations from the text above

- **The whole loss path is fp32, not only the cross-entropies.** Once the head outputs are fp32, `buildTrainingOps` builds everything downstream of them in fp32: the z / vBaseline / legalMask feeds (no in-graph narrowing cast), the advantage, every per-position loss, every reported scalar, and every loss-side scalar feed (loss weights, entropy coefficient, both ε, complement enable), not only the two ε. The fp16-only −3e4 mask magnitude is gone: the mask is added in fp32.
- **Targets and centering live in `Training/HeadLossGraph.swift`** (`policyTargets`, `valueTarget`, `renormalizedTarget`, `centerLogits`), so their invariants are tested on a standalone graph (`HeadLossGraphTests`).
- **Complement target at |legal| = 1:** zero weight via a per-position `complementValid` mask; the renormalization's denominator is floored so the zero-weighted row is an all-zero target rather than NaN.
- **The legacy `.dcmmodel` writer also writes the marker** (in its metadata JSON), so every writer marks its files; production only writes safetensors. Legacy decode infers a trainer file from its tensor count and throws on a count that is neither.
- **Loaders log `[NUMERICS]`** from `CheckpointManager.loadModelFile` and `loadSession` (champion and trainer), not from `decodeAnyModelFile`, which the post-save verification also runs.
- **The launch marker** is a `[NUMERICS]` line after the `[APP] launched` banner on every launch of a build with the fix; the first log carrying it is the annotation point.
- **Lichess bot:** `LichessBotGenerationInfo.valueHeadRecenteredOnLoad` is `Bool?` — nil for in-memory sources and for records written before the field existed.
- **Also changed:** `ChessNetwork.readFloatsFP32(from:into:count:)` now traps on a non-fp32 or wrongly sized tensor instead of copying raw bytes blindly; `computeBatchStats` reads each BN's batch stats by the tensor's own dtype (the policy pre-BN's are fp32 now); `policyOutputReadback` and `valueOutputFP32` are gone (the outputs themselves are fp32).
- **Monitoring:** `pLogitMean` / `vLogitMean` on `[STATS]`, `[STATS] arena-start`, `[REPLAY]` and `[VS-UCI]`, and `policy_logit_mean` / `value_logit_mean` in `results.json`. On the scalar-tanh head `vLogitMean` is the raw pre-tanh logit's mean (not centered).

### Phase 3 (separate ROADMAP item): full-precision checkpoints

- Save the trainer's fp32 masters, optionally with the velocity, instead of the bf16 working copy. See ROADMAP "Full-precision weights in every model checkpoint".
- Per-variable fp32 storage for the head tails (inventory D and trap 5) belongs with that item, not here. With the offset removed and held at zero, bf16-stored head weights are adequate.

### Not pursued: 2-logit value head (decided 2026-09-28)

The owner decided against this. Recentering at load plus centering in the loss handle the offset, so the value head stays 3 logits. The idea as proposed:

- Give the value head 2 logits with draw fixed at 0. The offset then can't exist by construction.
- Old checkpoints convert exactly: subtract the draw row and bias from the other two, then drop them.

## Expected behavior changes (not bugs)

These change with identical weights once Phase 1 lands:
- **Top-move tie-breaking,** which today randomizes among bf16-tied moves:
  - tactical/Lichess probe bestRank → **pElo** (SessionController+TacticalProbe.swift:187);
  - `top1Legal`;
  - argmax play at τ = 0.01;
  - the Lichess bot's top-move order.
- **Threshold decisions on W/D/L:** the self-play draw-watch (0.95, BatchedSelfPlayDriver.swift:443) and the bot's resign and draw rules (LichessBotPlayPolicy.swift:34/47/54).
- Probe trends will show a one-time step at the switch. Log a `[NUMERICS]` marker at first launch with the fix so charts can be annotated.

## Tests needing attention

Changing any existing test needs the owner's approval.
- **Compile:**
  - `TrainingLiveStatsGatingTests.makeTiming` (:19-54) and `CliTrainingRecorderTests.makeStats` (:197-290) build `TrainStepTiming` and `StatsLine` memberwise.
  - Give new fields defaults, or approve updating these helpers.
- **Exact or tolerance-based comparisons that may move:**
  - `ChessNetworkValueDistributionTests` :62 (exact policy equality at :69; bf16 tolerance at :77, :82)
  - `PolicyHeadCorrectnessTests`: batched vs single (:324), uniform-at-zero-input (:1049), `testEveryTrainableVariableReceivesGradient` (:544)
  - `FP16ComputePathTests` (:81, :202)
  - `DropoutGraphWiringTests.testValueBaselineIsDropoutFree` (:182)
  - `RepPlaneProbeTests` (:119, raw `policy.max()`)
- **Optimizer:** `MomentumOptimizerTests` (:154, :222, :251, :429, :516, :602).
- **Checkpoint bit-exact:** `CheckpointManagerSafetensorsTests` (all 7), `BlockGroupArchitectureTests` (:197-292).
- **Forensic:** `MacOS27NaNIsolationTests`. Its Σpolicy checksum (:336-387) loses its signal under centering.
- **New tests:**
  - recenter at decode: softmax unchanged, the marker written and honored, velocity centered, and new files still round-trip bit-exactly;
  - targets sum to 1 in fp32; the |legal| = 1 complement case;
  - centering gives a zero shared-direction gradient;
  - the scalarTanh head is not centered;
  - Phase 0 static checks on synthetic tensors.
- **Note:** `DrewsChessMachine.xctestplan:14-15` sets `DCM_RUN_SLOW_TESTS=1`, so the "gated" slow suites run under the scheme. CLAUDE.md says the opposite; reconcile them.

## Inventory: every site this plan touches

From a read-only survey of the code (2026-09-28). Paths are relative to `DrewsChessMachine/DrewsChessMachine/`.

**One head builder:** `ChessNetwork.init` (Network/ChessNetwork.swift:595). `policyHead` is defined at :2893 and called at :929; `valueHead` is defined at :3009 and called at :948. The trainer, probes and CLIs reuse `network.graph` or the inference APIs; only test mirrors duplicate head logic.

**Paths that execute the heads, all through that builder**

| path | entry |
|---|---|
| single-position `evaluate` / `evaluateWithValueDistribution` | ChessNetwork.swift:1144/1240 → :1168 |
| `evaluateBatched` (self-play, arena, train-vs-UCI, probes) | :1366/1402 → :1419; compiled :1556 |
| value-baseline forward (the 2nd forward of every training step) | :1620/1711; called from ChessTrainer.swift:4713 |
| training forward + backward | ChessTrainer.swift:2433; compiled :6362; run :6443 |
| KL probe (post-update forward) | ChessTrainer.swift:6256; run :6914 |
| legal-mass snapshot | ChessTrainer.swift:5058 → :5122 / :5129 |
| random-data sweep | :7093; SweepCLI.swift:41; ArchSweepCLI.swift:82 |
| self-play / arena / UCI / Lichess bot | BatchedSelfPlayDriver.swift:413; TickTournamentDriver.swift:525/549; UCIEngine.swift:339; LichessBotMoveChooser.swift:91 |
| probes and diagnostics | SessionController+TacticalProbe.swift:269/291/345; +CandidateProbe.swift:199/206; +Diagnostics.swift:147/230/240; ReplayBufferAnalyzer.swift:781 |
| checkpoint verification | CheckpointManager.swift:1190/1202 |
| train-vs-UCI / corpus replay | TrainVsUciDriver.swift:299; TrainVsUciRunner.swift:173; CorpusReplayRunner.swift:441/764 |

**In-graph logit consumers** (ChessTrainer.swift `buildTrainingOps`): masked logits :2606-2611; policy CE :2816; complement CE :2828; per-position CE × advantage :2839-3057; pLossWin/Loss :3072-3100; value CE :3200; scalarTanh MSE :3223; vMean/vAbs :3245-3255; pW/pD/pL :3266-3271; legal softmax :3330; entropy :3351-3388 (feeds the loss); illegal mass :3395-3420 (feeds the loss); non-negative counts :3429-3492; pLogitAbsMax :3503-3516; played-move probability :3544-3631; policy weight norm :3933-3960; KL :6271; baseline feed :2582-2587.

**CPU consumers**
- **Sampler and move choice.** They do their own max-subtract, so they are shift-invariant: `MoveSampler` (MoveSampler.swift:123-190), the Lichess bot top moves (LichessBotMoveChooser.swift:149-159).
- **Value thresholds:** BatchedSelfPlayDriver.swift:443; LichessBotPlayPolicy.swift:34/47/54; the UCI centipawn score (UCIEngine.swift:369-374).
- **Raw-logit readers** (not shift-invariant):
  - TacticalProbe `logitAbsMaxPerPos` :381-391 → ProbeModelCLI :180-196/248-257;
  - the Diagnostics conditioning check :230-338;
  - HoverPolicyOverlay.swift:83/92-117/177;
  - ChessRunner.swift:49-100.

**Targets** (ChessTrainer.swift): policy ε :2704; one-hot :2710; uniform over legal moves :2728-2743; smoothed :2745-2764; complement :2779-2810; value one-hot :3164-3183; value smoothed :3186-3196; value ε :3140.

**Dtype plumbing**
- `mpsDataType(for:)` ChessNetwork.swift:2265; `weightStorageDataType` :575; `castWeightForForward` :582.
- Readback casts: :941-943 (policy fp32), :960-962 (value scalar fp32, baseline only).
- fp32 masters, update and sync: ChessTrainer.swift:4034, :4099-4135, :4166-4213, :2412-2431.
- Master I/O :5611-5727; trainer export and load :5393-5437.
