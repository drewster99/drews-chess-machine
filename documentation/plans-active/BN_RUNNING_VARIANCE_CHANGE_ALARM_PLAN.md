# BN running-variance change alarm plan: raise when a batch-norm channel's running variance jumps, not only when it is already huge

Status (2026-10-07): **P1–P4 implemented; P5 (validation runs V-1 … V-7) not run.** The owner approved the direction ("it needs a plan then an implementation"); the decisions below (BVJ-1 … BVJ-18) were implemented as written, and the open questions (Q-1 … Q-9) were answered with this plan's own recommendations for the implementation, pending the owner's review with the validation numbers: Q-1 critical at 100× (accepted; owner 2026-10-07: firing on clip1 / clip2 / clip5 / B-leaky / B-leakyall is acceptable and expected, no extra tuning); Q-2 warnings on healthy runs accepted at the 10× level; Q-3 the count arm stays a warning; Q-4 event-like clear (about one lookback after the last jump); Q-5 default action `log` (owner 2026-10-07: never stop a run by default); Q-6 **owner 2026-10-07: reads before the process (launch, resume, segment, GUI start) has trained 100 trainer steps only build the baseline** (`batchNormRunningVarianceJumpSettleSteps`, from the observation stamp's steps trained by this process; no extra read); after a rewind the first read is a baseline only; Q-7 older builds refuse a lineage summary naming the rule, as for earlier rule additions; Q-8 the count is of channels at ≥ 10× (3× was descriptive); Q-9 lookback 1,000 trainer steps (owner 2026-10-07: as planned). Implementation deviations: a profile stores `Double` ratios (not `Float`), so its outlier count can never differ from the summary's at the 10× boundary; `runningVarianceRatios` returns `nil` ratios for a site whose median is not positive. The X3 expectations were checked against the fixture before the tests were written (same steps as Part X3).
- Every `file:line` was checked against `main` at `6df0125f`.
- Paths are relative to `DrewsChessMachine/DrewsChessMachine/` unless they start with `DrewsChessMachine/` (project folder), `DrewsChessMachineTests/` (= `DrewsChessMachine/DrewsChessMachineTests/`), `documentation/`, `experiments/` or `scripts/`.
- Session logs are under `~/Library/Logs/DrewsChessMachine/`, checkpoints under `~/Library/Application Support/DrewsChessMachine/{Models,Sessions}/`. Every number in Part E was re-measured for this plan from those files (method: E5).
- "BN" is batch normalization. A channel's **ratio** is its running variance divided by the median running variance of its own BN site: the quantity `[LAYER-HEALTH]` already reports as `rvMaxOverMedian` for the largest channel.

**The request (owner, 2026-10-07, verbatim).**

> The existing bn_running_variance_runaway rule raises only when a channel's running variance reaches 1000× its site's median. B-silu shows why that is too late. Its worst channel sat at 50–62× from step 12k to 19k, apparently harmless. Then at step 20k a different channel, 76, appeared at 980×, just under the threshold, and by step 21k it was at 45,186×. At that point outliers had jumped from 62 to 207 channels and 104 channels were pushed to β/|γ| < −2; the run never recovered. A rule based on change rather than level would have fired at step 20k, on either a new channel rising more than 10× between live reads or the count above 10× doubling (5 → 11). The clip arms show that slow growth of a single channel, as in AgG3's channel 88 or clip1's 31 reaching 154×, is not the dangerous pattern, so the level threshold can stay high without missing it. The gradient-spike rule sees the 11-step burst itself but is transient by design. The running-variance jump is the signature that persists in the weights and in every later checkpoint, which makes it the right signal for stopping or suspending a run before compute is wasted on a damaged network.

**What this plan does.**
- Adds rule 14, `bn_running_variance_jump`, to the one training-health evaluator (`Training/TrainingHealth.swift`). Rule 8 (`bn_running_variance_runaway`, level ≥ 1,000×) is unchanged.
- Two arms, both judged on every live evaluation (every 50 trainer steps, all paths):
  - **Jump:** a channel whose ratio is now ≥ 10× **and** rose ≥ 10× above its own lowest ratio in the previous 1,000 trainer steps. Warning; **critical when the jumped channel is ≥ 100×**.
  - **Outlier count:** the number of channels at ≥ 10× rose to at least `max(2 × baseline, baseline + 5)`, the baseline being the lowest count in the previous 1,000 trainer steps. Warning.
- Carries the per-channel ratios the rule needs from the live read the paths already make (no new GPU read).
- One new parameter, `training_health_action_bn_running_variance_jump`, through the full CLAUDE.md checklist.
- Logs: a `rvOver10xMedian=` field on every `[LAYER-HEALTH]` line, a per-site column in the checkpoint table, `rvRiseMax=` / `rvOutliers=` on `[HEALTH] check`, and the usual `[ALARM] health …` events.
- `--replay-health-log` judges the jump arm as a lower bound from the logged largest channel, and the count arm from the new field.

**Three findings that change the owner's framing (details in Part E).**
- At the live cadence the jump is visible at **trainer step 19,800**, not 20,000: B-silu's channel 76 at `blocks.2.bn1` was 0.02× at the 19,000 read and 154.0× at the 19,800 read (980.1× at 20,000 is the checkpoint value). 19,800 follows the precursor gradient steps 19,785 / 19,795 (E-0024) and is 800 steps before the 20,599–20,609 burst.
- The literal rule "a new channel rising more than 10× between live reads" also fires on runs that stayed healthy: clip1 (channel 76 0.02× → 38.9× at 20,000, the clipped precursor's footprint), B-leaky (0.11× → 35.0× by 21,000) and B-leakyall (0.24× → 39.9× by 13,000; 0.30× → 46.6× and 0.21× → 40.5× by 30,000). Every healthy-run jump, and B-silu's two largest, start from a channel whose variance had decayed to ≤ 0.3× of its site's median. What separates B-silu is the level the jumped channel reaches (154× at its first read, 980× at 20,000) — hence the 100× critical level (BVJ-5). clip2 and clip5 (which survived) also reach it.
- In the app, rule 9 (`gradient_spike`) already raises at **19,800** on the precursor and clears at 19,850: relcap V-1 (`dcm_log_20261007-060735.txt`, build 2390, bit-identical to B-silu) logs `raise rule=gradient_spike … trainerStep=19800 value=max/ref=28.26 max=8.4497`. (The offline replay of the older B-silu log sees only the 50-step rows, so its first rule-9 raise is 20,600.) The new rule fires at the same evaluation. What it adds is not an earlier step but a signal that the spike changed the weights, with a critical level and an alarm that stays active for one lookback: rule 9 reads pre-clip norms, so it would raise at 19,800 on clip1, clip2 and clip5 too (identical weights up to 19,785), while their BN footprints range from 38.9× to 513×.

Rules this plan follows (CLAUDE.md files and the owner's standing rules):
- one evaluator, one rule definition, one code path for GUI, corpus replay, train-vs-UCI and offline replay;
- thresholds are declared constants measured against the incident runs (alarm plan OD-14), never parameters;
- the monitor stays an observer: no random draws, no trainer, optimizer or replay-buffer state, no new GPU read;
- the full parameter checklist; `absentValue` declared;
- no silent defaults: missing data is "no data" or "hold", never "healthy";
- no `try?`, no force unwraps; one SwiftUI `View` per file;
- tests are never modified or deleted beyond the edits the new rule forces, which are listed (X6) and reported.

---

## Summary

| # | Item | Phase |
|---|---|---|
| 1 | `LayerHealth.runningVarianceRatios(_:)`: the one definition of a channel's ratio to its site median, used by the existing max/median, the new per-site outlier count and the new per-channel profile | P1 |
| 2 | `BatchNormRunningVarianceProfile` (pure, not `Codable`): every BN site's per-channel ratios from one read | P1 |
| 3 | `LayerHealthSummary.BatchNormSiteHealth.runningVarianceOutlierCount` (encoded `running_variance_outlier_count`) and the summary's total | P1 |
| 4 | `LayerHealthDigest.runningVarianceChannels` (`everyChannel` in-app, `largestOnly` offline) with the outlier count | P1 |
| 5 | `TrainingHealthRule.batchNormRunningVarianceJump` (rule 14), its constants, its history in the evaluator `State`, the pure `BatchNormRunningVarianceJump.read(...)`, rewind reset | P1 |
| 6 | New fixture `TrainingHealthRunningVarianceCheckpoints.json` (running variances read from the evidence checkpoints) and the tests of Part X | P1 |
| 7 | Parameter `training_health_action_bn_running_variance_jump` (Part K); session field and restore | P2 |
| 8 | Live wiring: `LayerHealthLog.LiveOutcome` carries the profile; `TrainingHealthReads.liveLayerHealth(from:)` puts it in the digest | P2 |
| 9 | Log fields (`rvOver10xMedian=`, table column, check-line `rvRiseMax=` / `rvOutliers=`), `results.json` | P2 |
| 10 | Offline replay: parse the channel of `rvMaxOverMedian=` and `rvOver10xMedian=`; lower-bound jump; header lines | P2 |
| 11 | GUI: display name and description (`TrainingHealthRule+Display.swift`); the Health tab row and the alarm-list row come from `allCases` | P3 |
| 12 | Docs: `documentation/training-health-alarms.md`, `CLAUDE.md`, `TRAINING_HEALTH_ALARMS_PLAN.md` pointer, `CHANGELOG.md`, `ROADMAP.md` line | P4 |
| 13 | Validation runs V-1 … V-7 | P5 |

---

# Part E — Evidence

## E1. The owner's numbers, checked

| Claim | Measured | Source | Verdict |
|---|---|---|---|
| worst channel 50–62× from 12k to 19k | 50.1× at 12,000 → 61.9× at 19,000 (`blocks.1.bn1[31]`, then `blocks.2.bn1[31]`); live-line maximum in 12,000–19,750 is 63.4× (19,600) | live lines of `dcm_log_20261005-234437.txt`; checkpoints `20261005-lrBsilu-cyc1-replay-step{12000…19000}` | verified |
| channel 76 at 980× at 20k | 980.1× at `blocks.2.bn1[76]` (checkpoint and live line at 20,000); it was 0.02× at 19,000 | checkpoint `…-step20000`; fixture log | verified; **first visible at 19,800** (154.0×) on the 50-step live lines |
| 45,186× at 21k | 45,186.0× at `blocks.1.bn1[76]` (1.64× at 20,000). `blocks.2.bn1[76]` was 42,205.4× at 21,000 | checkpoint `…-step21000` | verified; at 21k the largest is the same channel index at a different site (pre-activation tower: every `bn1` and `tower_final_bn` normalize the residual stream, so channel 76 is one residual-stream channel seen at several sites) |
| outliers 62 → 207 | channels at ≥ 3× their site median: 62 (19,000), 68 (20,000), 207 (21,000) | checkpoints | verified **if "outlier" means ≥ 3×**; the request does not define it |
| 104 channels pushed to β/\|γ\| < −2 | 0 at 19,000 and 20,000; 104 at 21,000 (11 `blocks.0.bn1`, 19 `blocks.0.bn2`, 17 `blocks.1.bn1`, 1 `blocks.1.bn2`, 21 `blocks.2.bn1`, 3 `blocks.2.bn2`, 27 `policy.pre_bn`, 5 `value.bn`) | checkpoints (γ, β) | verified |
| count above 10× doubling 5 → 11 | 5 at 19,000, 11 at 20,000 (all BN sites together); 53 at 21,000 | checkpoints | verified (global count) |
| AgG3 channel 88 slow growth | 3.1× (step 351) → 13.6× (1,001) → 33.9× (2,002) → 48.1× (4,124) → 47.0× (5,261); at most 5.5× within 1,000 steps (3.1× at 351 → 17.0× at 1,252, before the gate), at most 2.2× from 1,607 on | `dcm_log_20261007-084445.txt` live lines; trainer saves in `Sessions/20261007-1*-49tn-promote.dcmsession` | verified |
| clip1 channel 31 reaching 154× | 61.9× (19,000) → 153.6× (40,000), never more than 1.2× per 1,000 steps | checkpoints `20261006-lrBsilu-clip1-replay-seg1-step*` | verified |

Not verifiable from existing data:
- The per-channel ratios between checkpoints. A live line names only its largest channel; enumerated checkpoints are 1,000 steps apart. The **critical** raise at 19,800 is exact: no channel exceeded 63.4× on any live line from 19,000 to 19,750, so none was at 100× before 19,800, and channel 76 was 0.02× at the 19,000 read. Whether a **warning** came earlier (a channel crossing 10× after a 10× rise between 19,000 and 19,750) and the count arm's first step inside 19,800–20,000 are unknown. V-1 measures both.
- Whether the jumped-channel level predicts the burst. clip2 and clip5 left B-silu's trajectory at 19,800, so their weights at 20,600 differ; their survival does not show that 129× or 513× is safe had the precursor gone through unclipped (the README's own caveat, E-0023).

## E2. The checked fixture

`DrewsChessMachineTests/Resources/TrainingHealthIncidents/TrainingHealthBSiluBatchNorm.json` holds γ and β only, at trainer steps 20,000, 21,000 and 22,000 — **no running variance and no pre-jump checkpoint**. It cannot carry this rule's regression test; a new fixture is needed (BVJ-17, X3). The B-silu log excerpt (`TrainingHealthIncident-Bsilu.log`) has a live line every 50 steps with the largest channel, which is enough for the offline lower bound (X5).

## E3. Every evidence run under the rule

Ratios from the enumerated checkpoints (1,000-step spacing, so consecutive checkpoints are exactly one lookback apart) plus the live lines' largest channel (50-step spacing in these builds; about every 120 steps in build 2390's time-cadence logs). "In-app" is what the app would raise with every channel read every 50 steps; "offline" is `--replay-health-log` on the existing logs (BVJ-12).

| Run | What happened | Jump arm, in-app | Count arm, in-app | Offline replay of its log |
|---|---|---|---|---|
| **B-silu** (cap 15) `dcm_log_20261005-234437.txt` | blowup at 20,599–20,609 (E-0021) | **critical at 19,800** (exact, E1): `blocks.2.bn1[76]` 0.02× at the 19,000 read → 154.0× at 19,800 (×7,700); at 20,000 also `[34]` 0.01× → 418.7×, `tower_final_bn[34]` 0.69× → 27.2×, `[76]` 0.64× → 52.5× | warning by 20,000: 5 → 11 (needs 10) | critical at **19,900** (851.4× against 61.3×, the smallest of the lines' largest ratios in 18,900–19,850) |
| ctl15 (cap 15, resume 18k) `dcm_log_20261006-181510.txt` | bit-identical to B-silu | as B-silu | as B-silu | critical at 19,900 |
| relcap V-1 (log only, resume 18k) `dcm_log_20261007-060735.txt` | bit-identical to B-silu | as B-silu | as B-silu | critical at **19,960** (time-cadence lines: 19,720, 19,840, 19,960) |
| clip1 (cap 1.0) `dcm_log_20261006-170000.txt` | survived, level with B at 40k | **warning by 20,000**: `blocks.2.bn1[76]` 0.02× → 38.9×; not critical through 22,000 (every logged largest ≤ 96.9×, channel 31's slow growth), and that channel's checkpoint values are 18.3–30.1× from 21,000 to 40,000. Channel 31's slow growth never fires | none: largest 1,000-step rise +5 (14 → 19 at 31k → 32k, needs 28) | none |
| clip2 (cap 2.0) `dcm_log_20261007-035017-2.txt` | survived to 23k | warning by 19,850 (71.0×), **critical at 19,900** (104.9×) | none at 1k spacing (5 → 8 → 9 → 11 → 13) | none (129.5× / 61.7× < 10) |
| clip5 (cap 5.0) `dcm_log_20261007-035017.txt` | survived to 23k | warning by 19,800 (70.9×), **critical at 19,850** (310.0×) | none at 1k spacing (5 → 8 → 10 → 10 → 11); **unknown between 20,000 and 20,750**, where a read at 10 with 19,750's 5 still in the lookback would warn | none |
| B-leaky `dcm_log_20261005-204108.txt` | healthy | **warning by 21,000**: `blocks.2.bn1[76]` 0.11× → 35.0×; never critical (logged largest ≤ 62.7× in 20,000–22,000) | none (largest rise +4) | none |
| B-leakyall `dcm_log_20261005-234434.txt` | healthy | **warning by 13,000**: `blocks.2.bn1[76]` 0.24× → 39.9×; **warning by 30,000**: `blocks.1.bn1[43]` 0.30× → 46.6×, `blocks.2.bn1[22]` 0.21× → 40.5×; never critical (≤ 45.9×, ≤ 80.4×) | none (largest rise +3) | none |
| B (ReLU) `dcm_log_20261005-013235.txt` + `-171541.txt` | healthy (value-BN losses early) | none after the gate. `stem.bn[102]` 1.70× → 24.5× between 1,000 and 2,000 is before the gate (2,000 for these runs) | none (largest rise +4, 7 → 11) | none |
| A (constant LR) `dcm_log_20261005-013220-2.txt` + `-171451.txt` | healthy | none: no channel ever reaches 10× (max 7.2×) | none | none |
| C `dcm_log_20261005-090417.txt` + `-121841.txt` | diverged at 300 | critical at the first judged read past the gate (2,513 checkpoint: `blocks.2.bn1[65]` 26.35× → 350.8×); C was already broken | none (counts 158–163) | none (the logged largest is `value.bn[12]` ≈ 5×10⁵ throughout, which hides every other channel) |
| C-leaky `dcm_log_20261005-231335.txt` | diverged | none: ended at 1,175, before the gate | none | none |
| V-3 (fresh B, relcap clip) `dcm_log_20261007-060832.txt` | bit-identical to B through 3,000 | none (0 channels ≥ 10× at 3,000) | none | none |
| **AgG3** (GUI, SiLU 3×11×11) `dcm_log_20261007-084445.txt` | healthy | none: trainer saves at 1,607 / 2,831 / 4,655 / 5,261 show no ≥ 10× rise (largest 2.16×, `blocks.2.bn1[88]` 20.3× → 43.8×) | none: 2 → 4 → 5 → 5 (2 → 4 is the "1 → 2"-style doubling the +5 minimum excludes) | none |
| R7, R8 (`dcm_log_20261002-011124.txt`, `-035513.txt`) | healthy | not measured: logs predate `[LAYER-HEALTH]`; checkpoints not analysed | — | no data |

Reading:
- The rule fires on B-silu 800 steps before the blowup. Existing rules in the app (relcap V-1 log, bit-identical to B-silu): rule 9 (`gradient_spike`, warning) at 19,800, cleared at 19,850, raised again at 20,600; rule 8 (`bn_running_variance_runaway`) at 20,600 (1,031.9×); `dead_channels` and `loss_spike` at 20,650; rule 4 (`illegal_mass`, critical) at 20,700. (`policy_offset_drift` was active from 18,100, unrelated.) Offline on the old log, rule 9 first raises at 20,600.
- At warning severity it also fires on clip1, B-leaky and B-leakyall: each jump is a real, persistent change (the channels stay 12–40× for the rest of the run), not noise, but those runs stayed healthy.
- At critical severity it fires on B-silu (and its replicas) and on clip2 / clip5, never on a run that stayed healthy at cap 15, and never on clip1. With action `stop_on_critical` B-silu stops at 19,800; so would clip2 and clip5 (Q-1).
- Neither slow-growth case fires: AgG3 channel 88 and clip1 channel 31 never rise 10× within 1,000 steps.

## E4. What the data says about the trigger

- Every healthy-run jump and B-silu's two largest started from a decayed channel: B-silu `blocks.2.bn1[76]` 2.6× (1k) → 0.0× (14k–19k), `[34]` 1.6× → 0.0× (its smaller 20,000 jumps, `tower_final_bn[34]` / `[76]`, start from 0.69× / 0.64×); B-leaky `[76]` 3.2× (2k) → 0.1× (20k); B-leakyall `[76]` 1.6× → 0.2× (12k), `[43]` 2.0× → 0.3× (29k), `[22]` 3.9× → 0.2× (29k). A channel whose BN input variance is near zero gets normalized by a tiny σ; a large update to the conv row feeding it reactivates it with a large variance. That is why the rise factor is huge (155× to about 64,000×) in all of them and does not separate healthy from damaged; the level reached does. (C, already broken, also has a 13× rise of a channel that was at 26×.)
- The jumps sit at LR peaks or the climb to one (B cycle: peaks near 1k, 11k, 21k, 31k): B-silu 19,800, B-leaky ≤ 21,000, B-leakyall ≤ 13,000 and ≤ 30,000.

## E5. Reproduce

- Ratios: for every tensor `<site>.running_var` of a checkpoint (float32 little-endian, safetensors header offsets), divide each channel by the site's median (`numpy.median`); "≥ 10×" counts channels with ratio ≥ 10; β/\|γ\| from `<site>.bias` / `abs(<site>.weight)`. Trainer step = the header's `cum_trainer_step` (`__metadata__`).
- Jump at checkpoint spacing: channels with ratio ≥ 10 at step s and ≥ 10 × their ratio at s − 1,000.
- Offline lower bound: per live line, the largest channel's ratio against the minimum, over the lines in the previous 1,000 trainer steps, of each line's largest ratio (an upper bound on every channel there).
- The scratch scripts used for this plan are not committed (no code changes in this step); P1 adds the reader to `experiments/20261005-lr-schedule-ab/bn_liveness.py` (BVJ-17), which also writes the fixture.

---

# Part R — The rule

## R1. Ratio (BVJ-2)

- `ratio(site, c) = running_var[site][c] / median(running_var[site])`, the median over the site's finite values; no ratio for a non-finite variance (rule 1 reports it) or a site whose median is not positive.
- Every BN site (`stem.bn`, every block's `bn1` / `bn2`, `tower_final_bn`, `policy.pre_bn`, `value.bn`), whatever activation follows: rule 8 already covers every site.
- One function, `LayerHealth.runningVarianceRatios(_:)`, used by `batchNormSiteHealth` (today's max/median, `Training/LayerHealth.swift:541`), the new outlier count and the new profile — the one source of truth for "ratio".
- A channel is identified by (site, channel index). One residual-stream channel seen at several sites counts once per site.

## R2. Lookback and baseline (BVJ-3)

- Lookback: the live reads at trainer steps `[s − 1000, s)` of the current history (inclusive lower bound, as `TrainingHealthReference.make`, `Training/TrainingHealth.swift:843-878`). At the 50-step cadence that is up to 20 reads. A read exactly 1,000 steps back counts, so consecutive 1,000-step checkpoints are comparable (X3).
- A channel's baseline is its **lowest** ratio over those reads. Not only the immediately previous read: a rise spread over several reads (clip2: 71.0× → 104.9× → 122.5× after the first read) and sparse or irregular cadences (GUI pauses, offline time-cadence logs) must give the same answer.
- The count arm's baseline is the lowest outlier count over the same reads.
- No reads in the lookback → no baseline: the rule **holds** (value `baseline=none`), it neither raises nor clears.

## R3. Arms and severity (BVJ-4, BVJ-5, BVJ-6)

| Arm | Condition at read s | Severity |
|---|---|---|
| Jump | some channel with `ratio ≥ 10` and `ratio ≥ 10 × baseline` | warning |
| Jump, critical | the same channel also has `ratio ≥ 100` | critical |
| Outlier count | `count ≥ max(2 × baseline_count, baseline_count + 5)`, `count` = channels with ratio ≥ 10 over all sites | warning |

- "New channel" (the owner's word) is any channel meeting the jump condition. Whether it was already ≥ 10× before does not matter: a channel going 980× → 42,205× (B-silu at 21,000) or 26× → 351× (C at 2,513) is the same signal.
- The minimum rise of 5 keeps small counts quiet: 1 → 2, 2 → 4, 3 → 6 and 4 → 8 do not raise; 0 → 5 does.
- One rule, one active alarm: the highest severity of the arms that hold. The event names every arm that holds.

Margins (Part E):

| Constant | Value | Incident | Nearest non-incident |
|---|---:|---|---|
| outlier level | 10× | B-silu jumped channels 27–980× | (a level, used by both arms) |
| rise factor | 10× | B-silu ≈ 7,700× at 19,800 | the slow-growth cases rise at most 5.5× in 1,000 steps (AgG3, before its gate) and 1.2× (clip1 channel 31); every jump in E3 rises ≥ 13× (C, 26.35× → 350.8×) and every healthy-run jump ≥ 155× (B-leakyall `[43]`), because each starts from a decayed channel (E4) |
| critical level | 100× | B-silu 154× at its first read, 980× at 20,000 | clip1 ≤ 96.9× (through 22,000), B-leaky ≤ 62.7×, B-leakyall ≤ 45.9× and ≤ 80.4× (log bounds); clip2 104.9×, clip5 310× (survivors, Q-1) |
| count rise | ×2 and +5 | B-silu 5 → 11 | clip1 +5 at ×1.36; B +4; B-leaky +4; B-leakyall +3; AgG3 2 → 4 |
| lookback | 1,000 trainer steps | — | equal to the spike rules' look-back and one tenth of B's LR period |

## R4. Gate (BVJ-7)

- Reads before `TrainingHealthConfig.learningGateTrainerStep` (`lr_warmup_steps + training_health_learning_grace_steps`, `Training/TrainingHealth.swift:372`; 2,000 for every evidence run) are neither judged nor stored. The first read at or after the gate is the first baseline.
- Why: BN running statistics start at the init (variance 1) and converge during warm-up. B went 1.70× → 24.5× (`stem.bn[102]`) and B-leaky 0.96× → 80.4× (`stem.bn[90]`) between 1,000 and 2,000; both settle. A run with `lr_warmup_steps` 0 and the default grace gates at 1,000 and would warn on those (documented, not handled).
- Same gate as rule 4's not-learned form and rule 13. No new parameter.

## R5. Resets (BVJ-9)

| Event | History | Active alarm |
|---|---|---|
| process start: CLI run, exact resume, a new segment, GUI Play-and-Train start (also a continue after Stop) | empty (one monitor per run, alarm plan D2) | none |
| GUI promotion (announced rewind) or an unannounced rewind (`TrainingHealthMonitor.noteTrainerClockRewind`, `recordStep`) | cleared in `resetForTrainerClockRewind` (`Training/TrainingHealth.swift:1258`): the reads describe discarded weights | kept; clears only by its clear rule against post-rewind reads |
| check interval, checkpoint pass, value-FC1 read | untouched (live tier only, BVJ-8) | — |

- After a reset, the first read is a baseline only, so the first 50 trainer steps of every process are not judged (Q-6). Rule 8's level check still applies to them.

## R6. Sustain, clear, worsen (BVJ-10, BVJ-11)

- Raise: immediate (one evaluation), as rules 8 and 9.
- Escalate: warning → critical when a jumped channel reaches 100× while the jump still holds against the lookback (clip2: warning 19,850, critical 19,900).
- Clear: neither arm holds against the current lookback on 2 evaluations spanning ≥ 1 trainer step (`layerHealthRuleClearSustain`, `Training/TrainingHealth.swift:418`). Because the baseline is the lookback minimum, an alarm stays active until the pre-jump reads leave the lookback: about 1,000 trainer steps after the last jump. That makes the alarm an event that lasts one lookback; the persistent damage stays visible in rule 8, the `[LAYER-HEALTH]` lines and every checkpoint (Q-4 for the alternative).
- Stop: as every rule, the stop is decided at the evaluation that makes it active (CLI: `byEvaluator`; GUI: the caller), so the clear timing does not delay a stop.
- Worsen (counted rules): the count is the outlier count (≥ 10× channels). B-silu: 11 at 20,000 → 53 at 21,000 → one `worsen` line per check interval.

## R7. Rule 14 rather than a change to rule 8 (BVJ-1, BVJ-16)

- Separate input (per-channel history), separate semantics (change, not level), a critical level rule 8 does not have, separate clear, and a separate action: the owner may want to stop on the jump while the 1,000× level stays log-only. `results.json`, the lineage `health_alarms` summary and grep all need to tell them apart.
- Rule 8 is unchanged: warning at ≥ 1,000×, clear < 300× on 2 evaluations. Its B-silu raise stays at 20,600.
- Appended at the end of rule order (rule 14, `ruleOrder` 13) so rules 9–13 keep their numbers in docs, tests and logs. Stop decisions take the first qualifying alarm in rule order; a jump and an earlier-ordered critical in one evaluation stop on the earlier one, which changes only the name on the stop line.

## R8. Live tier only (BVJ-8)

- `LayerHealthDigest.Tier.feeds(.batchNormRunningVarianceJump)` is true for `.live` only. A save's checkpoint pass reads the same BN state at the same step as that step's live read and adds nothing; GUI checkpoint passes are detached and can arrive late, which would need stale handling for no gain.

---

# Part D — Design

## D1. Data (BVJ-13)

What a live read already has: `ChessTrainer.readLayerHealthLiveState()` returns every BN site's γ, β, running mean and **running variance** (`Training/LayerHealth.swift:135-137`, scope `batchNormStateOnly`). `LayerHealth.batchNormSiteHealth` reduces running variance to max, its channel, median and max/median (`:583-602`); `LayerHealthDigest.init(summary:)` keeps only the largest site's max/median (`Training/TrainingHealth.swift:751-770`). So today **no per-channel ratio and no count survive past the summary**. New:

```swift
// Training/LayerHealth.swift
extension LayerHealth {
    /// The one definition of a channel's ratio to its site median.
    static func runningVarianceRatios(_ runningVariance: [Float]) -> (median: Double?, ratios: [Double?])
    /// Ratio at or above which a channel counts as an outlier (rule 14, `rvOver10xMedian=`).
    static let runningVarianceOutlierRatio: Double = 10
}

/// Every BN site's per-channel ratios from one read. Not Codable: never written to results.json.
struct BatchNormRunningVarianceProfile: Sendable, Equatable {
    struct Site: Sendable, Equatable {
        let site: String
        /// nil when the site's median is not positive; an element is nil for a non-finite variance.
        let ratios: [Float?]?
    }
    let sites: [Site]            // batchNormSites(for:) order
    var outlierCount: Int { get } // channels with ratio ≥ runningVarianceOutlierRatio
}
```

- `BatchNormSiteHealth` gains `runningVarianceOutlierCount: Int` (`running_variance_outlier_count`); `LayerHealthSummary` gains `runningVarianceOutlierCount` (the sum). Both encoded, so `results.json`'s `layer_health` records carry them; the profile is not.
- `LayerHealth.summarizeLiveState` (`:354`) returns the profile beside the summary (one pass over the same tensors, on the same GCD queue `LayerHealthLog.runOffPool` already uses).
- `LayerHealthLog.LiveOutcome` (`Training/LayerHealthLog.swift:32`) gains `runningVarianceProfile`.
- `LayerHealthDigest` gains:

```swift
struct RunningVarianceChannels: Sendable, Equatable {
    enum Coverage: Sendable, Equatable {
        /// In-app: every channel of every site.
        case everyChannel(BatchNormRunningVarianceProfile)
        /// Offline: only the line's largest channel; every other channel is at most `ratio`.
        case largestOnly(site: String, channel: Int, ratio: Double)
    }
    let coverage: Coverage
    /// Channels at ≥ 10× their site median; nil offline when the line has no `rvOver10xMedian=`.
    let outlierCount: Int?
}
let runningVarianceChannels: RunningVarianceChannels?   // live tier only; nil otherwise
```

- `TrainingHealthReads.liveLayerHealth(from:)` (`Training/TrainingHealthReads.swift:14`) builds it from the outcome's profile; a live outcome with a summary but no profile is a code bug (`preconditionFailure`, the project's convention for an impossible state).
- Memory: one history entry is one `Float?` per channel. 1,168 channels (the B-family nets) × 20 reads ≈ 183 KB; a 6,500-channel net ≈ 1 MB.

## D2. Evaluator (BVJ-5, BVJ-6, BVJ-7, BVJ-9, BVJ-10)

- `TrainingHealthRule.batchNormRunningVarianceJump = "bn_running_variance_jump"`, `ruleOrder` 13, `hasCriticalLevel` / `hasWarningLevel` true, `raiseSustain .immediate`, `clearSustain layerHealthRuleClearSustain`; every exhaustive switch over the rule gets its case (the compiler lists them: `TrainingHealth.swift` lines 23–123, 199–267, 665–683, 1378–1395, 1411–1430; `TrainingParameters.swift:1825`, `:3266`; `SessionCheckpointFile.swift:1019`, `:1036`; `SessionParameterResume.swift:484`; `TrainingHealthRule+Display.swift:18`, `:38`).
- Constants in `TrainingHealthThresholds` (with the measured margins as doc comments):

```swift
// Rule 14 — bn_running_variance_jump.
static let batchNormRunningVarianceJumpOutlierRatio = LayerHealth.runningVarianceOutlierRatio   // 10, one declaration
static let batchNormRunningVarianceJumpRiseFactor: Double = 10
static let batchNormRunningVarianceJumpCriticalRatio: Double = 100
static let batchNormRunningVarianceJumpLookbackSteps = 1000
static let batchNormRunningVarianceOutlierCountRiseFactor = 2
static let batchNormRunningVarianceOutlierCountMinimumRise = 5
/// Defensive cap on stored reads; the lookback holds at most 20 in-app.
static let batchNormRunningVarianceJumpHistoryCapacity = 64
```

  `batchNormRunningVarianceJumpOutlierRatio` is defined as `LayerHealth.runningVarianceOutlierRatio` (one declaration, not two equal literals).
- Pure reading, `enum BatchNormRunningVarianceJump` (new file `Training/BatchNormRunningVarianceJump.swift`):

```swift
struct JumpedChannel: Sendable, Equatable {
    let site: String; let channel: Int
    let baseline: Double          // lowest ratio in the lookback, or an upper bound of it
    let baselineIsExact: Bool     // false offline unless the channel was the largest on every line
    let ratio: Double
}
struct Reading: Sendable, Equatable {
    let jumped: [JumpedChannel]   // ratio descending
    let largestRise: JumpedChannel?  // over channels ≥ 10×, whether or not they jumped (check line)
    let outlierCount: Int?
    let outlierBaseline: Int?
}
static func read(_ current: LayerHealthDigest.RunningVarianceChannels,
                 history: [HistoryEntry], trainerStep: Int) -> Reading?   // nil: no read in the lookback
```

  A past read's bound for a channel is its exact ratio (`everyChannel`, or `largestOnly` naming that channel) or the read's largest ratio (`largestOnly`, another channel). The current read's candidates are every channel (`everyChannel`) or the largest only (`largestOnly`). A mismatch between a history entry's site layout (names, channel counts) and the current one is a code bug within one monitor: `preconditionFailure` naming the site.
- `State` gains `runningVarianceJumpHistory: [HistoryEntry]` (trainer step + channels). After the rule is judged on a live observation at or past the gate (and not stale), the read is appended and entries older than `trainerStep − lookback` are dropped (the same "judged, then included" order as rule 4's running minimum, `:1333-1336`).
- `resetForTrainerClockRewind` empties the history.
- Assessment: before the gate → `.hold(value: "gate=<n>")`; no read in the lookback → `.hold(value: "baseline=none")`; an arm holds → `.raise`; otherwise `.clear`. No `runningVarianceChannels` in a live digest, or no live digest (a failed live read) → `.noData`.

## D3. Rendering (BVJ-14)

- `[LAYER-HEALTH]` compact line (live and checkpoint headline): `rvOver10xMedian=<n>` after `rvMaxOverMedian=` (`LayerHealthSummary.compactLine`, `Training/LayerHealth.swift:1081`).
- Checkpoint table: one more column at the end, `rv≥10x`, the site's outlier count (`detailedLines`, `:1175`). The offline table parser reads tokens 0–3 only (`TrainingHealthLogReplay.tableRow`), so an extra trailing column is safe.
- Events (the existing format, `TrainingHealthLog`; the values below show the shape and are illustrative, not measured):

```
[ALARM] health raise rule=bn_running_variance_jump severity=critical trainerStep=19800 value=jumped=1 outliers=6/5 threshold=jump>=10xmin&ratio>=100 action=log lr=0.384 mom=0.85 channels=blocks.2.bn1[76]:0.02->154.0
[ALARM] health escalate rule=bn_running_variance_jump severity=critical trainerStep=19900 since=19850 value=jumped=1 outliers=7/5 threshold=jump>=10xmin&ratio>=100 …
[ALARM] health raise rule=bn_running_variance_jump severity=warning trainerStep=20000 value=jumped=0 outliers=11/5 threshold=outliers>=2xmin&+5 …
[ALARM] health worsen rule=bn_running_variance_jump severity=critical since=19800 trainerStep=20650 value=jumped=9 outliers=27/5 was=11 …
```

  - `value`: `jumped=<n> outliers=<count>/<baseline>` (`--` for an unknown count offline).
  - `threshold`: the arms that hold, joined by `|`: `jump>=10xmin&ratio>=100`, `jump>=10xmin&ratio>=10`, `outliers>=2xmin&+5`.
  - `detail`: `channels=` up to 8 jumped channels, ratio descending, `site[c]:<baseline>-><ratio>` (offline `<=<bound>` when the baseline is an upper bound), then `more=<n>` when there are more.
- `[HEALTH] check` line (`TrainingHealthLog.checkLine`, `Training/TrainingHealthLog.swift:140-198`): `rvRiseMax=<factor>` (largest rise of any channel at ≥ 10×, over the interval; `--` when none; `inf` for a zero baseline) and `rvOutliers=<count>/<baseline>` (the latest live evaluation's). The monitor accumulates `rvRiseMax` like `gradMaxRatio` (`noteRatios`, `Training/TrainingHealthMonitor.swift:556`). These are what V-1 … V-5 read to measure margins without per-channel log lines.
- `[HEALTH] config` lists the new action automatically (`configLine` iterates `allCases`).
- Command-line paths also write raise, escalate and stop lines to stderr (existing).

## D4. Paths

| Path | Change |
|---|---|
| `--replay-corpus`, `--train-vs-uci` | none beyond D1: the live outcome they already pass to `TrainingHealthReads.liveLayerHealth(from:)` carries the profile |
| GUI Play-and-Train, GUI `--train` | the same; promotion rewinds already reach `resetForTrainerClockRewind` |
| `--replay-health-log` | D5 |

Cost: per live evaluation one pass over ≤ 20 × channels values under the evaluation lock (≪ 1 ms for 1,168 channels); reported in `cost_ms=` as today.

## D5. Offline replay (BVJ-12)

- `TrainingHealthLogReplay.compactHealth` (`Training/TrainingHealthLogReplay.swift:369-445`) keeps the channel of `rvMaxOverMedian=value@site[channel]` (parsed and dropped today) and parses `rvOver10xMedian=` when present. A live line gives `RunningVarianceChannels(coverage: .largestOnly(…), outlierCount: <field or nil>)`.
- Jump arm offline = a **lower bound** on the app: only the largest channel can trigger, judged against the past lines' largest ratios as upper bounds. It never raises where the app would not; it can raise later or not at all (B-silu 19,900 vs 19,800; clip2 / clip5 / B-leaky / B-leakyall / clip1 not at all).
- Count arm offline: exact at the logged cadence when the lines carry `rvOver10xMedian=`; no data on older logs.
- Header lines (`headerLines`, `:566`): `bn_running_variance_jump: the jump arm sees only each live line's largest channel (a lower bound on the app's); the outlier-count arm has no data on <n> of <m> live lines (no rvOver10xMedian=)`.
- GUI logs (live lines at `[STATS]` cadence) get the rule too; logs without live lines get no data.

---

# Part K — Parameters (1), with the full CLAUDE.md checklist (BVJ-15, BVJ-18)

| Swift type | id | Type, default, range | `absentValue` | Why |
|---|---|---|---|---|
| `TrainingHealthActionBatchNormRunningVarianceJump` | `training_health_action_bn_running_variance_jump` | Int, **0** (log), 0…2 | `.currentSetting` | operational: does not change training math; the same as the other 13 actions |

Name "Health Action: BN Running Variance Jump", `category: "Health"`, `liveTunable: true`. Description: "What the bn_running_variance_jump alarm (a batch-norm channel's running variance at ≥ 10× its site's median that rose ≥ 10× above its lowest value in the previous 1,000 trainer steps — critical when the channel is at ≥ 100× — or the number of channels at ≥ 10× rising to at least twice, and at least 5 more than, its lowest in the previous 1,000 trainer steps; judged from the learning gate on) does besides logging: 0 = log only (default), 1 = also stop the run while it is active at critical, 2 = also stop the run while it is active at any severity. A stop ends a command-line run through its final save (exit status 35) and suspends GUI training."

Thresholds are constants, not parameters (alarm plan OD-14).

1. **Declare.** `@TrainingParameter` after `TrainingHealthActionLegalMassStall` (`Training/TrainingParameters.swift:1594-1605`); add to `allKeys` (`:3110` neighborhood). `absentValue: .currentSetting`.
2. **Singleton.** Stored property `trainingHealthActionBatchNormRunningVarianceJump` (pattern `:1993`), `init` read (`:2124`), `collectValues` (`:2242`), `applyOne` (`:2522`), `trainingHealthAction(for:)` (`:1825` switch), `trainingHealthActionKeyPath(for:)` (`:3266` switch). `TrainingHealthActions` gains the field, its `init` line and both subscript cases (`Training/TrainingHealth.swift:199-267`). Update `TrainingHealthAlarmsEnabled`'s description ("thirteen rules" → "fourteen", `:1418`).
3. **`parameters.json`.** Verify the key in `--show-default-parameters` and the `--create-parameters-file` → edit → reload round trip (macro-generated; nothing is hand-listed in the CLI path).
4. **Session (`.dcmsession`).** Optional `trainingHealthActionBatchNormRunningVarianceJump: String?` on `SessionCheckpointState` (`Persistence/SessionCheckpointFile.swift:635-640`) and both cases of `subscript(savedTrainingHealthActionFor:)` (`:1009-1045`). `buildCurrentSessionState` already writes every rule through `TrainingHealthActions` (`App/SessionController+Checkpoint.swift:1408`). One `case` in `SessionParameterResume.applyGuiSession`'s rule switch (`App/SessionParameterResume.swift:467-497`). An older session without it keeps the current setting (logged `[RESUME-DIFF]` as today).
5. **`results.json`.** `alarm_config.actions` gains the key automatically (`TrainingHealthActions.encode` iterates `allCases`); `alarms` carries the events; `layer_health` records carry `running_variance_outlier_count`.
6. **Runtime log.** `[HEALTH] config … actions=…,bn_running_variance_jump:<action>`; `[PARAM]` line on a GUI edit (existing).
7. **UI.** The Health tab's rule list is `ForEach(TrainingHealthRule.allCases)` (`App/UpperContentView/TrainingHealthTab.swift:25`) over `TrainingSettingsPopoverModel.trainingHealthActionsValue` (`TrainingHealthActions`), so the row appears with the new field. Display name "BN running-variance jump" and description "A BN channel's running variance rose 10× within 1,000 steps to ≥ 10× its site median (critical ≥ 100×), or the count of such channels doubled (+5)" in `App/UpperContentView/TrainingHealthRule+Display.swift`.
8. **Live tunability.** The GUI resolves `TrainingHealthConfig` at every evaluation and reads the actions when a result arrives (existing); the CLI paths read the run-start snapshot (existing).
9. **Renames.** None. Rule 8's id and parameter are unchanged.

---

# Part T — Touch points

| # | File | Change | Phase |
|---|---|---|---|
| T1 | `Training/LayerHealth.swift` | `runningVarianceRatios(_:)`, `runningVarianceOutlierRatio`, `BatchNormRunningVarianceProfile`, per-site and total outlier counts, `summarizeLiveState` returns the profile, compact field, table column | P1 |
| T2 | `Training/BatchNormRunningVarianceJump.swift` (new) | D2 pure reading | P1 |
| T3 | `Training/TrainingHealth.swift` | rule case and switches, constants, `LayerHealthDigest.RunningVarianceChannels`, `Tier.feeds`, `State` history, assessment, rewind reset, `TrainingHealthActions` field; "thirteen" → "fourteen" in comments | P1 |
| T4 | `experiments/20261005-lr-schedule-ab/bn_liveness.py` | `read_running_variance` and `--write-running-variance-fixture` (BVJ-17) | P1 |
| T5 | `DrewsChessMachineTests/Resources/TrainingHealthIncidents/TrainingHealthRunningVarianceCheckpoints.json` (new) | X3 fixture, in the test target's Copy Bundle Resources | P1 |
| T6 | `Training/LayerHealthLog.swift`, `Training/TrainingHealthReads.swift` | profile through `LiveOutcome` into the digest | P2 |
| T7 | `Training/TrainingHealthMonitor.swift`, `Training/TrainingHealthLog.swift` | `rvRiseMax` / `rvOutliers` counters and check-line fields; event detail rendering | P2 |
| T8 | `Training/TrainingHealthLogReplay.swift` | D5 | P2 |
| T9 | `Training/TrainingParameters.swift`, `Persistence/SessionCheckpointFile.swift`, `App/SessionParameterResume.swift` | Part K | P2 |
| T10 | `App/UpperContentView/TrainingHealthRule+Display.swift` | display name, description | P3 |
| T11 | `documentation/training-health-alarms.md`, `CLAUDE.md`, `documentation/plans-active/TRAINING_HEALTH_ALARMS_PLAN.md`, `CHANGELOG.md`, `ROADMAP.md` | P4 docs | P4 |

Verified untouched: the trainer, the training graph, `ChessTrainer.readLayerHealthLiveState` (it already reads running variance), the step-line formats, the lineage schema (`TrainingHealthSegmentSummary` records any rule by id; `Persistence/LineageRecordSchema3.swift:234-304`), the behavior fingerprint, `TrainingAlarmController` (no banner detector).

---

# Part X — Tests

### X1. Pure (`DrewsChessMachineTests/BatchNormRunningVarianceJumpTests.swift`, new)
- `runningVarianceRatios`: median of even / odd counts, non-finite variance → nil ratio and excluded from the median, non-positive median → nil site; the existing `rvMaxOverMedian` values are unchanged on every `LayerHealthTests` fixture.
- Thresholds as tables, each on both sides of the boundary: ratio 9.99 / 10; rise 9.99× / 10×; critical 99.9 / 100; count (baseline → now): 1→2, 2→4, 3→6, 4→8, 4→9 (no), 5→9 (no), 5→10 (yes), 0→4 (no), 0→5 (yes), 11→53 (yes).
- Lookback bounds: a read at `s − 1000` counts, one at `s − 1001` does not; the baseline is the minimum, not the previous read (a read at 0.02 followed by reads at 50 → a read at 154 jumps).
- Offline coverage: a `largestOnly` current read jumps only when its ratio ≥ 10 × the smallest past bound; a past read naming the same channel gives an exact baseline (`baselineIsExact`).
- The B-silu live-line sequence 18,800–19,950 (values copied from the excerpt) gives the first jump at 19,900 offline.

### X2. Evaluator (`DrewsChessMachineTests/TrainingHealthRunningVarianceJumpRuleTests.swift`, new)
- Raise immediate; escalate warning → critical; `worsen` on a rising outlier count, at most once per check interval; clear only after 2 evaluations ≥ 1 step apart with neither arm holding; an alarm raised at s with no further jumps clears once s's pre-jump reads leave the lookback.
- Gate: reads before `learningGateTrainerStep` give `hold` and are not stored (a jump measured against a pre-gate read never raises).
- Rewind: after `resetForTrainerClockRewind` a read that would jump against a pre-rewind read holds (`baseline=none`); the active alarm survives the rewind and clears against post-rewind reads.
- Stale observation (older than the newest applied) is neither judged nor stored.
- Checkpoint-tier and value-FC1 digests never feed the rule and never count it as `nodata`.
- Stop: `stop_on_critical` stops on the critical jump and not on a warning; `stop_on_any` stops on either arm.

### X3. Real data (`DrewsChessMachineTests/TrainingHealthRunningVarianceIncidentTests.swift`, new)
Fixture `TrainingHealthRunningVarianceCheckpoints.json`: for each checkpoint below, every BN site's `running_var` (float32 values written exactly, as `TrainingHealthBSiluBatchNorm.json` writes γ and β), with file name, file SHA-256, `model_id` and `cum_trainer_step`; generated by T4, read-only from the checkpoints. About 75 checkpoints × about 1,170 values ≈ 1 MB, smaller than `TrainingHealthIncident-Bsilu.log` (1.3 MB).

| Run | Checkpoints | Asserted (at 1,000-step spacing, gate 2,000) |
|---|---|---|
| B-silu | 2k … 22k | first event: **raise critical at 20,000** naming `blocks.2.bn1[76]` and `[34]`, threshold includes the count arm (5 → 11); nothing at 2k–19k; `worsen` at 21,000 (11 → 53) |
| clip1 | 19k–23k, 31k–33k, 39k–40k | one raise, **warning** at 20,000 (`blocks.2.bn1[76]` 0.02 → 38.9); never critical; no event from channel 31; no count raise at 32k (+5, ×1.36) |
| clip2, clip5 | 19k–23k | raise critical at 20,000 (129.5×; 513.1×) |
| B-leaky | 1k–3k, 20k–22k | warning at 21,000 only |
| B-leakyall | 1k–3k, 12k–15k, 29k–32k | raise warning at 13,000, clear at 15,000; raise warning at 30,000, clear at 32,000 (contiguous spans, so a hold across a gap does not keep the first alarm active) |
| B | 1k–4k, 21k–22k, 31k–32k | nothing (the 1k → 2k jump is before the gate; count rises +4) |
| A | 2k–6k | nothing |
| AgG3 | trainer saves 1,607, 2,831, 4,655, 5,261 | nothing (gate 2,000: 1,607 is not stored, 4,655 has no read in its lookback, so one judged read: 5,261 against 4,655) |

Each test feeds the fixture's vectors through `LayerHealth.runningVarianceRatios` into live digests, in step order, through the real evaluator. In the app the evaluation is every 50 steps; these tests pin the 1,000-step behavior, V-1 … V-4 pin the 50-step behavior. Expected steps are this plan's; a difference is reported to the owner, never absorbed by changing an expectation.

Written before the rule (owner's practice for regressions): the B-silu test is added first and fails (no such rule), then passes unmodified.

### X4. Layer health (`DrewsChessMachineTests/LayerHealthTests.swift`, additions)
- Per-site `runningVarianceOutlierCount` and the total; `rvOver10xMedian=<n>` on the compact line; the table's `rv≥10x` column; `LayerHealthSummary` JSON round trip with the new key.
- The profile from `summarizeLiveState` equals `runningVarianceRatios` per site.

### X5. Offline replay (`TrainingHealthIncidentReplayTests.swift`, `TrainingHealthLogReplayTests.swift`, additions)
- B-silu excerpt: `bn_running_variance_jump` raises **critical at 19,900**, before the offline `gradient_spike` (20,600) and `illegal_mass` (20,700); all existing B-silu expectations unchanged.
- A, R7, R8: still no event (existing tests `testArmARaisesNothing`, `testR7RaisesNothing`, `testR8RaisesNothing` guard it unchanged). Arm C: no `bn_running_variance_jump` event; existing C expectations unchanged. B: no event.
- Parsing: `rvMaxOverMedian=…@site[ch]` keeps the channel; `rvOver10xMedian=` parsed; a line without it gives a nil count; header line present.

### X6. Existing tests the new rule forces to change (owner's standing instruction: edits forced by an approved design change are made and reported)
- `TrainingHealthParameterTests.swift:35-49` (`actionKeyIDs` gains the rule), `:82` (`healthIDs.count` 16 → 17), `:86-101` (one more `absentValue` assertion).
- `TrainingHealthLogTests.swift:88` (the `[HEALTH] config` string gains `,bn_running_variance_jump:log`).
- Any `LayerHealthTests` assertion of a full compact line or table row (the new field / column); only `contains` checks were found (`LayerHealthTests.swift:500`), which do not change.
No assertion of an existing rule's behavior changes.

### X7. Run scope
Targeted: the X1–X5 classes plus `TrainingHealthEvaluatorTests`, `TrainingHealthLogTests`, `TrainingHealthParameterTests`, `TrainingHealthLogReplayTests`, `TrainingHealthIncidentReplayTests`, `TrainingHealthReplayCLITests`, `CliTrainingRecorderAlarmTests`, `TrainingHealthGuiValidationTests`, `LayerHealthTests`, `SessionCheckpointFile`-related tests. Full suite once at the end (persistence and session fields change).

---

# Part V — Validation

All on the frozen build of the implementing commit, `--seed 20261005`, corpus `20260624-192615-w3aA5b`, `--policy-tail-precision fp32_from_pre_bn`, every action `log` (the default), the launch shape of the relcap V-1 (`experiments/20261005-lr-schedule-ab/README.md`, "Relative gradient cap validation V-1 and V-3"). They can run beside live training.

- **V-1 — B-silu at 50-step resolution.** `--resume-exact` from `20261005-lrBsilu-cyc1-replay-step18000.safetensors`, `parameters-B.json`, to trainer step 21,000. **Pass:** `bn_running_variance_jump` is critical at **19,800** naming `blocks.2.bn1[76]` (a warning of this rule before 19,800 is acceptable with its channel and values reported); the count arm appears by 20,000; every other rule's events equal relcap V-1's (`gradient_spike` 19,800 / 19,850 / 20,600, `bn_running_variance_runaway` 20,600, `dead_channels` and `loss_spike` 20,650, `illegal_mass` 20,700, `policy_offset_drift` 18,100); every `[REPLAY]` line equals B-silu's and the 21,000 checkpoint is byte-identical to B-silu's (`scripts/safetensors_tensor_compare.py`, exit 0) — the observer changes nothing. Report: the step and values of the first raise, the `rvRiseMax=` / `rvOutliers=` of every check line, the count arm's first step.
- **V-2 — clip1.** The same resume with `parameters-Bsilu-clip1.json` (cap 1.0) to 21,000. **Pass:** a warning naming `blocks.2.bn1[76]`, never critical; report the step and the channel's peak ratio.
- **V-3 — the expected default cap.** The same resume with the relative cap in clip mode, k = 3 (`parameters-B-relcap-v3.json`'s cap settings on B-silu), to 23,000 — doubles as the relative-cap plan's unrun V-2. Report the rule's severity. This is the case that matters once the relative cap's P5 lands.
- **V-4 — healthy jumps at 50-step resolution.** Exact resumes: B-leaky from 20,000 to 22,000; B-leakyall from 12,000 to 14,000 and from 29,000 to 31,000 (`parameters-B.json`). **Pass:** warnings only, each jumped channel's peak < 100× (the logs bound them at ≤ 62.7×, ≤ 45.9×, ≤ 80.4×).
- **V-5 — fresh start.** B's start net, `parameters-B.json`, 3,000 steps. **Pass:** no raise; the first judged evaluation is at 2,050 (`baseline=none` at 2,000).
- **V-6 — offline.** `--replay-health-log` on every log in E3 (A and C with both their logs together). **Pass:** B-silu and ctl15 raise critical at 19,900, relcap V-1 at 19,960; no other log raises `bn_running_variance_jump`; every other rule's events are identical to the previous build's output (diff the event lines).
- **V-7 — GUI.** One Play-and-Train session through at least one promotion (a test model is enough): the `[HEALTH] trainer clock rewound …` line is followed by a `baseline=none` hold, not a raise; the Health tab shows 14 rows; set the rule to Stop at critical, confirm the setting survives a session save and resume (`[RESUME-DIFF]` for `training_health_action_bn_running_variance_jump`).

---

# Part P — Phasing

| Phase | Content | Done when |
|---|---|---|
| P1 | T1–T5, X1–X4 (X3's B-silu test first, failing) | targeted tests pass; build has no new warnings; commit |
| P2 | T6–T9, X5, X6 | targeted tests pass; `--show-default-parameters` shows the key; commit |
| P3 | T10; V-7's display checks | commit |
| P4 | T11 docs | commit |
| P5 | V-1 … V-6 (V-7 after P3) | results written into this plan and the experiment README; owner decides Q-1 … Q-5 with the numbers |

Build and commit per phase (owner's standing order for approved multi-phase plans); full test suite once before P5.

---

# Decisions made in this plan (recommendations until the owner confirms)

- **BVJ-1 New rule, not a change to rule 8.** `bn_running_variance_jump` (rule 14); rule 8 stays a warning at ≥ 1,000× (R7).
- **BVJ-2 Ratio = channel running variance ÷ its own site's median, every BN site,** one shared function (R1).
- **BVJ-3 Baseline = the channel's lowest ratio over the reads in the previous 1,000 trainer steps,** not only the previous read (R2).
- **BVJ-4 A "new channel" is any channel meeting the jump condition,** whether or not it was already ≥ 10× (R3).
- **BVJ-5 Jump: ratio ≥ 10× and ≥ 10× its baseline is a warning; ≥ 100× is critical** (R3; Q-1, Q-2).
- **BVJ-6 Count arm: channels ≥ 10× over all sites, raise at ≥ max(2 × baseline, baseline + 5); warning** (R3; Q-3).
- **BVJ-7 Gate: the learning gate (`lr_warmup_steps + training_health_learning_grace_steps`); earlier reads are neither judged nor stored** (R4).
- **BVJ-8 Live tier only;** checkpoint and value-FC1 reads do not feed the rule (R8).
- **BVJ-9 Resets: history per monitor (empty at every process, segment and GUI start) and cleared on every trainer-clock rewind; active alarms kept** (R5; Q-6).
- **BVJ-10 Raise immediately; clear when neither arm holds on 2 evaluations ≥ 1 step apart** — the alarm lasts about one lookback after the last jump (R6; Q-4).
- **BVJ-11 The `worsen` count is the outlier count** (R6).
- **BVJ-12 Offline replay: the jump arm as a lower bound from each live line's largest channel; the count arm from the new `rvOver10xMedian=` field; no data on older lines** (D5).
- **BVJ-13 Per-channel ratios travel in a non-`Codable` profile beside the summary;** only the per-site and total outlier counts are added to the encoded summary, so `results.json` does not grow by a vector per checkpoint (D1).
- **BVJ-14 New log fields:** `rvOver10xMedian=` on `[LAYER-HEALTH]`, an `rv≥10x` table column, `rvRiseMax=` / `rvOutliers=` on `[HEALTH] check` (D3).
- **BVJ-15 Default action `log`,** like every rule; the docs add this rule's `stop_on_critical` to the list recommended for unattended runs (Q-5).
- **BVJ-16 Appended at the end of rule order** so rules 9–13 keep their numbers (R7).
- **BVJ-17 The real-data fixture holds running variances read from the evidence checkpoints, written by `bn_liveness.py`** (the existing γ/β fixture has no running variance) (E2, X3).
- **BVJ-18 All thresholds are declared constants;** the only new parameter is the action (Part K).

# Open questions for the owner

- **Q-1 Critical on the survivors.** At 100× the rule is critical on clip2 (104.9× at 19,900) and clip5 (310× at 19,850), which survived; with `stop_on_critical` they would have stopped. No level in the data separates B-silu (154× at its first read) from clip5 without fitting to one run. Accept 100×, or keep the rule warning-only (then only `stop_on_any` stops it, and that also stops clip1, B-leaky and B-leakyall)?
- **Q-2 Warnings on healthy runs.** The jump arm warns on clip1, B-leaky (once) and B-leakyall (twice) — real reactivations of decayed channels at LR peaks. Acceptable as warnings, or raise the warning level? A 50× level would silence all four at checkpoint spacing (largest 46.6×), but their 50-step peaks are unknown until V-4.
- **Q-3 Count-arm severity.** Warning, because its margin rests on one incident (B-silu +6 against healthy +3 … +5 at lower ratios). Make it critical?
- **Q-4 Clear semantics.** Event-like (clears ~1,000 steps after the last jump) as planned, or "footprint" (stays active while the jumped channels stay ≥ 10×)? Footprint would keep the clip1, B-leaky and B-leakyall warnings active for the rest of those runs (the channels stay 12–40×).
- **Q-5 Default action.** Keep `log` (every rule's default), or ship this rule with `stop_on_critical`?
- **Q-6 Blind first 50 steps.** A process's first live read is only a baseline. Accept, or take one extra live read at the start step (an observer read, wired into all three paths)?
- **Q-7 Older builds.** A lineage record whose `health_alarms` names `bn_running_variance_jump` does not decode in a build without the rule (`TrainingHealthRule` decodes by raw value). Frozen experiment builds `--resume-exact`-ing a newer checkpoint that raised it would refuse it. Accept (as for earlier rule additions), or make the summary tolerate unknown rule ids?
- **Q-8 "Outlier" in the request.** The 62 → 207 count matches channels ≥ 3× their site median. The rule counts ≥ 10× (the request's own count arm). Confirm 3× was only descriptive.
- **Q-9 Lookback.** 1,000 trainer steps (one tenth of B's LR period, the spike rules' look-back). A longer lookback would also catch a jump spread over more than 1,000 steps but no evidence run needs it. Keep 1,000?

---

# Risks

| Risk | Mitigation |
|---|---|
| The critical level is fitted to one incident | E3's table is per run; V-1 … V-4 measure 50-step levels; default action `log` (BVJ-15) |
| A jump on a healthy run stops it (`stop_on_any`) | warning vs critical split; Q-1, Q-2 |
| In-app firing step differs from this plan's (19,800) | V-1 measures it; X3 pins only the 1,000-step behavior |
| Offline replay understates the app | stated in its header (D5); lower bound by construction, never a false raise |
| History memory on wide nets | ≤ 1 MB at 6,500 channels (D1) |
| A run with `lr_warmup_steps` 0 warns on early BN convergence | documented (R4); grace is live-tunable |
| Exhaustive switches missed | the compiler enforces every switch; X6 lists the tests |
| Older builds and lineage records (Q-7) | owner decision |

# Non-goals

- Changing rule 8's level or severity, or rule 9.
- A per-channel `[LAYER-HEALTH]` line (the check-line maxima and the event detail carry what validation needs).
- Judging the jump on checkpoint passes or across processes (no baseline persisted in the trainer file).
- Explaining or preventing the dormant-channel reactivation (E4); that belongs with the relative gradient cap and the activation experiments.
