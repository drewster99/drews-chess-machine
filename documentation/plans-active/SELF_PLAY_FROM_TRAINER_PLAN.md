# Self-play from the trainer's weights, no arenas — plan

Status: **proposed** (owner request 2026-10-08). Nothing here is implemented. Implementation
starts on the owner's go; the open decisions (§2) are answered first.

## 1. Goal

A Play-and-Train mode in which:

- self-play plays the weights being trained (the trainer's current weights, copied into the
  self-play network every K trainer steps), not an arena-gated champion;
- no arena runs, so nothing is promoted, the trainer is never rewound, and no new model
  generation is minted during the run.

This is AlphaZero's scheme (Silver et al. 2018 dropped AlphaGo Zero's 55% gate and generated
games with the latest network). The arena-gated loop stays the default and unchanged.

Motivation: E-0025 (ZlrA). In 51k steps of the gated loop the champion was replaced 19 times
from 134 arenas; between promotions self-play repeats one frozen network's games while the
trainer moves on. Whether gating helps or hurts here is an open question this mode lets us
measure against R-fixedlr.

### Precedents in this code base

- `--train-vs-uci` already plays the live trainer: `syncEvalNet()`
  (`CLI/TrainVsUciRunner.swift` ~610) copies `trainer.network.exportWeights().prefix(baseCount)`
  into the play network's `loadWeights` every `evalSyncEverySteps` (default 10; lineage
  `eval_sync_every_steps`). The copy needs no pause: `ChessNetwork` serialises `loadWeights` and
  `evaluateBatched` on its one `executionQueue`, so an evaluation sees the whole old or the whole
  new weight set, never a mix.
- CHANGELOG 2026-05-10 "no-arena 23-hour run" is **not** this: it kept a fixed champion (0
  promotions) while the trainer drifted away from it.

## 2. Decisions for the owner (my recommendation first)

1. **What happens to games in progress at a sync?**
   - **(Recommended) A: they continue on the new weights.** Load between ticks with no pause,
     as train-vs-UCI does. One game can span several weight versions. The value target is the
     game's result either way; the policy target is the move played.
   - B: each game keeps the weights it started with until it ends. That needs two or more live
     self-play networks and a batched evaluation per weight version per tick. It is more GPU work
     and a much larger change to `BatchedSelfPlayDriver`.
   - Not an option: syncing through `selfPlayGate`. A pause throws away every game in progress
     (`[SP-TICK] paused: dropped N in-flight games`, 100–250 games / about 20k plies per pause in
     recent logs). At one sync per K steps, most games would never finish.
2. **Sync interval K** (trainer steps). Recommended default **10**, matching train-vs-UCI. One sync
   costs one trainer export plus one load, about 100 ms when not waiting on the weight lock. ZlrA
   ran about 1.2 s per step (8,138 steps in 9,567 s over its last segment), so K = 10 is about 1% of
   training time and K = 50 about 0.2%.
3. **Arenas in this mode.**
   - **(Recommended) None.** Auto-arena is off, and Train ▸ Run Arena and Promote Trainee Now are
     disabled with a reason. Progress is measured by the puzzle probes and the test-set results
     in every saved file.
   - Later, separate item: an evaluation-only arena against a frozen reference snapshot (no
     promotion) as an Elo trend. Not in this plan.
4. **Changing mode mid-run:** **(Recommended) no.** The parameter is not live-tunable; it takes
   effect at the next Play-and-Train start. A resume restores the saved mode like any other
   parameter.
5. **`champion.safetensors` in a session saved in this mode:** **(Recommended) keep it, holding
   the self-play network's weights at the save (the last sync).** Every reader of sessions keeps
   working, and a resume restores self-play exactly as it was. Its lineage origin is a new
   `trainer_sync` origin naming the trainer step of the sync (§4.5).
6. **Lineage for syncs:** **(Recommended) no entry per sync.** At K = 10 that would be 5,000 entries
   per 50k steps. Instead:
   - record the mode and K once (they are in the parameter snapshot);
   - record one `champion_changes` entry at segment start with a new trigger `trainer_sync_start`;
   - keep the last sync's trainer step in the champion file's record.

## 3. Design

### 3.1 Parameters (`TrainingParameters`, full 9-step checklist in CLAUDE.md)

| id | type | default | range | liveTunable | absentValue |
|---|---|---|---|---|---|
| `self_play_weight_source` | Int-backed enum | 0 | 0…1 | false | `.preFeature(0)` |
| `self_play_weight_sync_interval_steps` | Int | 10 | 1…10000 | true | `.preFeature(10)` (no effect when the source is 0) |

- The Swift enum is `SelfPlayWeightSource: Int, CaseIterable`:
  - `arenaGatedChampion = 0`: today's loop.
  - `trainerSync = 1`: this mode.
- It follows `ArenaPromotionCriterion` exactly: `init(persistedRawValue:)` traps on drift, and
  `parameterRawValueRange` is pinned by a test. Log token: `arena_gated_champion` /
  `trainer_sync`.
- Category "Self-Play". UI goes in `TrainingSettingsPopover` (Self-Play tab), with a binding and
  validation in `TrainingSettingsPopoverModel`.
- Session field `selfPlayWeightSource: String?` (log token) and `selfPlayWeightSyncIntervalSteps:
  Int?` on `SessionCheckpointState`; one `restore(...)` line each in
  `SessionParameterResume.applyGuiSession`.
- The sync interval is read live from `TrainingParameters.shared` by the training worker, the way
  `arenaAutoIntervalSec` is read today.

### 3.2 One sync function, shared by both paths

Move the body of train-vs-UCI's `syncEvalNet()` into one function, for example
`TrainerWeightSync.copyTrainerWeights(from: ChessTrainer, into: ChessMPSNetwork) async throws ->
TrainerWeightSync.Result`, which exports `prefix(baseCount)` and loads. `Result` holds the trainer
step, the trainer `ModelID` and the elapsed ms.

- `TrainVsUciRunner` and the GUI both call it, so there is one source of truth for how trainer
  weights become play weights. This follows the owner's "share logic across paths" rule.
- No behavior change for train-vs-UCI: same export, same prefix, same load. Its tests must pass
  unmodified.

### 3.3 GUI wiring (`SessionController+Training.swift`)

- **Start / resume:** when the source is `trainerSync`, sync once before self-play's first tick,
  after the trainer is built or restored. The first games then play exactly the trainer's weights.
  - On a fresh start the two already match, since the trainer forks from the champion.
  - On a resume the session's `champion.safetensors` is the last sync, at most K−1 steps behind;
    syncing at start removes that gap.
- **Training worker:** after each completed step, when `completedTrainSteps % K == 0`, call the
  sync.
  - The step is the trainer clock, so which weights a sync copies does not depend on wall time.
  - Which games use them still does, as today, since self-play and training run independently.
  - Training is not paused beyond the export, which already serialises with SGD on
    `weightAccessLock`. Self-play is not paused.
- **Arena trigger:** the auto-trigger block (worker, after each step, `triggerBox.
  shouldAutoTrigger`) is skipped entirely in `trainerSync`. The arena coordinator task is not
  started, so nothing waits on `ArenaTriggerBox`. `arenasStartedThisRun` stays 0.
- **Refusals in `trainerSync`, each with a logged reason and a disabled menu item:**
  Train ▸ Run Arena, Abort Arena, Promote Trainee Now.
- **Training suspended by a health alarm:** syncing stops with the trainer, and self-play keeps
  playing the last synced weights. A resume from suspension syncs before training continues.
- **Per-champion statistics:** `resetSelfPlayGameStatsForNewChampion()` is never called by a sync.
  The run's self-play games all belong to one continuous self-play network, so their stats
  accumulate for the whole segment.
- **Saves:** unchanged mechanics (the consistent cut pauses self-play and training as today).
  `champion.safetensors` is the self-play network's current weights. Periodic autosave has no
  arena to defer around. The `-promote` trigger never fires in this mode.

### 3.4 Model identity

No new rule is needed: a sync is a weight copy, and copies inherit. After a sync,
`network.identifier = trainer.identifier` (for example `20261008-4-Ew5t-1`). With no promotion the
trainer's generation never advances, so the whole run is one trainer ID. `[STATS]` shows
`champion=` equal to `trainer=` (the champion field then means "the self-play weights"). Add a
paragraph to `documentation/sampling-parameters.md` "Model identity".

### 3.5 Lineage and session

- New `LineageRecord.ChampionChange.Trigger.trainerSyncStart = "trainer_sync_start"`, appended
  once at segment start in this mode, after `segment_start`. Validation keeps "a GUI record
  begins with `segment_start`".
- New champion origin `.trainerSync(trainerStep: Int)`.
  - `championFileLineageRecord` builds it from the trainer's record at that step without trainer
    state (`withoutTrainerState()`), with `segment_local_step` / `cum_trainer_step` at the sync
    step.
  - This keeps the champion file's step honest: the step of the weights it holds, not the save's.
- `configuration`: add `self_play` `{weight_source, sync_interval_steps, syncs, last_sync_trainer_step}`.
  It is recorded, never inferred: a record without it reads as unrecorded. This is a lineage
  schema addition, so `LineageRecord.currentSchema` goes from 4 to 5 under the existing
  versioning rules. `scripts/dcm_lineage.py` reads it.
- `results.json` (GUI `--train`): `self_play_weight_source`, `sync_interval_steps`, `syncs`.

### 3.6 Logging

- At start: `[SELFPLAY] weights: trainer_sync every K steps (no arenas)`, or `arena_gated_champion`.
- On every step line: `spSync=(every=K n=<syncs> last=<trainerStep> ms=<last> msMax=<since last line>)`.
- One `[SELFPLAY] sync failed: <error>` line per failure. A failed sync leaves self-play on the
  previous weights and retries at the next multiple of K. Two consecutive failures raise
  `[ALARM]` (shown in the UI) and do not stop training.

### 3.7 UI

- The arena countdown chip and arena settings show "Arenas off: self-play uses the trainer's
  weights (sync every K steps)".
- The champion label reads "Self-play: trainer @ step N".
- The Self-Play settings tab gets the source picker (applies at the next start) and the K field
  (live).

## 4. Phases

Each phase: build, then commit (owner rule). Tests named here are written in the phase that
needs them.

1. **Parameters:** both keys with the full checklist, enum, session fields, resume lines, UI fields.
   Tests: `parameterRawValueRange` pin; `parameters.json` round-trip; resume applies the saved
   values and logs `[RESUME-DIFF]`.
2. **Shared sync function:** extract it from `TrainVsUciRunner` and switch train-vs-UCI to it.
   Tests:
   - after a sync, the play network's exported weights equal the trainer's `prefix(baseCount)`
     bit for bit;
   - the play network's outputs on fixed positions equal those of a fresh inference network
     loaded with the trainer's weights;
   - the existing train-vs-UCI tests pass unmodified.
3. **GUI mode:** start/resume sync, step-keyed sync in the worker, arena trigger and coordinator
   skipped, refusals, per-champion stats rule, `[SELFPLAY]` / `spSync` logging. Tests: the pure
   "is this a sync step" schedule (K = 1, K > steps, a resume at a non-multiple of K); refusal
   reasons.
4. **Lineage, session and results:** new trigger, origin, schema 5, `configuration.self_play`,
   `dcm_lineage.py`. Tests:
   - schema 4 records decode with `self_play` unrecorded;
   - a schema 5 round trip;
   - the champion file record carries the sync step;
   - a session saved in `trainerSync` resumes in `trainerSync` and syncs before the first tick.
5. **Docs:** CLAUDE.md (loop section, networks, ModelID note, lineage schema 5), `sampling-parameters.md`
   "Model identity", `training-health-alarms.md` (sync-failure alarm), CHANGELOG, and a ROADMAP
   entry (with owner permission).

## 5. Validation (done means all of these hold)

- Full test suite passes, slow forensic suites included (graph builders and persistence change).
- `--train` with `self_play_weight_source = 1`, K = 10, a 3,000-step limit, on R-fixedlr's
  parameters and fresh model:
  - no `[ARENA]` line;
  - exactly one `[SELFPLAY] weights:` line;
  - `spSync n` equals ⌊3000 / 10⌋ + 1;
  - no `[SP-TICK] paused` line outside saves;
  - `champion=` equals `trainer=` on every step line;
  - steps per hour within 5% of the same run with source 0 when arenas are excluded.
- Save mid-run, quit, resume:
  - `[RESUME]` restores the source and K;
  - the champion file's lineage step equals its last sync step;
  - the first tick after resume plays synced weights;
  - `dcm_lineage.py` prints `self_play`.
- Train-vs-UCI: a short run has the same `[VS-UCI]` lines and the same results as before the
  extraction, compared on the same seed.
- The experiment that motivates this (separate from the implementation): a run in this mode on
  R-fixedlr's settings and fresh model, compared with R-fixedlr's wide pElo at equal trainer steps.

## 6. Risks

- **No gate means a bad update reaches self-play within K steps.** AlphaZero accepted this.
  - Mitigations already in place: the health alarms (`loss_spike`, `divergence`, `gradient_spike`,
    `non_finite`), the gradient caps, and a suspension that freezes self-play on the last good
    weights.
  - Not proposed: automatic rollback.
- **Faster feedback between policy and data.** Self-play now samples from a policy that tracks the
  buffer it trains on, the effect E-0025 suspects for the value head. That is why the first
  experiment in this mode should use a fixed LR of 0.01, not the cycle.
- **Weight-lock waits.** A trainer export waits for an in-flight SGD step (`weightAccessLock`);
  probe snapshots show 0.1–1.4 s including such waits. `spSync msMax` makes this visible.
