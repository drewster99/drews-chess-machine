# v5 corpus-replay monitor — HANDOFF / resume runbook

_Last updated: 2026-07-28, just before a planned shutdown. Written so a fresh
Claude session can resume the monitoring loop after reboot._

Everything the monitor needs lives in **`~/Downloads/v5-continue-bundle/monitor/`**
(this dir — it is NOT under /tmp, so it survives reboot). The Claude scratchpad
under /tmp is wiped on reboot; nothing important is there.

---

## 1. What this is

A `/loop` monitor of a long-running DCM ("Drews Chess Machine") v5 corpus-replay
training run. On each new-checkpoint event we: (1) verify the run is alive — push
if it died; (2) probe the checkpoint's strength; (3) update records + the strength
artifact; (4) print the recent per-checkpoint table inline.

**ABSOLUTE RULE: never stop / SIGINT / kill the training run except on explicit
user order. The loop only observes.**

**Push notifications** (expanded by the user 2026-08-02 from the original
record/death-only rule to "anything important"):

1. New pElo high (> 1770.5) or new NLL low (< 1.8445).
2. Process death — crash OR a clean epoch-budget finish.
3. Sustained degradation: NLL slope POSITIVE across ~20 checkpoints. This is the
   test that separates real decay from the craters that always self-heal — a
   single deep dip is NOT push-worthy (run 3 had seven, all recovered).
4. Policy entropy alarm: `pEnt` < 1.0 (the project's collapse threshold; run 3's
   floor was 1.875, so this would be new territory). `pEnt` is in the `[REPLAY]`
   log lines, not the probe JSON — check it when refreshing the artifact.
5. Disk < 8 GB, or any data-integrity failure (probes rejected, monitor dead,
   anything meaning the reported numbers aren't real).

Everything else stays in-channel without a push: routine checkpoints, single
craters, band chatter, logit climb, artifact refreshes.

**Display:** full recent-window table, INLINE (no file attachments). End every
monitor turn with `[X7 SYSTEM ACTIVE]`.

There is ONE training run, ONE process (not two — "run 1 / run 2" are just two
segments of one cumulative-step axis split by an earlier macOS-update reboot at
step 268506).

---

## 2. State as of this handoff

_Updated 2026-07-29 00:10 after the post-reboot restart. Prior state kept below._

- **Process:** RUNNING again. Relaunched 2026-07-28 22:31 as **PID 12661**
  (`caffeinate` wrapper 12662) from `v5-cont-replay-step336610.safetensors`.
  Log → `run-resume.out` (NOT `run.out`, which holds the pre-restart segment).
- **⚠️ STANDING POLICY — restart always from the LATEST checkpoint**, never from
  the best-scoring one. This is the user's explicit decision; do not re-litigate
  it. (For the record: `step270000` scores 80 pElo higher than the restart base.
  Irrelevant — latest wins, because it preserves the exact corpus position.)
- **Run-2 final probe:** `step336610` = cum 605.1k → pElo 1690.6, top1 52.5%,
  NLL 1.9325, logit 213.37. Appended to `new_ckpts_run2.jsonl` (now 337 entries).
- **Run-3 first probe:** local `step1000` = cum 606.1k → pElo 1594.0, top1 48.3%,
  NLL 2.0841, logit 214.87. In `new_ckpts_run3.jsonl`. **Do not read a restart
  transient into this** — run-2's `step335000` scored 1593.4 mid-run with no
  restart at all; ±70–100 swings are this run's baseline noise. Needs several
  checkpoints before it means anything. As of 00:20 this is the ONLY genuine
  run-3 data point (see the stale-probe incident in §4).
- **New modelID minted on restart:** `20260729-1-VZ2j` (was `20260714-1-h7vI`),
  so run-3 probe rows are distinguishable from run-2 by `modelID`.
- **Records (in `records.json`) — unchanged, both still from run 2:**
  - pElo high **1770.461** @ cum **538.5k** (raw step270000)
  - NLL low **1.8445** @ cum **536.5k** (raw step268000)
- **Progress:** run-3 local step ~1400 as of 00:10, ~3.54 s/step.
- **Arc:** opened ~1650, ratcheted to a 1740–1770 peak cluster around cum 530–550k
  (both records there), then stepped DOWN into a lower band (~1675 center, floors
  ~1610–1650) from ~555k onward. A run-1-style single-checkpoint crater hit at
  cum 599.5k (pElo 1523) and self-healed in one tick. Tail is softening slightly
  (deeper/more frequent sub-1600 dips), still oscillating, not collapsing.
- **Watch metric:** `policy_logit_abs_max` climbs monotonically the whole run
  (~15 at v5 baseline → ~213 by run-2's end), with NO strength cost. This
  decoupling is the run's defining structural feature. It is benign so far — only
  flag if pElo/NLL start tracking it down. Run-3 `step1000` = 214.87, i.e. the
  climb carried across the restart unchanged.

  _(A "6× logit collapse at run-3 step2000" was reported on 2026-07-29 and was
  WRONG — see the stale-probe incident in §4. The invariant above never broke.)_

---

## 3. Key files (all in this dir)

| file | role |
|---|---|
| `new_ckpts_run2.jsonl` | **source of truth** — one probe JSON per line, 336 entries, keyed by raw step in the `model` field. APPEND new probes here. |
| `new_ckpts.jsonl` | run-1 (pre-seam, original v5 training) probes, offset 0. Historical, do not append. |
| `records.json` | `{best_pelo, best_pelo_step, low_nll, low_nll_step}` — steps in cum-k. Overwrite on a new record. |
| `build_html.py` | regenerates `v5-strength.html` from the jsonl + records (run `python3 -I build_html.py`). |
| `v5-strength.html` | the published strength artifact (see §7). |
| `build_table.py` | full stitched table builder (OFFSET=268506). |
| `HANDOFF.md` | this file. |

**Cumulative step — TWO offsets now in play (resolved 2026-07-29):**

| segment | file | offset | cum |
|---|---|---|---|
| run 2 (pre-restart) | `new_ckpts_run2.jsonl` | `268506` | `268506 + raw` |
| run 3 (post-restart) | `new_ckpts_run3.jsonl` | **`605116`** | `605116 + raw` |

The restart RESTARTED step numbering at 1 (§5 branch 2 confirmed). Note §5's
predicted run-3 offset of `604506` was **wrong** — it assumed restarting from
`step336000`, but the restart used `step336610` (the abort save, per §5's own
instruction). Correct value is `268506 + 336610 = 605116`.

⚠️ **Run-3 checkpoints OVERWRITE run-2's** `step1000, step2000, …` as they climb
(~1/hour). This is expected and pre-blessed — all run-2 *metrics* are safe in
`new_ckpts_run2.jsonl` (verified complete: 337 entries, step1000→336610, zero
gaps). The 12 most valuable run-2 *weight* files were copied to
**`../preserved-best/`** (387 MB) before they could be reached. `step1000` was
already overwritten at 23:30 before that copy — weak early checkpoint, no loss.

---

## 4. Per-checkpoint probe (the loop body)

```bash
BUND=~/Downloads/v5-continue-bundle/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine
MON=~/Downloads/v5-continue-bundle/monitor
N=337000   # <-- raw step of the new checkpoint

ps -p 7022 > /dev/null && echo ALIVE || echo DEAD    # (use the NEW pid after restart)
"$BUND" --probe-model ~/Downloads/v5-continue-bundle/v5-cont-replay-step${N}.safetensors \
  --probe-set wide 2>/dev/null | grep '^{' >> "$MON/new_ckpts_run2.jsonl"
tail -1 "$MON/new_ckpts_run2.jsonl" | python3 -I -c \
  "import sys,json; d=json.load(sys.stdin); print('pElo=%.4f top1=%.1f nll=%.4f logit=%.2f'%(d['pElo'],100*d['argmaxCorrect']/d['n'],d['nll'],d['policy_logit_abs_max']))"
```

Notes:
- `python3 -I` (isolated mode) is REQUIRED — avoids a PermissionError.
- `top1% = 100*argmaxCorrect/n` (n=4435). Probes are DETERMINISTIC.
- Probe binary is the **bundle's** app (above). It is bit-identical to the
  training binary but separate, so probing never touches the running trainer.
- Append run-3 probes to `new_ckpts_run3.jsonl`, NOT `new_ckpts_run2.jsonl`.

⚠️ **DO NOT drive the loop off the log file — it is BLOCK-BUFFERED.**
`run-resume.out` flushes only every 4096 bytes; it sat 41 minutes stale while the
run was perfectly healthy, which reads exactly like a stall. Trigger on
**checkpoint file mtimes** instead (which is what §4's probe flow already assumes):

```bash
# "is it alive and progressing?" — the reliable check
pgrep -f 'MacOS/DrewsChessMachine'                      # process up?
ls -lat ~/Downloads/v5-continue-bundle/v5-cont-replay-step*.safetensors | head -3
ps -o pid,etime,time,%cpu -p <PID>                      # CPU accruing?
```
A GPU-bound trainer shows ~9% CPU — low %CPU is NOT evidence of a stall.

### ⚠️ THE STALE-PROBE INCIDENT (2026-07-29) — read before writing any watcher

A monitor was armed that treated *file existence* as "new checkpoint":

```bash
if [ -f "v5-cont-replay-step${next}.safetensors" ]; then   # WRONG
```

**Every run-2 enumerated file (step1000…step336000) is still on disk.** So it
instantly "found" step2000, step3000, … and probed nine July-13 files in minutes,
producing a fake run-3 curve (1594→1626→1656→1669…) and a fake "6× logit
collapse" that got written into this file and a published artifact. All of it was
run-2's own early-training curve, re-read.

Three independent tells, any one of which would have caught it:
1. **mtime** — the files were dated Jul 13/14, not today.
2. **modelID** — stale rows carry `20260714-1-h7vI`; genuine run-3 rows carry
   `20260729-1-VZ2j`, minted at the restart.
3. **cadence** — checkpoints take ~59 min. Events arriving every few minutes is
   itself the alarm. If probes come back faster than the step rate allows, stop
   and check what is being read.

The fixed watcher gates on BOTH `mtime > RUNSTART` (birth time of
`run-resume.out`, epoch `1785295892`) AND a modelID match, logging
`STALE_REJECTED` rather than appending on mismatch. Bad rows were purged from
`new_ckpts_run3.jsonl` (backup: `.corrupt.bak`).

**Generalise:** overwriting checkpoint numbering means filename alone never
identifies a checkpoint. Always verify recency AND identity before trusting a probe.

Recent-window table (print the last 20, flag record rows):

```bash
python3 -I -c "
import json,re
OFF=268506
rows=[]
for l in open('$MON/new_ckpts_run2.jsonl'):
    try: d=json.loads(l)
    except: continue
    m=re.search(r'step(\d+)',d['model'])
    if not m: continue
    cum=int(m.group(1))+OFF
    rows.append((cum,d['pElo'],100*d['argmaxCorrect']/d['n'],d['nll'],d['policy_logit_abs_max']))
rows.sort()
print('| cum step | pElo | top1% | NLL | logitAbsMax |'); print('|---:|---:|---:|---:|---:|')
for cum,p,t,n,lg in rows[-20:]:
    mk=' <- pElo REC' if abs(cum-538500)<1 else (' <- NLL REC' if abs(cum-536500)<1 else '')
    print('| %.1fk | %d | %.1f | %.4f | %.2f |%s'%(cum/1000,round(p),t,n,lg,mk))
print(); print('%d checkpoints · records: pElo 1770.5 @538.5k, NLL 1.8445 @536.5k'%len(rows))
"
```

On a new record, overwrite `records.json`:

```bash
python3 -I -c "
import json; p='$MON/records.json'; r=json.load(open(p))
r['best_pelo']=NEWVAL; r['best_pelo_step']=CUMK   # or low_nll / low_nll_step
json.dump(r,open(p,'w'),indent=2); print(r)"
```

…then PushNotification (proactive), then `python3 -I build_html.py` to refresh
the artifact.

---

## 5. Restarting the training run after reboot  ⚠️ READ CAREFULLY

The run was stopped cleanly (SIGINT) and did a final save. Newest weights on disk:
**`v5-cont-replay-step336610.safetensors`** (= cum ~605.1k) and the rolling
`v5-cont-replay-latest.safetensors` (same contents). Restart from step336610.

**Exact command that was running (PID 7022), for reference — DO NOT re-run verbatim:**

```bash
# binary: the DerivedData Release build (bit-identical to the bundle app):
#   ~/Library/Developer/Xcode/DerivedData/DrewsChessMachine-*/Build/Products/Release/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine
--replay-corpus ./corpus --start-model ./v5-cont-replay-step268506.safetensors \
  --out-model ./v5-cont-replay-latest.safetensors --parameters ./parameters.json \
  --resume-exact --enumerate-checkpoints --epochs 5
```

⚠️ **Re-running that verbatim restarts from step 268506** — it would REDO ~336k
steps (days) AND overwrite the enumerated `step1000..step336000` files.

**Recommended restart (continue from the latest checkpoint):** change only
`--start-model` to the newest checkpoint:

```bash
cd ~/Downloads/v5-continue-bundle
caffeinate -i ~/Library/Developer/Xcode/DerivedData/DrewsChessMachine-cazpchwaqvwjwsaamygqpwdjshfo/Build/Products/Release/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine \
  --replay-corpus ./corpus \
  --start-model ./v5-cont-replay-step336610.safetensors \
  --out-model ./v5-cont-replay-latest.safetensors \
  --parameters ./parameters.json \
  --resume-exact --enumerate-checkpoints --epochs 5 &
# note the new PID; the monitor's ALIVE/DEAD check must use it, not 7022.
```

**⚠️ VERIFY before trusting §4 unchanged — two things the user should confirm:**
1. **`--resume-exact` from a mid-run enumerated checkpoint.** Per the bundle README,
   `--resume-exact` needs the exact corpus stream position and a matching build
   (git `4d60ca2`); the DerivedData binary above satisfies the build. Confirm it
   resumes cleanly from `step336000` rather than erroring. If it errors, fall back
   to plain `--epochs 5` (approximate continuation) from the same `--start-model`.
2. **Step-numbering of the new run** — this decides the monitor offset:
   - If new checkpoints are named `step337000, step338000, …` (numbering CONTINUES):
     keep `OFFSET = 268506`, keep appending to `new_ckpts_run2.jsonl`. §4 works as-is.
   - If numbering RESTARTS at `step1000, step2000, …` (from the 336000 base):
     those new files would collide with existing ones. Historical DATA is already
     safe in `new_ckpts_run2.jsonl` (we don't need the old .safetensors anymore).
     For the new segment use a NEW file `new_ckpts_run3.jsonl` and
     `OFFSET = 604506` (= 268506 + 336000), and teach `build_html.py` to read the
     third file. Do NOT overwrite `new_ckpts_run2.jsonl`.

Do not overthink it: probe the FIRST new checkpoint, look at its filename's step
number, and pick the offset branch above accordingly.

---

## 6. Re-establishing the monitor loop after reboot

The `/loop` Monitor task (id was `b7hbohilt`) is session-bound and will NOT survive
reboot. To resume: start a fresh Claude session in this repo and either re-issue
the `/loop` with a Monitor watching for new `v5-cont-replay-step*.safetensors`
files (or a checkpoint log line), or drive it manually per §4. The standing loop
spec is in §1.

---

## 7. Strength artifact

Published (private) at:
**https://claude.ai/code/artifact/1268c8d5-39e7-45eb-9248-8e783137b3e3**

To refresh after new probes/records: `python3 -I build_html.py` (regenerates
`v5-strength.html` from the jsonl + records.json), then re-publish that file path
via the Artifact tool with the SAME url to keep the link. Favicon ♟️.

---

## 8. Reference facts

- Net: v5, 5-block 7×7 128ch lc0-style, SE + ReZero, WDL 3-logit value head,
  basic30 (30-plane) encoding, bfloat16, ~8.45M params, 4864 policy logits.
  Single forward pass (no MCTS). modelID of current weights: `20260714-1-h7vI`.
- Corpus: lichess_db_standard_rated_2026-05, ~20.9M games, 46 `.dcmgames` shards.
- Probe set `wide` = 4435 fixed puzzles; NLL on it is the honest strength proxy.
- v5 baseline (pre-training reference on the charts): pElo 1584, top1 47.8%, nll 2.097.

### What `--resume-exact` actually restores (verified 2026-07-29 from the header)

A checkpoint is **110 tensors, all F32, 8,447,028 elements = 33,788,112 bytes** —
model weights and nothing else. Verified: **zero optimizer-state tensors.**

RESTORED (from safetensors `__metadata__`, and it demonstrably works):
`replay_next_game_index` (7212496 — matched the abort log exactly), `replay_epoch`,
`replay_corpus_id`, `training_step`. "Exact" refers to the **corpus stream
position** — that is what resumes.

NOT restored, structurally impossible because the data isn't in the file:
1. **Momentum velocity** — log says `velocity zeroed`. With momentum 0.93 the
   steady-state step size is ~14×, so rebuilding from zero does perturb weights.
2. **LR warmup** — restarts at `lr=0.0002`, ramps to 0.01 over 500 steps.
3. **Replay buffer contents** — only capacity + populated count are stored. Log
   confirms a cold refill (`pre-filled: bufCount=1000000 … gamesFed=22650`).

So a restart is NOT seamless even with `--resume-exact`. But do not reflexively
blame a low post-restart probe on it — see §2's note on baseline noise.

### RESOLVED (2026-07-29) — the epoch counter is CORRECT, and run 3 stops ~Aug 2

An earlier note here claimed the `epoch=4` reading contradicted the games-fed
math and might be over-incrementing. **It does not.** The counter is *cumulative
across the resume chain*, not per-run: `CorpusReplayRunner.swift` sets
`var epochsCompleted = startEpoch`, seeded from the checkpoint's `replay_epoch`.
Both figures were right and measure different things — run 2 fed ~2.07 epochs of
its own, on top of the epoch 2 it inherited, giving 4.07 → `epoch=4`.

Verified empirically. `run.out` holds TWO concatenated runs, both numbering from
step 1 (split them where the step number decreases, or the wraps look impossible):

| segment | steps | epoch | wraps at |
|---|---|---|---|
| run 1 | 1…268,506 | 0 → 2 | 62k, 225k |
| run 2 | 1…336,610 | 2 → 4 | 119k, 281k |

Both segments give **~162k steps/epoch** (163k and 162k), which at ~129 games/step
is ~20.9M games — matching the corpus. Note `corpus/corpus.json` canNOT confirm
this: it was never finalized (`gamesAdded: 0`, `state: "recording"`), so the
runner counts shards itself. No shards were skipped in either run (`grep -c
"skipping unreadable shard"` = 0), so no stream drift.

**Consequence — the budget is nearly spent.** `epochLimit = config.epochs` (5) and
`nextGame()` returns nil once `epochsCompleted >= epochLimit`. Run 3 began at
epoch 4, so **it stops at its FIRST epoch wrap**, not after five passes:

- resumed at `nextGame = 7,212,496` of ~20,935,171
- remaining in epoch 4 ≈ 13.72M games ≈ **106k steps**
- at 3.54 s/step ≈ **4.4 days** → self-terminates ~**2026-08-02**, near run-3 step
  106,000, leaving a clean `(nextGame=0, shard=0, epoch=5)` resume point.

To train longer, restart with a higher `--epochs` (6+). `--epochs 5` will NOT give
five more passes.
