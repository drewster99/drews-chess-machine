# Architecture-Experiment Analysis Prompt

Hand this to whoever (or whatever) writes the next entry in `ARCH_EXPERIMENTS.md`. It is the
procedure for turning a finished/plateaued training run into one experiment summary. Follow it
in order; the output is a new `## Experiment N` section appended to `ARCH_EXPERIMENTS.md`,
using that file's exact section structure and conventions.

**Style:** terse, factual, step-anchored. Numbers live in tables. No extra words. Causal claims
must pass the precedence test below. Don't overclaim (see Pitfalls).

---

## 0. Identify and isolate the run
A run is identified by **`arch_hash`** (from the `[APP]` banner) and its **live ModelID lineage**
(e.g. `bzw3`). Always isolate by these — fresh-start and prior-arch logs share low step numbers
and will contaminate early marks.

- `arch_hash`, build, git, `inputPlanes`, `policySize`: `grep -m1 '\[APP\]' <log>`
- Confirm architecture string: `grep -m1 '\[BUILD\]' <log>` (blocks, kernel, SE, params, dtype).
- Filter every later extraction by the lineage tag (e.g. `grep 'bzw3'`) to exclude abandoned runs.

## 1. Data sources
- **Session logs:** `~/Library/Logs/DrewsChessMachine/dcm_log_YYYYMMDD-HHMMSS.txt` (one per launch;
  a run spans many, often resuming from an earlier-step checkpoint — read by **step**, not wall-clock).
- **Saved sessions:** `~/Library/Application Support/DrewsChessMachine/Sessions/<ts>-<ModelID>-<trigger>.dcmsession`.
  Each is a point-in-time snapshot; read its step from `session.json`.
- **Tags:** `[APP]` launch banner · `[STATS]` ~1/min training snapshot · `[ARENA]` arena/promotions ·
  `[TACTICAL-LICHESS]` pElo/NLL probe (200-set and `set=wide`) · `[CHECKPOINT]` · `[ALARM]`.

## 2. Metrics — what to pull and what each means
**Strength (the point of the experiment):**
- **pElo / NLL** (`[TACTICAL-LICHESS]`): policy Elo and move-NLL vs Lichess puzzles. The primary
  *continuous, ungated* strength signal. **Wide-set (4435) is the default** (lower variance);
  200-set is noisier but may have earlier coverage. Lower NLL = better; pElo mirrors it.
- **Arena promotions & cadence** (`[ARENA]`): gated, threshold-based, **lagging** strength signal.
  Useful for generation count and the *cliff*, but it lags the true plateau (this run: ~70k steps).

**Stability (health, NOT strength — a flat run can be perfectly healthy):**
- `pEnt` (uniform ≈ log(4864)=8.49; healthy masked ~2.5–3.6; collapse alarm low),
  `pIllM` (illegal mass, want ≪0.1), `gNorm` (grad L2 vs `grad_clip`),
  `pD`/`vAbs` (value-head collapse = `pD→1`), `pwNorm`/`pLogitAbsMax` (logit inflation —
  benign with flat entropy, only a concern if runaway).

**Optimizer schedule:**
- `lr=` and `μ=` in `[STATS]`. **Detect schedule by distinct values, not min/max range** — a single
  stray reading makes a constant run look cyclic (this bit us). Stamp every change with its step.

## 3. Analysis recipes
Cache the probe trace once, then bucket:
```bash
cd ~/Library/Logs/DrewsChessMachine
grep -h 'TACTICAL-LICHESS' dcm_log_*.txt | grep '<LINEAGE>' > /tmp/probe.txt   # isolate run
```
```python
# pElo/NLL at marks + 20k-bucket curve (run per set: filter 'set=wide' in / out)
import re,statistics as st
rows=[]
for L in open('/tmp/probe.txt'):
    if 'set=wide' not in L: continue            # or 'in L: continue' for the 200-set
    s=re.search(r'step=(\d+)',L); n=re.search(r'NLL=([\d.]+)',L); p=re.search(r'pElo=(-?\d+)',L)
    if s and n and p: rows.append((int(s[1]),float(n[1]),int(p[1])))
rows.sort()
B=20000; bk={}
for s,n,p in rows: bk.setdefault(s//B,[]).append((n,p))
prev=None
for k in sorted(bk):
    pe=st.mean(p for _,p in bk[k]); d='' if prev is None else f'{pe-prev:+.0f}'; prev=pe
    print(f'{k*B:>7} pElo={pe:6.0f} {d:>6} NLL={st.mean(n for n,_ in bk[k]):.3f} n={len(bk[k])}')
```
From the buckets, identify: **steepest-gain window**, **peak**, and **plateau onset** (first bucket
after which Δ stays within the ±noise band; the noise band is the std of the flat tail).

Promotions & cadence:
```bash
grep -hE '#[0-9]+ (kv|prom)' dcm_log_*.txt | grep '<LINEAGE>' \
 | sed -E 's/.*(#[0-9]+).*step=([0-9]+).*score=([0-9.]+).*promoted=([01]).*/\2 \1 score=\3 prom=\4/' \
 | sort -un > /tmp/arenas.txt
grep -c 'prom=1' /tmp/arenas.txt        # total promotions = generations
grep 'prom=1' /tmp/arenas.txt           # promotion steps -> compute cadence per 10k, find the cliff
```
LR/momentum schedule:
```bash
for f in <pre-plateau logs>; do
  echo "$f LR:$(grep '\[STATS\]' $f|grep -oE 'lr=[0-9.e+-]+'|sort -u|tr '\n' ' ') \
        μ:$(grep '\[STATS\]' $f|grep -oE 'μ=[0-9.]+'|sort -u|tr '\n' ' ')"; done
```

## 4. Reconcile the signals (the core insight)
- pElo/NLL plateau = the **true strength flatline**. The arena cliff usually comes **later** —
  the gated arena keeps squeaking marginal promotions on variance after real gains stop. Report
  both, and the lag.
- Cross-check stability metrics at the *final* step: a healthy plateau (flat entropy/value/gNorm)
  is a **capacity ceiling**, not a blow-up.

## 5. Causal-reasoning rules
- **Temporal precedence:** a setting change at step X cannot cause an effect that began before X.
  Always place each change on the step axis before blaming it.
- **Optimization vs capacity:** LR/momentum/schedule are optimization knobs — they change *how
  fast/cleanly* you reach the architecture's frontier, not *what* it can represent. A plateau under
  constant, healthy settings, where stability metrics are fine, is capacity-limited.
- **Warm-restarts-failed = capacity signature:** LR cycling (SGDR-style) is *the* standard
  plateau-escape technique. If it earned nothing, that's strong evidence the limit is capacity, not
  optimization. Treat any high-LR cycling episode as a (confounded) high-LR control.
- The one untested static lever with real upside is a **monotonic anneal down** (e.g. 1e-2→1e-3):
  lowers the SGD noise floor, may polish a few pElo toward the ceiling — never raises it.

## 6. Pitfalls (we hit these — don't repeat)
- **Don't infer cycling from LR min/max range — count distinct values.** One stray reading faked a
  "1e-1 cycle" that never existed.
- **Isolate by arch_hash/lineage.** Abandoned fresh-start logs (steps near 0) collide with early marks.
- **Don't overclaim "trained cleanly throughout."** Scope Wins to *no instability* and to the
  *productive window*; post-saturation regression (sub-0.5 candidates, logit inflation) goes in
  Shortcomings. "No collapse" ≠ "productive the whole run."
- **Wide-set is the cross-experiment default.** If only the 200-set covers early steps, show both and
  mark the exception (offset ~90–100 pElo).
- **Saved sessions are point snapshots, not ranges** — read the step from `session.json`.
- **Read the run by step, not time** — resumes restart wall-clock and can rewind the step counter.

## 7. Output — append `## Experiment N` to `ARCH_EXPERIMENTS.md`
Match Experiment 1 exactly. **Title, `arch_hash`, lineage (saved/live), and date range go in the
unnumbered `## Experiment N — …` header + metadata line** (not a numbered section). Then seven
numbered sections:
1. **Architecture** — input/policy/value, stem, tower (blocks, kernel, SE, skip), heads, dtype, params, arch version; one context line on the param-budget tradeoff.
2. **Relevant saved sessions** — table: session folder · step @ snapshot.
3. **Factuals** — table `Step | pElo (wide) | NLL (wide) | pElo (200) | NLL (200) | Detail`,
   ordered by step. Rows = start, first promotion, steepest-gain mark, probe-instrumentation step,
   **plateau onset**, turning point(s)/cliff, each param change (with step), last promotion, final
   assessment. `—` where a set lacks coverage.
4. **Wins** — only what's true (no-instability; productive window with numbers).
5. **Shortcomings** — plateau step, unproductive span, any regression, capacity verdict.
6. **Analysis** — capacity-vs-optimization call, cadence-as-strength-curve, schedule findings, concerns.
7. **Suggested future variants / changes** — the next lever (usually depth), regularization, LR plan.

## 8. Validation checklist (before finalizing)
- [ ] Run isolated by arch_hash + lineage; no foreign-run rows.
- [ ] pElo/NLL are wide-set where available; exceptions marked.
- [ ] Plateau onset (pElo) and arena cliff both reported, with the lag.
- [ ] Every LR/momentum change stamped with its step; schedule confirmed by distinct values.
- [ ] Every causal claim passes temporal precedence.
- [ ] Wins contain no degradation; degradation is in Shortcomings.
- [ ] Final-step stability metrics cited to justify capacity-vs-blow-up call.
- [ ] Numbers in tables; prose trimmed of filler.
