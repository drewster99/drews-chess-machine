# Architecture-Experiment Analysis Prompt

Hand this to whoever (or whatever) writes the next entry in `ARCH_EXPERIMENTS.md`. It is the
procedure for turning a finished/plateaued **or still-running** training run into one experiment
summary. Follow it in order; the output is a new `## Experiment N` section appended to
`ARCH_EXPERIMENTS.md`, using that file's exact section structure and conventions.
(In-progress runs are valid entries — mark the header "in progress", give **no capacity verdict**,
and classify everything **relative to the comparison baseline**, normally the prior experiment.)

**Style:** terse, factual, step-anchored. **Short words. Short sentences. Short paragraphs.** Numbers
live in tables. Causal claims must pass the precedence test below. **Map hypotheses to the variables
you changed; give competing hypotheses, not one confident mechanism — don't theorize past the
evidence. A Win must beat the comparator; matching it is not a Win.** Don't overclaim (see Pitfalls).

---

## 0. Identify and isolate the run
**Do NOT identify or equate runs by `arch_hash`.** Under runtime-configurable architecture it is not
authoritative — it can be *identical across genuinely different encodings* (Exp 1 and Exp 2 both log
`0xdf23a86c`), and the `[APP]` banner's `inputPlanes`/`policySize` can be stale (Exp 2 ran 200-plane
`full10ply200` while the banner still said `inputPlanes=30`). Identify a run by its **log file**,
**build number**, and **live ModelID lineage** (e.g. `2Gd1`). The **saved** lineage differs from the
**live** one (e.g. `oItC` saved / `2Gd1` live) — sessions on disk carry the saved tag.

- **TRUE architecture** from the **`[ARCH]`** line (`architectureSummary`: encoding+planes, stem,
  blocks, kernel, SE, value head, dtype, params): `grep -m1 '\[ARCH\]' <log>`. Builds predating `[ARCH]`
  carry the same string elsewhere, or read it from a saved session's `session.json`
  (`architecture.inputPlanes` / `input_encoding`) — never trust the `[APP]` banner's arch fields.
- build, git: `grep -m1 '\[APP\]' <log>` (use these fields, not its arch fields).
- Filter every later extraction by the lineage tag (e.g. `grep '2Gd1'`) to exclude abandoned/foreign runs.

Don't guess. Check.

## 1. Data sources
- **Session logs:** `~/Library/Logs/DrewsChessMachine/dcm_log_YYYYMMDD-HHMMSS.txt` (one per launch;
  a run spans many, often resuming from an earlier-step checkpoint — read by **step**, not wall-clock).
- **Saved sessions:** `~/Library/Application Support/DrewsChessMachine/Sessions/<ts>-<ModelID>-<trigger>.dcmsession`.
  Each is a point-in-time snapshot; read its step from `session.json`.
- **Tags:** `[APP]` launch banner · `[STATS]` ~1/min training snapshot · `[ARENA]` arena/promotions ·
  `[TACTICAL-LICHESS]` pElo/NLL probe (200-set and `set=wide`) · `[CHECKPOINT]` · `[ALARM]`.

Actually look for sources and make sure they're the correct ones.

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

Be sure you're fetching the right data. Never guess. Always check.

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
- **Never isolate or equate runs by `arch_hash`** — under runtime-config it collides across different
  encodings, and the banner's `inputPlanes` can be stale. Isolate by log file + build + live lineage;
  confirm the encoding from `[ARCH]` or `session.json`. Abandoned fresh-start logs (steps near 0) also
  collide with early marks — filter by lineage.
- **A Win must beat the comparator.** Matching it (both stable, both fast) is not a Win — just no
  regression. Stable training is a Win only if the comparator was unstable. Worse than it (slower
  bootstrap, lower matched-step pElo) is a Shortcoming.
- **Don't over-theorize cause.** Map candidate causes to the variables you actually changed and list
  them as competing hypotheses; don't commit to one mechanism (or an elaborate just-so story) without
  evidence. "One, the other, or both — undetermined" is an acceptable, honest conclusion.
- **Don't overclaim "trained cleanly throughout."** Scope Wins to *no instability* and to the
  *productive window*; post-saturation regression (sub-0.5 candidates, logit inflation) goes in
  Shortcomings. "No collapse" ≠ "productive the whole run."
- **Wide-set is the cross-experiment default.** If only the 200-set covers early steps, show both and
  mark the exception (offset ~90–100 pElo).
- **Saved sessions are point snapshots, not ranges** — read the step from `session.json`.
- **Read the run by step, not time** — resumes restart wall-clock and can rewind the step counter.

## 7. Output — append `## Experiment N` to `ARCH_EXPERIMENTS.md`
Match the format of Experiment 1 exactly. **Title, `arch_hash`, lineage (saved/live), and date range go in the
unnumbered `## Experiment N — …` header + metadata line** (not a numbered section). Then seven
numbered sections:
1. **Architecture** — input/policy/value, stem, tower (blocks, kernel, SE, skip), heads, dtype, params, arch version; one **Context** line stating — for a controlled experiment — **exactly what is held fixed vs changed relative to the comparison baseline** (plus any param-budget tradeoff).
2. **Relevant saved sessions** — table: session folder · step @ snapshot. Saved lineage ≠ live lineage — list **every** snapshot for the run (find them by date/window + saved tag, not the live tag).
3. **Factuals** — table `Step | pElo (wide) | NLL (wide) | pElo (200) | NLL (200) | Detail`,
   ordered by step. Rows = start, first promotion, steepest-gain mark, probe-instrumentation step,
   **plateau onset**, turning point(s)/cliff, each param change (with step), last promotion, final
   assessment. `—` where a set lacks coverage.
4. **Wins** — only what **beats** the comparator. A trait it lacked, or a metric better than it at
   matched steps. Matching it is **not** a Win — that's just no regression; drop it or note it neutrally.
   Stable training is a Win **only if** the comparator was unstable and this run isn't.
5. **Shortcomings** — name the comparator ("Comparing primarily against Experiment N"). Then: anything
   **worse than it** at matched steps (with numbers — slower bootstrap, lower pElo, delayed value
   differentiation), or **objectively bad** on its own (instability, regression, plateau + unproductive
   span if finished). Capacity verdict only if finished; in-progress → "no capacity verdict yet".
6. **Analysis** — for a **controlled experiment**, enumerate candidate causes as **competing hypotheses
   that map 1:1 to the variables you changed** (e.g. "H1: +9 history planes slowed it · H2: −10
   repetition planes slowed it"), and state it may be one, the other, or both — **no winner without
   evidence, no elaborate mechanism story.** Then capacity-vs-optimization (if finished),
   cadence-as-strength-curve, schedule findings, the n=1 seed caveat, concerns.
7. **Suggested future variants / changes** — lead with the **single most concrete next lever, tied to a
   hypothesis** (e.g. "add back the 10 repetition planes"); one clear move beats a menu of five. At most
   a couple of secondary notes.

## 8. Validation checklist (before finalizing)
- [ ] Run isolated by log file + build + live lineage (NOT arch_hash); encoding confirmed via `[ARCH]`/`session.json`; no foreign-run rows.
- [ ] pElo/NLL are wide-set where available; exceptions marked.
- [ ] Plateau onset (pElo) and arena cliff both reported, with the lag.
- [ ] Every LR/momentum change stamped with its step; schedule confirmed by distinct values.
- [ ] Every causal claim passes temporal precedence.
- [ ] Wins contain nothing worse-than-baseline; relative regressions (slower/weaker at matched steps) are in Shortcomings, with the baseline named.
- [ ] Analysis hypotheses map 1:1 to the variables changed; no unsupported single-mechanism claim; seed/n=1 caveat stated.
- [ ] §7 leads with one concrete lever tied to a hypothesis.
- [ ] Final-step stability metrics cited to justify capacity-vs-blow-up call.
- [ ] Numbers in tables; prose trimmed of filler.
