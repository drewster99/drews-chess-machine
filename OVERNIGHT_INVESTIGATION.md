# Overnight investigation (2026-06-09, autonomous loop) — running notes

Uncommitted working notes. Two tracks. Raw Track-1 data: `/tmp/arch_bench.jsonl`.

## TRACK 1 — "training hangs at large block counts" — SOLVED

Tool: headless `--arch-sweep` CLI (`App/ArchSweepCLI.swift` + handler in `DrewsChessMachineApp.swift`). Builds a trainer per block count in the regime that hung (basic30, 32ch, 3×3 blocks, SE+/4, WDL 32→FC128, bf16, batch 512), times build + 8 steps, streams JSONL.

### Data (gpuRunMs)

| blocks | params | autodiff build | **step 1 (first encode)** | **steady (step 2-8)** |
|---|---|---|---|---|
| 20 | 0.67M | 0.5s | 6.9s | **55 ms** |
| 50 | 1.25M | 2.6s | 42.3s | **128 ms** |
| 80 | 1.84M | 5.8s | 112s | **231 ms** |
| 110 | 2.42M | 9.1s | 252s | **330 ms** |
| 140 | 3.01M | 14.9s | **488s (8.1 min)** | **422 ms** |
| 170 | 3.60M | 21.8s | (running) | — |

### Findings

1. **Steady-state is fast and scales ~LINEARLY (~3 ms/block) — it runs on the GPU.** 140 blocks = 422 ms/step is fully trainable. CPU execution of that net would be many seconds/step. **The user's "graph moved to CPU" hypothesis is refuted for execution** — there is no whole-graph CPU fallback in steady state. (Earlier process sample also showed `GPURegionRuntime` + AGX pipeline compilation = GPU-targeted.)
2. **The "hang" is the one-time first-step `encode`** = Metal compiling a GPU pipeline state (PSO) for every distinct op, serialized through the Metal compiler. **~Quadratic in depth** (fits blocks²: 6.9s@20 → predicted 338s@140, actual 488s, slightly super-quadratic). One-time; cached after step 1.
3. **Autodiff graph *build* (trainer init) also scales superlinearly** (~quadratic): 0.5s@20 → 22s@170. Separate, smaller, happens before step 1.
4. **The "9 steps then stopped" mystery — SECOND compile wall.** The trainer caches executables by `(batchSize, includeDiagnostics)` and `includeDiagnostics = nextStep % batchStatsInterval == 0` with `batchStatsInterval = 10`. So **step 10 (the first stats step) compiles a SECOND, larger executable** (`leanTargets + diagnosticTargets`, ~20+ extra ops) → a second multi-minute PSO-compile wall, right at "9 steps in". This is almost certainly what the user saw.

### Verdict & implications
- Deep towers are NOT broken and NOT CPU-bound at runtime. Doubling/tripling blocks is reasonable for *throughput* (140 blocks ≈ 2.4 steps/s). The ONLY barrier is the one-time PSO-compile latency (and it hits twice: lean + diagnostics), which grows ~quadratically with depth.
- Mitigation options (future): (a) **`MTLBinaryArchive`** to persist compiled PSOs across launches (skip recompilation); (b) **eagerly compile the diagnostics executable up front** alongside the lean one so both walls happen during startup, not as a surprise at step 10; (c) a progress indicator during first-encode so it doesn't read as a hang; (d) `MPSGraphCompilationDescriptor` tuning.

### Still to verify (next iterations)
- Measure the SECOND (diagnostics) wall's magnitude directly at deep counts.
- 170-block step1 number (running).
- Bulletproofing: a forced MPSGraph CPU-device comparison to triple-confirm steady is GPU (optional; performance already conclusive).

## TRACK 2 — Experiment 4 blowup — IN PROGRESS (analysis done, experiments pending)

### Onset signature (wide-set tactical probe, `dcm_log_20260608-140857.txt`)
- NLL stable ~3.78–3.88 through step ~5014 (trainer = jaq1-3).
- **Trainer forks to jaq1-4 at step ~5014 (just after arena #10 promotion, step 4988). NLL begins a gradual, accelerating climb from exactly there:** 3.78(5025) → 4.0(5126) → 4.2(5401) → 4.5(5500) → 5.3(5600) → … → 8–17 later.
- Simultaneously: argmax top1 slips 8.8%→7.5%, avgRank worsens 15.0→17.9, and specific themes collapse (promotion 90→32, endgame 55→35). Hanging-piece holds.
- In-distribution training metrics (pLogitAbsMax, pwNorm, gNorm, pEnt) stayed FLAT throughout → the blowup is OOD-only.

### Hypotheses
- **H1 (distribution narrowing / over-sharpening on self-play) — leading.** Onset coincides with a promotion (champion changes → self-play distribution shifts/narrows as the champion gets more decisive). The trainer over-specializes to the narrowing distribution; OOD tactical NLL explodes while in-dist stays clean. Predicts: arch-independent in principle, but a wider net (256ch, Exp 4) overfits the narrow distribution faster/harder than 128ch (Exp 3, which never blew up in the same window) — explains why Exp 4 and not the others, without "LR too high."
- **H2 (bf16 + 256ch numerical over-sharpening) — weakened.** Flat pLogitAbsMax/pwNorm/gNorm argue against an unbounded logit/weight blowup.
- **H3 (decisive-game target shift + lr) — related to H1.**

### ⚠️ ITERATION 3 — CAPACITY HYPOTHESIS REFUTED by the live 50-block run

The user's **live 50-block run is NARROW** — `basic30, stem 32 (3×3), 50×[3×3], 32ch, value WDL(16→FC32), 1.02M params` — and **its wide-set NLL is already blown up (8–16), top-1 oscillating 3.6–9.9%** (steps 9k–15k). A **1.02M-param** net blew up while the **9.57M Exp 3** and **8.45M Exp 1** did not. So "surplus capacity / width → over-fit" is **wrong** — params/width do not predict the blowup.

Re-tabulated (the real split):

| run | enc | blocks | ch | kernels | stem | params | blowup? |
|---|---|--:|--:|---|---|--:|---|
| Exp1 | basic30 | 5 | 128 | 7×7+7×7 | 7×7 | 8.45M | NO (→470k) |
| Exp3 | 210 | 5 | 128 | 7×7+7×7 | 7×7 | 9.57M | NO (→48k) |
| Exp4 | 210 | 5 | 256 | 7×7+**3×3** | **3×3** | 20.3M | **YES ~5k** |
| 50blk | basic30 | 50 | 32 | **3×3**+3×3 | **3×3** | 1.02M | **YES ~9k** |

**Cleanest correlation now: kernel/stem.** The two STABLE runs are pure **7×7** (tower + stem); both BLOWUPS contain **3×3** kernels and a **3×3 stem**. Capacity, width, depth, params, and encoding all FAIL to separate the groups; the 7×7-vs-3×3 axis is the only one that does. **This is a correlation across 4 confounded runs, NOT established cause** — 3×3 kernels causing OOD NLL blowup is mechanistically odd, so treat the kernel lead as a hypothesis to test, and suspect a hidden common factor. Either way: **the blowup is common and reproducible, not an Exp-4 quirk.** "LR too high" remains refuted (Exp 1/3 fine at lr 0.01).

NEXT: (a) SAFE — expand this table from ALL available run logs (more 3×3 and 7×7 points) to test/break the kernel correlation; look for a 3×3 net that stayed stable or a 7×7 that blew up. (b) controlled experiment varying ONLY kernel size (needs a code enabler — `--architecture-file` is a stub, not wired; or `--resume`; or an in-process harness — all code changes, best run attended with GPU headroom).

### ITERATION 4 — ruled out code-regression; Exp3 durably stable; lead narrowed to 3×3 second-conv / stem

- **Code-regression hypothesis REFUTED.** The only commits between the Exp 3 build (06-07 17:49) and the Exp 4 build (06-08 14:06) are `6390bd1` (weight-analyzer labels), `1bc57fa` (channel names), `08a74a5` (status-bar/chart UI) — **none touch the training loop, loss, sampling, or network.** So the Exp3→Exp4 difference is architecture, not a code change. (My own overnight commits all postdate Exp 4.)
- **Exp 3 is DURABLY stable, not just slow.** Its wide-set NLL is flat **~3.75 across all 53,826 steps** (range ~3.4–4.1, one transient outlier). So "blows up vs doesn't" is **qualitative**, not a rate difference.
- **Feature-separation analysis (the 4 long runs).** A causal feature must be equal within stables AND within blowups, differing between. Result — only two features qualify:
  - **k2 (2nd block conv): 7×7 in both stables, 3×3 in both blowups.** ✅
  - **stem kernel: 7×7 in both stables, 3×3 in both blowups.** ✅
  - Everything else FAILS: enc (mixed), blocks (5/5 vs 5/50), channels (128/128 vs 256/32), **k1 (7/7 vs 7/3 — Exp 4 has a 7×7 first conv and still blew up!)**, SE reduction (4/4 vs 2/4), params (8–9M vs 20M/1M), value FC. So **k1, width, depth, params, capacity, SE, encoding, and LR are all ELIMINATED.**
- **Surviving lead:** the **3×3 second-conv and/or 3×3 stem**. Confounded (the two co-vary; only 2 long runs/side) and mechanistically odd (a 3×3 stem causing OOD blowup is implausible — Exp 1 forensics showed even a 7×7 stem collapses to ~1×1), so SUSPECT a hidden factor that co-varies with "3×3-ness." Cannot be isolated from logs alone.

### ITERATION 5 (2026-06-10) — executable/.level1 kernel-numerics hypothesis TESTED and REFUTED at unit level

The timeline lead from iteration 4+ ("every 3×3 blowup ran on the compiled-executable training step (a1f1e7e + .level1, 06-02); the stable 12-block 3×3 ran pre-change on graph.run; stable new-code runs are pure 7×7; 3×3 = Winograd-eligible in bf16") was tested directly by `ConvKernelExecutionPathNumericsTests` (new, in-repo; report at `/tmp/conv_kernel_path_numerics_report.txt`):

- conv(k)→relu→conv(k) towers, C=128, 8×8, fp32 masters + in-graph bf16 cast — the trainer's exact weight regime.
- **Single step (batch 512): gradients are BIT-IDENTICAL across graph.run, executable(.level0), executable(.level1)** — both kernels, both dtypes (`differing=0/802816`). bf16-vs-fp32 deltas are symmetric quantization noise, bias ≈ 1e-7 relative.
- **100-step SGD trajectories (batch 128): run-vs-level1 masters bit-identical at every checkpoint**, both kernels. bf16 drift vs fp32 stays ~1e-5 absolute vs 1e-2 weight RMS, no bias growth.

So MPSGraph selects the same (or numerically identical) conv kernels on all three execution paths at these shapes; the 06-02 trainer change is strongly disfavored as the blowup cause. Caveat: live batch is 4096 — kernel selection is size-dependent, and bit-identity at 128 and 512 doesn't formally cover 4096 — but two sizes × two opt levels all matching makes a 4096-only divergence unlikely. Suite green (685/686, 1 pre-existing skip).

**Extension (same day, per user): batch 4096 + ULP-adversarial regimes — STILL BIT-IDENTICAL.** Two more tests (`testGradientNumericsAtProductionBatchSize`, `testUlpBoundaryAdversarialGradients`):
- **Batch 4096** (the production size): head-to-head bf16 graph.run vs executable(.level1) gradients `differing=0` for both kernels. The size caveat is CLOSED.
- **Six adversarial regimes** (batch 512, bf16, both kernels, run vs .level0 vs .level1): inputs+weights a hair below / exactly at / a hair above the bf16 half-ULP rounding boundary (RNE/ties stress incl. any .level1 cast→conv fusion); mixed-exponent catastrophic-cancellation (accumulation-order probe, 13 binades); extreme-exponent products (1e18-scale weights × 1e-18-scale inputs); near-subnormal gradients (products at the fp32/bf16 min-normal edge). **Every regime: `differing=0`, identical losses, identical non-finite counts.** Notably the near-subnormal regime flushed conv2's weight-gradient to exactly zero in ALL paths identically (refRMS=0) — even flush-to-zero policy is uniform.

The executable/.level1/kernel-selection hypothesis is now refuted comprehensively: on this hardware/OS, MPSGraph's conv forward+backward is bit-identical across graph.run, compiled .level0, and compiled .level1, at production batch size, under rounding-tie, cancellation, extreme-exponent, and subnormal stress. Suite green (687/688).

Standing facts after iteration 5: the 3×3-vs-7×7 separation remains real and UNEXPLAINED by any audited code path (builder faithful except the ReZero-α Build-screen regression, which only mis-set the 50-block run; tick-driver concurrency race-free; vBaseline handoff dtype-clean; batch/LR identical across stable+blowup runs). Most decisive remaining experiment: train the `v4_12block_3x3` preset on the current build with standard params — stable ⇒ blowup is architecture-intrinsic (depth/kernel dynamics); blows up ⇒ a new-code factor still hides outside everything audited so far. The 7×7 5-block control run started 2026-06-10 09:09 (JhJQ) is the other arm.

### ITERATION 6 (2026-06-10) — SUPER-TABLE: architecture × parameters × outcome, all eight runs

Motivation: the stable/blowup comparisons so far tabulated architecture only; nobody had tabulated the sampling/loss/training parameters per run. Extracted from each run's own `[STATS]`/`[ARCH]`/banner lines (one log per run, identified by lineage ID + build).

**Parameter axis: ELIMINATED.** Every run below — stable and blowup, 2026-05-31 through today — ran with byte-identical knobs:
`sp.tau=1.00/0.50/0.007 · ar.tau=0.60/0.20/0.020 · drawKeep=1.00 · batch=4096 · lr=1.0e-02·√b (warmup 1/500 on fresh builds) · promote≥0.53 · arenaGames=400 · arenaAutoSec=900 · workers=800 · buffer=1M · clip=30 · wd=1e-04 · ent=0 · illM=1.0 · drawPen=0 · pLossW=vLossW=1.00 · μ=0.90 · complCE=on`
(Note: `sp.tau` floor has been **0.007** — near-greedy late-game — in every run including the stables; the 1.0→0.4 schedule in the design docs is stale.)

| run (lineage) | dates | build(s) | builder / step-exec era | encoding | stem | tower | SE | value | params | ReZero α (should be) | steps reached | wide-NLL | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LWKa | 05-31→06-01 | ≤1562 | old fixed · graph.run | basic30 | 128 **(3×3)** | 12×[3×3] 128ch | /4 | WDL 16→FC128 | ~3.9M | 0.289 (✓ 1/√12) | ~106,740 | n/a (pre-wide); 200-set flat 3.57–3.69 | **STABLE** |
| bzw3 = Exp 1 | 06-01→06-06 | 1612→1716 | old fixed → config-on-load (1693+) · spans the 06-02 executable change | basic30 | 128 (7×7) | 5×[7×7] 128ch | /4 | WDL 16→FC128 | 8.45M | 0.447 (✓ 1/√5) | **470,819** | flat 3.16–3.18 from 240k→470k | **STABLE** |
| 2Gd1 = Exp 2 | 06-06 | 1760 | config · executable | full10ply200 | 128 (7×7) | 5×[7×7] 128ch | /4 | WDL 16→FC128 | ~8.9M | 0.447 (✓) | 58,933 | flat 3.81–3.82 | **STABLE** |
| eaRt = Exp 3 | 06-07 | 1770 | config · executable | full10Ply10Reps210 | 128 (7×7) | 5×[7×7] 128ch | /4 | WDL 16→FC128 | 9.57M | 0.447 (✓) | 53,837 | flat ~3.75 | **STABLE** |
| jaq1 = Exp 4 | 06-08 | 1781/1782 | config · executable | full10Ply10Reps210 | 256 **(3×3)** | 5×[7×7,3×3] 256ch | **/2** | WDL 16→**FC256** | 20.3M | 0.447 (✓) | 13,868 | blowup onset **~5,014** (at promotion #10) | **BLOWUP** |
| LMGh = 50blk | 06-09 | 1793 | config · executable | basic30 | 32 **(3×3)** | 50×[3×3] 32ch | /4 | WDL 16→FC32 | 1.02M | **0.289 (✗ should be 0.141** — Build-screen α regression) | 87,507 | 7.8→11.1 (blown from ~9k) | **BLOWUP** |
| WjRY/tGOH = 8blk | 06-09→06-10 | 1795 | config · executable | basic30 | 128 **(3×3)** | 8×[3×3] 128ch | /4 | WDL 16→FC128 | 2.66M | 0.354 (✓ 1/√8) | ~107,548 | flat 3.58 → break 72–76k → 9–14 | **BLOWUP** (onset right after last promotion #26 @62,415) |
| JhJQ | 06-10→ | 1795/1801 | config · executable | basic30 | 128 (7×7) | 5×[7×7] 128ch | /4 | WDL 16→FC128 | 8.45M | 0.447 (✓) | running | — | control arm (Exp-1 arch on current code) |

Designed total residual gain `blocks·α²` = 1.0 for every run except 50blk (4.17, the α bug).

### What the super-table settles
1. **Parameters are NOT the axis** — identical across all eight runs. Hypothesis "recent default/param drift" (incl. the 06-04 defaults-alignment commit) is refuted; the 0.007 tau floor, 82%-draw data regime, lr 0.01 (·√(b/4096), a no-op at batch 4096 — see iteration 7 correction), and complement-CE were all present in the 470k-step stable run too.
2. **The clean separation is now: current-era (config-builder/executable) runs with a 3×3 STEM all blew up (3/3); all 7×7-stem runs are stable (3/3 + control pending); the old-era 3×3 stem (LWKa) was stable to ~106k** — beyond WjRY's 73k onset, though onset varies wildly with arch (5k–73k) so step counts aren't directly comparable.
3. bzw3's final ~90k steps ran on the CONFIG builder (built-by-embedded-config on load, build 1693+) and stayed flat — the config builder per se is exonerated for 7×7. Combined with the bit-exact kernel tests, "new code × 3×3" survives only as a correlation whose code-level mechanism has been eliminated everywhere we've looked.
4. Checkpoint phantom-ID naming (saves stamped with an ID absent from [STATS]/[ARENA]) is visible as far back as Exp 4 (cwkO save vs jaq1 lineage), same pattern as tGOH vs WjRY — one bug, ongoing.
5. The decisive experiment is unchanged and sharper: **`v4_12block_3x3` preset on the current build** replicates LWKa's exact architecture on new code. Stable past ~110k ⇒ blowups are arch-intrinsic to the *specific* 3×3 configs tried (stem width/depth combos), not the code. Blows up ⇒ a new-code factor exists that every audit so far has missed.

### ITERATION 7 (2026-06-10) — pre-bf16 era recovered: the 8×3×3 ran 495k steps STABLE… at 10× lower LR

User recalled an early 8-block 3×3 multi-day run pre-bf16. Found and verified:

| run (lineage) | dates | arch_hash | arch | precision | lr | μ | ent / drawPen / complCE | steps | trainer gens (≈promotions) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **KbHZ** | 05-14→05-22 | 0x13ba0b55 | **8×[3×3] 128ch, stem 3×3**, basic30, pre-v4 tower | **fp32** | **1.0e-03·√b** | 0.65 | 2.5e-3 / 0.05 / absent | **~495,000** | -3→-18 in the logs (≈15+; steady) | **STABLE** — pLoss 0.77→0.55, pIllM 0.037→0.011, promoting throughout |
| sMe9 | 05-25→05-28 | 0x13ba0b55 | same 8×3×3 fp32 | fp32 | 1.0e-03·√b | 0.85→0.90 | 0 / 0 / **on** (complCE existed by 05-25) | ≥115,628 | -3→-29 (≈26!) | healthy in logs |
| ysdg | 05-29→05-30 | 0x5347c53d | 16-block era (053f919) | fp32 | 1.0e-03·√b | 0.90 | 0 / 0 / on | — | — | short-lived era |
| KXvb→LWKa | 05-31 | 0xbad32ced | v4 12×3×3 (cb3b4cf + c249df2) | **bf16**+fp32 masters | **1.0e-02·√b ← 10× JUMP** | 0.90 | 0 / 0 / on | →106k | 9+ | stable to retirement |

**Corrections to iteration 6:** "parameters identical across all runs" holds only for the post-05-31 era. Before that: lr was **10× lower** (1e-3 vs 1e-2; the √-batch factor is sqrt(b/4096) per ChessTrainer.sqrtScaleBaseBatchSize — a NO-OP at batch 4096, so effective LRs are simply 0.001 vs 0.01. An earlier draft of this section mis-read the display as raw √b; corrected 06-11), momentum 0.65→0.85→0.90 over May, entropy bonus and drawPenalty were on during KbHZ, complement-CE arrived ~05-25. **On 05-31, four things changed at once:** bf16 default + fp32 masters + architecture v4 (pre-activation/ReZero/scale-bias-SE) + lr ×10. Every blowup is on the far side of that day; every 3×3 run on the near side (KbHZ 495k, sMe9 115k) was healthy.

Caveats: KbHZ/sMe9 predate the tactical/Lichess probes entirely (zero `[TACTICAL]` lines) — OOD health unmeasured; their stability evidence is in-dist metrics plus **sustained promotion velocity** (the strongest behavioral signal: KbHZ minted ~15 trainer generations across the week, sMe9 ~26 — vs WjRY's lock-out after 7).

**Sharpened hypothesis:** `lr=1e-2` (·√(b/4096) is a no-op at batch 4096) with μ=0.90 — effective per-gradient step ≈ lr/(1−μ) ≈ 0.10, vs KbHZ's 0.001/(1−0.65) ≈ 0.003, a ~35× aggression gap — sits **above the optimization-stability threshold for shallow/narrow 3×3 towers but below it for 7×7 towers** (and was tolerated by the deeper 12×3×3 with its smaller per-block ReZero gain). bf16 may shave the remaining margin. This fits: the same 8×3×3 shape = 495k steps at lr 1e-3 (fp32 era) vs blowup at 73k at lr 1e-2 (bf16/v4 era); 7×7 stable at lr 1e-2 to 471k; onset-at-promotion (distribution shift perturbs an optimizer already at the stability edge); the minute-scale function-space oscillation in probes.

**Decisive experiments, now even cheaper:** (a) rerun WjRY's exact 8×3×3 v4 bf16 config at **lr 1e-3·√b** — sails past 100k ⇒ LR is the trigger; (b) the 12-block-on-current-code arm still separates code-era for free; (c) optionally KbHZ-exact (fp32) at lr 1e-2 to test precision-independence.

### ITERATION 8 (2026-06-11) — retro-probe of all checkpoints (`--probe-model` CLI) + FINAL RUN TABLE

New enabler: `--probe-model <path> [--probe-set 200|wide|both] [--probe-out f]` (ProbeModelCLI.swift) — loads any saved champion (weight file / .dcmsession / directory of sessions) by its embedded/legacy arch and runs the exact `TacticalProbeRunner.runBatch` path the live watchers use. Swept ALL 80 session checkpoints (incl. legacy v3 era), 0 errors. Raw: `/tmp/checkpoint_probe_sweep.jsonl`.

**Cross-model metric note:** NLL is sharpness-confounded (KbHZ at pEnt≈1.7 reads NLL 6.58 with a *better* pElo than diffuse June models at NLL 3.7). `top1` and `pElo` are the comparable columns; NLL is only comparable within a lineage at similar entropy.

#### FINAL RUN TABLE (all lineages, 2026-05-14 → live)

| run | arch | prec | lr / μ / complCE | steps reached | wide-set top1 / pElo trajectory (probed ckpts) | verdict |
|---|---|---|---|---|---|---|
| **KbHZ** | 8×3×3 v3, stem 3×3 | fp32 | 1e-3 / 0.65 / – | 494,927 (May) | 6.7% / 562 @495k | **STABLE** |
| **KbHZ-cont.** (live) | same, on current code | (loaded arch) | 1e-3 / 0.65 / on→OFF @498.7k | →536k+ | 6.7→**7.4** (post-complCE kick) →7.0% / 562→585→571; NLL dipped 6.58→4.81 (diffusion) → re-sharpened to 6.27 | **STABLE**, promoting (gen 17→22) |
| **sMe9** | 8×3×3 v3 | fp32 | 1e-3 / 0.85→0.90 / on | ≥372,748 | **11.1% / 709** @373k — best 3×3 ever recorded | **STABLE** |
| **ysdg** | 16×3×3 v3 | fp32 | 1e-3 / 0.90 / on | ~78,222 | 5.5% / 512 @~75k (slow learner) | stable-while-run |
| **KXvb** | 12×3×3 **v4 bf16** | bf16 | **1e-2** / 0.90 / on | ~51,551 | 6.5% / 552 @~51k | stable-while-run |
| **LWKa** | 12×3×3 v4 bf16 | bf16 | 1e-2 / 0.90 / on | ~106,740 | 9.2% / 648 @~100k | **STABLE→retired** |
| **bzw3 / Exp1** | 5×7×7 v4 | bf16 | 1e-2 / 0.90 / on | **470,819** | 16.6→**17.3% / 876** @251k→471k — strongest model overall | **STABLE** |
| **2Gd1 / Exp2** | 5×7×7, full10ply200 | bf16 | 1e-2 / 0.90 / on | 58,933 | 6.6→10.5→9.5% / 555→688→660 (gens 1-11) | **STABLE** |
| **eaRt / Exp3** | 5×7×7, 210enc | bf16 | 1e-2 / 0.90 / on | 53,837 | 7.1→10.3% / 577→682 monotone-ish (gens 1-14) | **STABLE** |
| **jaq1 / Exp4** | 5 blk 256ch, stem 3×3, k2 3×3 | bf16 | 1e-2 / 0.90 / on | 13,868 | gens 1-3: 6.5→8.7% / →632 healthy; **gen 4 (the onset promotion): NLL 3.77→6.40, top1 8.7→6.3%** — the damage was already in the promoted candidate | **BLOWUP @~5k** |
| **LMGh / 50blk** | 50×3×3 32ch, **α=1/√12 bug** | bf16 | 1e-2 / 0.90 / on | 87,507 | gen1 4.7%/3.98 → **gen2 NLL 13.1** → gen3 8.5%/5.50 — violent oscillation, champions promoted while OOD-blown | **BLOWUP @~9k** |
| **WjRY** | 8×3×3 v4, stem 3×3 | bf16 | 1e-2 / 0.90 / on | ~107,548 | gens 1-7 (7.7k→62.4k): 5.2→**8.2% / 614**, NLL 3.73→3.60 — every saved champion healthy & improving; collapse happened entirely in the post-gen-7 trainer (no further promotions → no degraded saves) | **BLOWUP @~73k** |
| **JhJQ / Exp1-retry** (live) | 5×7×7 v4, current code | bf16 | 1e-2 / 0.90 / on | →57k+ | gens 1-9: 5.0→**8.6% / 630**, NLL 3.94→3.49 monotone | **STABLE**, promoting |

Constant across every run: batch 4096 (√-scale pivot ⇒ effective lr = displayed lr), tau 1.00/0.50/0.007, drawKeep 1.0, clip 30, illM 1.0, buffer 1M, arenas 400g/900s/promote≥0.53. wd: 1e-4 except sMe9/ysdg/KXvb-era 1e-3.

#### What the retro-probe adds
1. **Blowup damage rides INSIDE promoted candidates at onset** (Exp4 gen-4: champion itself probes 6.40) — arena score (in-dist, vs sibling) and OOD quality decouple exactly at onset. The 50blk promoted a champion at NLL 13.1.
2. **WjRY's saved champions never degraded** — the rot lived in the trainer after the last promotion; promotion lock-out preserved the champion line. Matches the lock-out-ratchet reading.
3. **KbHZ continuation is healthy and slightly above its May self** (top1 6.7→7.0-7.4%, pElo 562→571-585) — the complCE excursion's NLL dip (6.58→4.81) and re-climb (→6.27) is a clean lab demonstration of the NLL-sharpness confound; top1/pElo never degraded. 3×3 + current code + native cool regime = fine through +41k steps so far.
4. **sMe9 (11.1%/709) and bzw3 (17.3%/876) set the reference bars**: the cool-regime 3×3 beat every hot-regime 3×3 measurement; the 7×7 marathon dwarfs everything.
5. ysdg/KXvb single points are unremarkable-healthy — no blowup evidence anywhere in the v3/fp32 era or the 12-block v4 era.

### STATUS / handoff
Hypotheses eliminated: capacity, width, depth, params, learning rate, code-regression. Surviving: 3×3 second-conv / 3×3 stem (confounded, needs isolation). **The only way to finish Track 2 is a controlled experiment** that flips ONE axis at a time from a stable config. Recommended decisive run: take a known-stable config (e.g. basic30, 5-block, 128ch, **7×7+7×7**, 7×7 stem — Exp-1-like, fast on basic30) and flip ONLY the block-2 kernel to 3×3 (then separately only the stem to 3×3); train from scratch ~10k steps each, watch wide-set NLL. If the 3×3 variant blows up and the 7×7 control stays flat → confirmed. Needs a code enabler (`--architecture-file` is a stub/not parsed; or `--resume`; or in-process harness) + GPU headroom — **best run attended** (verify the isolated launch + watch contention with the live run). Until then, **practical guidance: prefer 7×7 (at least for the 2nd block conv + stem) for stable runs; treat 3×3-heavy configs as blowup-prone pending the controlled test.**

---
### (SUPERSEDED — iteration 2 reasoning, kept for history) "it's CAPACITY, not LR"

Compared Exp 4 (256ch) vs **Exp 3 (128ch)** — same lr 0.01, same wd 1e-4, same `full10Ply10Reps210` encoding, same 5 blocks — using the wide-set probe **top-1 (argmax) tactical accuracy** over steps:

| step | Exp 3 (128ch) top1 | Exp 4 (256ch) top1 |
|---|---|---|
| ~2500 | 7.4% | 8.0% |
| ~5200 | **9.0%** | **8.8% (peak)** |
| ~5700 | 9.0% | 5.4% (collapse begins) |
| ~7000 | 8.8% | **3.3%** |
| ~7600 | — | **3.0%** |
| ~9000 | **9.1% (stable)** | 5–6% (erratic) |

**Exp 3 (128ch) climbs to ~9% and holds rock-solid through 9k steps — no blowup. Exp 4 (256ch) peaks ~8.8% at step ~5,225 then collapses to 3–5%.** Same LR. → **The blowup is architecture-driven (capacity), not the learning rate.** Exp 4's in-dist entropy also starts dropping right at the peak (pEnt 2.74@5081 → 2.53@8891) = mild in-dist over-sharpening, catastrophic OOD.

**Coherent mechanism (ties to Exp 1 weight forensics):** the 128ch/5-block net was *capacity-saturated* (Exp 1: 0% dead channels, ~93% rank). The 256ch net has **surplus capacity**. The self-play distribution is quiet/narrow (diverge ~1.5, ~85% draws at this stage). The net first learns the genuine signal (top1 climbs to ~8.8% by ~5.2k, matching Exp 3's ~9% ceiling), then — signal exhausted — the **surplus capacity over-fits/over-sharpens the narrow distribution**, degrading OOD (tactical) generalization while in-dist metrics stay ~clean. The narrower net has no surplus, so it plateaus and holds.

**Teaching point:** more capacity is not free here. On a low-information self-distillation distribution, network capacity in excess of the distribution's content over-fits → OOD collapse, invisible to in-dist [STATS]. Fix is NOT lower LR; candidates: (a) richer self-play (raise temperature / broaden distribution to *use* the capacity), (b) stronger regularization (wd↑, maybe dropout), (c) match capacity to distribution.

**Caveat (confounds):** Exp 4 differs from Exp 3 on 5 axes (256ch, stem 3×3, block 7×7+3×3, SE/2, FC256), not just width — so "capacity" is the leading but not yet *isolated* cause. Confirmatory experiments below isolate it.

### Confirmatory experiments — SAFE path confirmed (HOME-isolation)
App is **not sandboxed** (no app-sandbox entitlement), so a second instance can be fully isolated from the user's live run by overriding `HOME` (separate `~/Library` → separate UserDefaults, sessions, logs, LastSessionPointer — no pollution of the user's params/pointer, which a naive second instance WOULD clobber). Enabler: `loadSessionFrom(url:, startAfterLoad:true)` already exists; add a `--resume <path>` CLI flag wired to it. Then set up `$FAKE_HOME/Library/Application Support/DrewsChessMachine/Sessions/<copied jaq1-3>` and run with `HOME=$FAKE_HOME`. Vary params by editing the copied session.json (avoids live-override ordering issues).

Experiments (resume jaq1-3 ≈ step 4988, ~600–1000 steps each, watch wide-set top1/NLL):
- **R0** baseline (lr0.01/wd1e-4): reproduce the collapse? (validates the isolated harness)
- **R1** wd 3e-4 (and/or 1e-3): does regularization prevent it? (tests "surplus capacity over-fit" fix)
- **R2** a **128ch** variant of the Exp-4 arch from scratch-ish: isolate width from the other 4 axes (does halving width alone kill the blowup?).
- **R3** higher self-play temperature / broader distribution: does enriching the distribution prevent it?

### (original) Experiment design (needs a resume enabler)
`--train` builds fresh; no headless resume of a specific checkpoint exists. NEXT: add a `--resume <session-path>` CLI flag (mirror `--arch-sweep`/`--sweep` discipline) that loads a given `.dcmsession` and starts Play-and-Train, honoring `--parameters` overrides. Then run, resuming jaq1-3 (`~/Library/Application Support/DrewsChessMachine/Sessions/20260608-220114-20260608-5-cwkO-promote.dcmsession`, ~step 4988) for ~600 steps each:
- **R0 baseline** (lr 0.01, wd 1e-4): reproduce NLL 3.8→5+? (validates resume + baseline)
- **R1** wd 3e-4: prevents? (regularization)
- **R2** lr 0.005: prevents? (LR)
- **R3** (killer discriminator for H1) raise replay ratio / slow self-play so the buffer barely changes: if NLL still blows up with a near-frozen distribution → NOT distribution-narrowing; if it stays clean → distribution-narrowing confirmed.
