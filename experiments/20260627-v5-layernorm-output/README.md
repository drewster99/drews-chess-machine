# 2026-06-27 — v5: per-block output LayerNorm, the wd / momentum sweep, and the long corpus-replay lineage

**Status:** done (lineage ended at cum step 859,769 on 2026-08-05; no run active).

Primary docs, which stay where they are and are only summarized here:

- [`documentation/v5-layernorm-output.md`](../../documentation/v5-layernorm-output.md) — design, the v4 failure runs that motivated it, the full per-1000-step table for segments 0-2, and the wd sweep.
- [`documentation/v5-lineage.md`](../../documentation/v5-lineage.md) — the 8-segment chain, cum-step derivations, three chart axes, run 4's truncated tail, checkpoint inventory.
- Data: `documentation/dashboards/data/v5.csv` (856 rows, source of truth), `documentation/dashboards/registry.json` → `runs.v5`, `documentation/dashboards/data/v5-source/`.

**Step numbers.** Every segment restarts its step counter. In this write-up a bare "step" is segment-local; comparable positions are always given as **cum** = `cumstep_base + step`.

## Question

Every v4 (pre-activation, clean-add, ReZero) corpus-replay run trained cleanly and then, somewhere between about 4k and 9k steps, the inference probe collapsed to noise (`nll` well above `ln 4864 ≈ 8.49`) while training-mode metrics only degraded mildly. Bounding the ReZero α (runs 1-4 in the design doc) was necessary but not enough. Run 4 had α fully bounded and still broke at about 8.8k, and a fresh-init control (run 5) broke the same way, only sooner. So the question was:

- Does a channel-wise **LayerNorm on each residual block's output** fix this? It re-centres the residual stream every block and has no train/eval statistics gap.
- If it does, how far does the architecture climb on the lichess `2026-05` corpus?
- Secondary, segments 0-2: is weight decay or effective LR (via momentum) the strength lever?

## Setup

- **Architecture** (embedded JSON in `20260804-v5cont-resume4-replay-latest.safetensors`, identical line in `experiments/20260708-arena-38engines/engines_models.tsv`):
  `in basic30(30) -> stem 128 (7x7) . 5x[7x7+7x7 @128, SE+/4, relu/pre, clean_add, ReZero(0.447·tanh≤0.447), out:layer_norm, drop*1] . act relu . policy intermediate_conv(4864) . value WDL(16->FC128) . bfloat16 . 8,447,028 params`
  - The shipped α bound is `C·tanh(α/C)` with `C = α₀ = 1/√5 ≈ 0.4472`, which was run 4's fix.
  - LayerNorm is over C at each square (ConvNeXt convention), with learnable γ/β and no running stats. That adds 1,280 params over v4.
- **Preset name:** `v5_5block_7x7_lnout`. Fresh start net `20260628-1-tWtk` (`20260627-v5_5block_7x7_lnout-fresh.safetensors`).
- **Corpus:** `20260624-192615-w3aA5b` (lichess std `2026-05`), 20,935,171 games / 1,386,486,078 plies. Offline `--replay-corpus`.
- **Constant hparams, all segments:** lr 0.01 flat with a 500-step warmup and no decay, batch 4096, gradClip 30, replayRatio 0.48, buffer cap 1M, minPrefill 500k.
- **Varying hparams:**

| seg | label | wd | momentum | source |
|---:|---|---|---|---|
| 0 | wd1e-4 | 1e-4 | 0.90 | design doc (logs lost) |
| 1 | wd5e-4 | 5e-4 | 0.90 | design doc (logs lost) |
| 2 | m0.93 | 2.5e-4 | 0.93 | design doc (logs lost) |
| 3-7 | cont-run1..5 | 2.5e-4 | 0.93 | `[REPLAY-HPARAMS]` in each log; segments 3-7 also carry pLabelSmooth 0.1, vLabelSmooth 0.013, complementCE on, sqrtBatchLR on |

  Label smoothing, loss weights, complementCE and sqrtBatchLR are unknown for segments 0-2.
- **Probe:** `--probe-model --probe-set wide`, 4,435 lichess puzzles, deterministic. `pElo` values are on the replay-era (current) probe scale.
- **Machines:**
  - Segments 0-2 ran in a Pacific-TZ VM on the M5 host, at about 1.327 s/step.
  - Segments 3-7 ran native on an M4 Pro, at about 3.42-3.45 s/step.
  - The 2.59× by-time slope break at cum 100,320 compares a VM with native hardware. It is not a chip comparison.
- **Build:** the final checkpoint's header says `built_by_build 2089`, `built_by_git 085356f`. Builds for the earlier segments are not recorded here.

## Runs

The chain was verified from `model_id` / `parent_model_id` in the `__metadata__` of surviving checkpoints on this machine (see Audit notes):

```
20260628-1-tWtk (fresh)
 -> seg0 20260628-2-a5fc -> seg1 20260628-9-OdUt -> seg2 20260629-1-Uf4p
 -> seg3 20260703-1-Dg5v -> seg4 20260714-1-h7vI -> seg5 20260729-1-VZ2j
 -> seg6 20260802-2-Xuub -> seg7 20260805-1-0pTW
```

| seg | label | model_id | cumstep_base | steps | final checkpoint (by metadata) | end next_game_index | session log |
|---:|---|---|---:|---:|---|---:|---|
| 0 | wd1e-4 | `20260628-2-a5fc` | 0 | 45,441 | `20260628-v5_5block_7x7_lnout-replay-latest` (step 45441) | not in header | `dcm_log_20260627-201127.txt` — lost |
| 1 | wd5e-4 | `20260628-9-OdUt` | 45,441 | 15,460 | `…-wd5e4-replay-latest` (step 15460) | 7,851,549 | `dcm_log_20260628-114648.txt` — lost |
| 2 | m0.93 | `20260629-1-Uf4p` | 60,901 | 39,419 | `…-wd2.5e4-m93-replay-latest` (step 39419) | 12,933,495 | `dcm_log_20260628-183831.txt` — lost |
| 3 | cont-run1 | `20260703-1-Dg5v` | 100,320 | 268,506 | `20260702-v5cont-replay-step268506` | 5,689,611 (epoch wrapped) | `dcm_log_20260702-201756.txt` |
| 4 | cont-run2 | `20260714-1-h7vI` | 368,826 | 336,610 | `20260713-v5cont-resume-replay-step336610` | 7,212,496 | `dcm_log_20260713-195047.txt` |
| 5 | cont-run3 | `20260729-1-VZ2j` | 705,436 | 106,333 | `20260728-v5cont-resume2-replay-step106333` | 0 | `dcm_log_20260728-223132.txt` |
| 6 | cont-run4 | `20260802-2-Xuub` | 811,769 | 46,000 used (49,374 run) | `20260802-v5cont-resume3-replay-step46000` | 5,924,823 | `dcm_log_20260802-152830.txt` |
| 7 | cont-run5 | `20260805-1-0pTW` | 857,769 | 2,000 | `20260804-v5cont-resume4-replay-step2000` (= `-replay-latest`) | 6,183,024 | `dcm_log_20260804-221739.txt` |

- **Segments 0-2** are warm starts: weights only, `--start-model` + `--start-game-index`. The buffer cold-refills and warmup re-runs each time.
- **Segment 3 base = 100,320** is the entry checkpoint's own step 39,419 added to 60,901. It is not the CSV's last seg-2 row (99,901).
- **Seg 6.** Around step 49,350, run 4 lost read access to shards 14-45.
  - Its step-49374 save carries false "epoch complete" metadata. That file is kept as `…-step49374-DO-NOT-RESUME`.
  - Run 5 resumed from step 46,000, so segment 7's base is 811,769 + 46,000.
  - The 46,000→49,374 tail is not charted. Details are in v5-lineage §5.

## Results

**Records, from `data/v5.csv`:**

| | value | cum step | segment (raw step) | wallclock |
|---|---:|---:|---|---|
| best pElo | **1770.46** | 638,826 | 4 (270,000) | 2026-07-25 07:09:26 |
| lowest NLL | **1.8445** | 636,826 | 4 (268,000) | 2026-07-25 05:12:05 |

Old notes that say "cum 538.5k" refer to the same checkpoint. Those figures are 100,320 off.

**Per segment, from `data/v5.csv`:**

| seg | label | rows | cum range | best pElo @ cum | lowest nll @ cum | games_fed at end |
|---:|---|---:|---|---|---|---:|
| 0 | wd1e-4 | 41 | 1,000-45,000 | 1501.6 @ 44,000 | 2.190 @ 42,000 | blank (not recorded) |
| 1 | wd5e-4 | 15 | 46,441-60,441 | 1528.8 @ 58,441 | 2.170 @ 59,441 | 7,791,885 |
| 2 | m0.93 | 39 | 61,901-99,901 | 1604.2 @ 97,901 | 2.071 @ 89,901 | 12,879,401 |
| 3 | cont-run1 | 269 | 101,320-368,826 | 1703.96 @ 229,320 | 1.9839 @ 230,320 | 47,582,603 |
| 4 | cont-run2 | 337 | 369,826-705,436 | **1770.46 @ 638,826** | **1.8445 @ 636,826** | 90,998,480 |
| 5 | cont-run3 | 107 | 706,436-811,769 | 1735.38 @ 714,436 | 1.863 @ 714,436 | 104,743,805 |
| 6 | cont-run4 | 46 | 812,769-857,769 | 1731.26 @ 817,769 | 1.8542 @ 855,769 | 110,668,628 |
| 7 | cont-run5 | 2 | 858,769-859,769 | 1559.04 @ 858,769 | 1.9934 @ 858,769 | 110,949,479 |

- **Lineage end** (cum 859,769):
  - pElo 1557.5 and nll 2.1803. That is the post-restart transient of a 2,000-step segment, not the net's level.
  - `games_fed` 110,949,479, which is 110.95M, or 5.30 corpus passes.
  - `elapsed_train_sec` 2,853,850.8 = **792.7 h clamped**; `wall_sec` 2,927,895.6 = **813.3 h raw**.
- **The sweep**, comparing each segment's plateau:
  - wd1e-4 ranged 1376.8-1501.6 over steps 36k-45k.
  - wd5e-4 held a 1403-1529 band.
  - m0.93's plateau rose from about 1535 (steps 18-24k) to 1575-1600 (37-39k), with peak 1604.2 at step 37k (cum 97,901).
  - Full per-mark table is in the design doc. Every value matches v5.csv.

**Arena** (`experiments/20260708-arena-38engines/ratings.txt`, 2026-07-08, Stockfish-anchored with `UCI_Elo 1320` as a scale label only). Only segment 0-2 weights existed then:

| rank | engine name (as in ratings.txt) | model_id | rating ± err |
|---:|---|---|---|
| 2 | `DCM v5_5block_7x7_lnout-wd2.5e4-m93 (latest)` | `20260629-1-Uf4p` (seg 2 end) | 605.9 ± 11.8 |
| 3 | `DCM v5` | `20260629-1-Uf4p` (same file) | 603.8 ± 11.5 |
| 12 | `DCM v5_5block_7x7_lnout-wd5e4 (latest)` | `20260628-9-OdUt` (seg 1 end) | 525.6 ± 10.7 |
| 19 | `DCM v5_5block_7x7_lnout (latest)` | `20260628-2-a5fc` (seg 0 end) | 472.9 ± 10.6 |

- Stockfish is rank 1 at 1320.0, so the best DCM trails it by about 714 Elo.
- Ranks 2 and 3 are the **same weights** entered twice. They agree within error, which is a useful consistency check on the arena.
- The next non-v5 engine is `DCM Qeu8-resume2 (latest)` at 588.4.
- The arena ordering seg0 < seg1 < seg2 matches the probe ordering.

## Conclusion

- **The output LayerNorm fixed the v4 inference collapse.**
  - Segment 0 went straight through the 5-9k zone where every v4 variant died: pElo 1089 at 5k, 1149 at 9k, 1223 at 10k.
  - It kept climbing to 1501.6 at 44k.
  - `bn1Mean` rose slowly and linearly (3.70 → 10.31) instead of exploding. The design doc traces this creep to undecayed γ/α/running-var growth that normalization keeps out of the forward pass.
- **Weight decay was not the strength lever; effective LR (momentum 0.9 → 0.93) was.**
  - wd 5e-4 was harmless but flat (peak 1528.8).
  - wd 2.5e-4 with momentum 0.93 lifted the plateau and set 1604.2 at cum 97,901, with run-best nll 2.071.
  - All three signals agreed: pElo, nll and legalMass.
  - This was a single ordered sweep, not a controlled A/B (see Caveats).
- **The long continuation kept paying.** Segments 3-7 kept segment 2's settings and added about 760k more steps over ~5 corpus passes.
  - pElo peaked at 1770.46 and NLL bottomed at 1.8445, both in segment 4 around cum 637-639k (the 3rd corpus pass).
  - Segments 5-6 hovered below that peak (bests 1735 / 1731).
- **Arena ranking.** At the 2026-07-08 arena, v5's seg-2 checkpoint was the strongest DCM entry.

## Caveats

- **Segments 0-2 logs are permanently lost.** The Pacific-TZ VM was deleted.
  - Their per-mark metrics exist only because they were transcribed into `v5-layernorm-output.md` during the run.
  - Timing and games_fed for segments 1-2 were recovered from checkpoint headers.
  - Segment 0 `games_fed` is blank by necessity. Segment 0 marks 24k, 25k, 27k and 28k were never frozen, so they are missing.
- **The sweep is sequential and warm-started, so it is confounded.**
  - Each change continues from the previous weights, further into the corpus, with a buffer refill and warmup transient.
  - wd and momentum changed together in segment 2.
  - No seeds were repeated. Single-mark pElo swings of ±50 are common (e.g. m0.93 1577.0 → 1506.7 between 7k and 8k).
  - The "momentum is the lever" read therefore rests on a sustained plateau shift, not an isolated A/B.
- **Hparam gaps.** Segments 3-7 ran with label smoothing / complementCE / sqrtBatchLR settings that are unknown for segments 0-2. Some of the seg-2 → seg-3 continuity could be hparam change, not just more training.
- **Time axis.**
  - The 419 steps between the last seg-2 mark (step 39,000) and the seg-2 end (39,419) are not counted, about 557 s.
  - Segment 4's clamp discards 20.54 h of wall time.
  - The VM-versus-native slope break is hardware and is left visible deliberately.
- **Run-1 step-1000 probe** (cum 101,320) is the one row flagged `recovered` (read off a chart). It is not re-probeable.
- **Probe scale.** pElo is the replay-era probe scale. Do not compare it with May-June in-app self-play pElo, e.g. v3's 1269 quoted in the design doc (see Audit notes).

## Follow-ups

- The capacity question the design doc handed off, how high a quarter-size net climbs, went to mini2b (`20260629-3-3MIV` and its resumes). It is tracked separately.
- Label smoothing / complementCE settings for segments 0-2 cannot be recovered. Any ablation of them needs a fresh run.

## Audit notes

**Verified against primary data:**

- **Records.** Best pElo 1770.46 at cum 638,826 (seg 4, 2026-07-25T07:09:26) and lowest nll 1.8445 at cum 636,826 (05:12:05) both match `data/v5.csv` exactly. v5.csv has 856 rows.
- **Totals.** Last row: `games_fed` 110,949,479, elapsed 2,853,850.8 s = 792.74 h, `wall_sec` 2,927,895.6 s = 813.30 h. These match the "110.95M / 792.7 h / 813.3 h" figures.
- **Segment bases and chain.** Checked against safetensors `__metadata__` on **this machine**, `Clark.local` (Apple M5 Max), in `~/Library/Application Support/DrewsChessMachine/Models/`.
  - This is the M5 host that v5-lineage §7/§10 refers to, and the weights are present here.
  - Surviving v5 files:
    - fresh 1 (`20260628-1-tWtk`, parent none)
    - seg 0: 5 frozen (steps 10000/20000/30000/40000/45000) + replay-latest step 45441 (`a5fc`, parent `tWtk`)
    - seg 1: 15 frozen + replay-latest step 15460 (`OdUt`, parent `a5fc`)
    - seg 2: 39 frozen + replay-latest step 39419 (`Uf4p`, parent `OdUt`)
    - seg 3: 268 (`Dg5v`, parent `Uf4p`, steps 2000-268506)
    - seg 4: 231 (`h7vI`, parent `Dg5v`, 107000-336610)
    - seg 5: 68 (`VZ2j`, parent `h7vI`, 9000-106333)
    - seg 6: 46 including DO-NOT-RESUME (`Xuub`, parent `VZ2j`, 2000-49374)
    - seg 7: 3 (`0pTW`, parent `Xuub`, 1000-2000, including replay-latest)
  - The v5 files total 21 GB (`du -ch`).
  - Every cumstep_base is exactly the previous segment's final `training_step`: 45441; 45441+15460=60901; 60901+39419=100320; +268506=368826; +336610=705436; +106333=811769. Segment 7's base is 811769 + 46000 = 857769.
  - The step-46000 header shows `replay_next_game_index` 5,924,823 and epoch 0, which matches v5-lineage §5.
  - Seg 2 end index 12,933,495 = registry `games_base` of seg 3. Seg 1 end index 7,851,549 matches the design doc's "game 7,851,549".
- **Architecture / params.**
  - Embedded JSON: 5 blocks @128, 7×7+7×7, `output_norm: layer_norm`, `rezero_alpha_init 0.4472136`, `scale_and_bias` SE /4, pre-act relu, clean_add, `intermediate_conv` policy (pre-conv 128), `wdl_softmax` value (16 conv ch, 128 hidden), bfloat16, `basic30`.
  - Summing stored tensor elements gives **8,447,028**, matching the design doc and `engines_models.tsv`. "8.45M" is correct rounding.
- **Design-doc table vs v5.csv.** All 95 table rows (wd1e-4 41, wd5e-4 15, m0.93 39), mapped to cum via offsets 0 / 45,441 / 60,901, match the CSV to the digit on pElo, nll, pLoss, vLoss, legalMass, bn1Mean, gNorm and Σαeff².
  - This is a consistency check only: the CSV's segment 0-2 metrics were imported from this doc, so it is not an independent measurement.
  - The doc's stated peaks check out: 1501.6 at 44k, 1528.8 at 13k, 1577.0 at 7k, 1601.7 at 29k, 1604.2 at 37k, final 1583.2 at 39k.
- **Arena names and ratings** come from `ratings.txt`. Model IDs come from `engines_models.tsv`.

**Corrections:**

- **Corpus fraction.** Old claim: segments 0-2 "one continuous pass over the corpus (games 0 → ~8.9M, ~43% of one epoch)" (v5-layernorm-output.md, "Continuation runs"). New claim: segment 2 ended at `replay_next_game_index` **12,933,495**, which is **61.8%** of 20,935,171. Evidence: `…-wd2.5e4-m93-replay-latest.safetensors` header, and registry seg-3 `games_base` 12,933,495. The ~8.9M figure was presumably written mid-segment-2.
- **Stale headline peak.** Old claim: "Outcome (so far)" says m0.93 "reached a new all-time high 1577 @ 7k", and the "Putting it together" paragraph says "(peak 1577)". New claim: the segment-2 peak is **1604.2 at step 37k (cum 97,901)**, and the lineage peak is 1770.46 at cum 638,826. Evidence: v5.csv. The doc's own "Status / open" section already says 1604. Only the earlier snapshot text is stale.
- **Registry ReZero cap.** Old claim: `registry.json runs.v5` `arch_blocks` "ReZero cap 1.0" / `rezero_cap: 1.0`. New claim: the α bound is `0.447·tanh(α/0.447)`, i.e. C = 1/√5. Evidence: embedded arch JSON `rezero_alpha_init 0.4472136`, `engines_models.tsv` "ReZero(0.447·tanh≤0.447)", design doc §2. The 1.0 might be meant as the Σαeff² design cap, but as a per-block cap it is wrong. **Fixed 2026-09-29:** `rezero_cap` 1.0 → 0.4472136, `arch_blocks` "ReZero cap 0.447", and `params` 8,450,000 → 8,447,028. The tracker computes `eff_alpha = cap·tanh(α/cap)` and `sae2 = Σ eff_alpha²` with the registry cap, so every checkpoint-derived row of segments 3-7 (614 rows) carried values computed with cap 1.0 (e.g. eff_alpha up to 0.9146, sae2 up to 4.18, impossible under the real 0.447 bound). They were re-derived from the checkpoints themselves (`replay.py recompute-internals v5`, matched by header `model_id` + `training_step`; the same command with cap 1.0 reproduced the old values exactly before the cap was changed). Segment 3's first mark now reads sae2 0.9593 (was 3.0127) and the lineage end reads 0.9962 (was 4.1821), consistent with the design doc's "Σαeff² ~0.958 inching toward its by-design 1.0 cap". Segments 0-2's sae2, imported from the design doc, were not touched. bn1Mean is independent of the cap and did not change.
- **Weights location.** v5-lineage §7/§10 says the weights are "on the M5". That is correct, and this machine *is* that M5 (M5 Max, `Clark.local`). It is not a separate host.
- **Pre-training baseline.** v5-lineage §4 gives "pElo 1584.2, top1 47.8%, NLL 2.097". This differs from v5.csv's last seg-2 row (1583.2 / 2.149 at step 39,000). Presumably it is a re-probe of the step-39,419 entry checkpoint. Not a contradiction, but the two should not be conflated.

**Unverifiable:**

- **Seg 0 detail.** Seg 0's end corpus index 5,852,793 and its start index cannot be checked: the frozen and latest headers predate the resume metadata, and the logs are lost.
- **Seg 0-2 hyperparameters.** The only record is the design doc. The logs are lost and the headers carry no hparams.
- **Fresh-net probe pElo ≈ 497.** No log, and no probe row at cum 0.
- **v4 failure runs 1-5 and the v3 contrast.** Not re-audited here: pElo, bn1Mean, and v3's 1269 all-time / 1137 at 13k / "1223 = v3's 25k value". Those are other runs and other logs. v3's figures are likely on a different (in-app, May-June) probe scale, so "~230 above v3's 1269" should not be read as same-scale.
- **Weight-internals claims in design doc §3/§4.** Examples: LN γ 1.0→1.85, `bn1.running_var` 1.4→24, policy conv L2 13.0→14.45. They could be recomputed from the surviving frozen checkpoints, but that was not done here.
- **Build/git for segments 0-6.** Only seg 7's final header was read (build 2089, git 085356f).
