# bf16 head-output shared offset — lineage survey (2026-09-28)

Read-only survey of the value and policy heads of the latest checkpoint of every model line, prompted by exact bf16 ties seen in the Lichess bot's recorded W/D/L outputs (avoB). No source changed. **No fix is approved yet** — the options at the end need a decision.

## Headline

- **The value-head degeneration is not avoB-specific.** Every bf16 line trained past ~80k steps is at least degraded. The long lines are BAD, including the whole v5 chain (Dg5v → h7vI → VZ2j → Xuub → 0pTW), avoB, pycz, sFzi, the sf100sl100 vs-UCI line (lTiK, NYAZ, syxR), bzw3, and every Ejp0 checkpoint (replay and self-play champions).
- **The fp32 lines never develop it.** mUF5, KbHZ (532k self-play steps) and wTp3 stay clean; KbHZ's fc2 bias mean is still exactly its init value ln6/3 = 0.597.
- **Recentering `value.wdl_fc2` fixes the value head on every affected line.** Subtracting the mean row and mean bias brings bf16-vs-fp64 CE to ≤ 4e-4 nats and bf16 ties to ≤ 1%, from up to +0.46 nats and 98% ties.
- **The policy head is fine on most lines but BAD on 5:** h7vI, VZ2j, Xuub, 0pTW and Ejp0.
  - Legal logits sit at −120 … −266 with a legal spread of ~0.6, so bf16 spacing there is 1–2.
  - Result: KL(fp64‖bf16) up to 0.16 and exact top-2 ties in up to 63% of positions.
- **The policy offset can't be removed exactly by recentering weights.** The shared part varies by square. The exact fix is computing the final policy projection (and value fc2) in fp32.

## Method

- **Identity:**
  - Headers were read from 3,754 `Models/*.safetensors`, giving 68 distinct metadata `model_id`s.
  - The latest checkpoint per line is the one with the highest `training_step`, then the newest. Files are identified by metadata, never by filename.
  - The champions of 16 `.dcmsession`s were also checked.
- **Architectures present:**
  - All bf16 except mUF5 (fp32); no fp16 lines.
  - Policy heads: simple_conv or intermediate_conv. Value heads: all wdl_softmax.
- **Forward pass:** a numpy forward (`scripts/fwd3.py`) covering every block and head style present.
- **Positions:**
  - 280 positions from the first 4 Lichess bot games (every ply, legal moves from python-chess).
  - 900 positions from corpus w3aA5b shard 45 (300 games × 3 positions); CE is scored on these against the game results and the played moves.
- **bf16 emulation, validated against the bot's recorded outputs** (avoB, 140 of our moves):
  - Exact math inside the network with only the head outputs rounded to bf16 matched the recorded W/D/L exactly in 106/140 moves. Rounding after every op matched only 60/140.
  - It reproduced 3 of the 6 recorded top-move ties exactly, including the triple tie at Qvlv85tG ply 78.
  - So the real engine's damage is dominated by bf16 rounding of the head outputs, and all tables use that emulation. Per-op rounding is also in `results/all_out.jsonl`.
- **Stored weights:** every bf16 Models checkpoint is 100% bf16-exact. Session `trainer.safetensors` files hold fp32 masters.

## Results

The full per-line table is in `results/final_table.txt`; the wide per-metric table is in `results/report.txt`; one JSON row per model is in `results/all_out.jsonl`.

Ratings:
- **fine:** value ΔCE < 0.005 and ties < 10%.
- **degraded:** ΔCE < 0.03, or ties ≥ 10%.
- **BAD:** ΔCE ≥ 0.03.

The June q2Bb experiments (Juoc, Pa83, I78x) have fp64 CE ~6–20 under current code, so their ratings are meaningless. Two untrained seeds (S916, B9SE) failed in the forward and weren't run.

BAD lines (value):

| model_id | step | shared value logit (median) | value ΔCE bf16−fp64 | bf16 ties | policy |
|---|---|---|---|---|---|
| 20260601-11-bzw3-31 | 467065 | +275 | +0.0817 (fp64 CE 1.43 — implausibly high under current code) | 100% | fine |
| 20260708-4-kEiZ | 21086 | −236 | +0.0449 | 37% | fine |
| 20260708-6-sFzi | 88107 | +559 | +0.1223 | 70% | fine |
| 20260709-1-avoB | 277000 | +521 | +0.1061 | 95% | fine |
| 20260711-17-pycz | 120000 | +513 | +0.2126 | 75% | fine |
| 20260703-1-Dg5v | 268506 | +504 | +0.0806 | 61% | fine |
| 20260712-6-lTiK | 220000 | +481 | +0.0438 | 54% | fine |
| 20260714-1-NYAZ | 758000 | +687 | +0.0554 | 78% | fine |
| 20260722-1-syxR | 558000 | +536 | +0.1079 | 62% | fine |
| 20260714-1-h7vI | 336610 | +515 | +0.2301 | 98% | **BAD** (legal −178, KL 3.5e-2, top-2 ties 44%) |
| 20260729-1-VZ2j | 106333 | +515 | +0.1251 | 97% | **BAD** (−257, 8.7e-2, 41%) |
| 20260802-2-Xuub | 49374 | +513 | +0.3321 | 97% | **BAD** (−265, 1.6e-1, 63%) |
| 20260805-1-0pTW | 2000 | +514 | +0.3817 | 82% | **BAD** (−266, 1.0e-1, 61%) |
| 20260727-1-Ejp0 | 1397000 | +513 | +0.4596 | 59% | **BAD** (−138, 3.4e-2, 41%) |

Session champions:
- **Ejp0 self-play (Ejp0-3 at 82k, Ejp0-9 at 197k, Ejp0-59/66/67 at 1.12–1.19M):**
  - Value is BAD in every one (ΔCE +0.12 to +0.18, ties 87–93%).
  - Policy is BAD at Ejp0-3 (KL 5e-2, top-2 ties 26%) and degraded at Ejp0-59/66/67 (legal mean ≈ −122, KL ≈ 5e-3, top-5 ties 96%).
- **bzw3-31:** value BAD (100% ties).
- **eBNC-10 and Mh5n-3:** fine.
- **fp32 KbHZ-22 and wTp3-3:** structurally clean. The wTp3 forward returned NaN under current code, so only its weight structure counts.

## Value-head structure

- **fc2 mean row:**
  - On BAD lines the mean-row norm is 14–47, while per-class residual norms stay ≈ 0.7–1.5 (v5 chain 46–47, avoB 29.5, Ejp0 29.4).
  - On the fp32 lines it stays at its init scale: 0.62 (mUF5), 0.82 (KbHZ), 0.71 (wTp3).
- **Bias drift is diagnostic.** Biases have no weight decay, and the loss gradient on a shared shift is exactly zero in exact math, so any drift in the bias mean is a gradient component that shouldn't exist.
  - fp32 lines keep an fc2 bias mean of exactly 0.597 and a policy bias mean of −0.0000.
  - On bf16 lines the fc2 bias mean drifts: Ejp0 +26.8 … +28.8, Xuub +15.1, h7vI +10.9, avoB +2.51, amlg −0.21. The policy bias mean drifts down to −0.98.
- **The shared value clusters on bf16 exponent boundaries.** For avoB, the 5/25/50/95th percentiles are 513.5 / 517.1 / 521.4 / 1024.3, with 92% of positions in [512, 1024). For Ejp0-67 the median is 512.5. Unrelated BAD lines all have medians of 504–559. A real-valued quantity piling up just above 512 and 1024 points at bf16 rounding in the training loop (inferred).
- **What does and doesn't explain it:**
  - Architecture (1×128, 2×64 15×15, 3×32 15×15, 5×128 7×7, simple_conv 2×64 3×3) and training mode (corpus replay, vs-UCI, self-play) don't. Ejp0's self-play fork inherited the offset from its replay seed and kept it.
  - Weight decay does: wd 2.5e-4 lines (v5 chain) have the largest mean rows (34–47) and wd 5e-4 lines (avoB) reach ~29, consistent with decay being the only restoring force on the fc2 weight. The fc2 bias isn't decayed.
  - Momentum 0.9/0.93 amplifies a steady drift ~10–14×. Gradient clipping at 30 essentially never fires (gNorm 1–6 in the avoB log).

## Does it hurt training?

- **Measured:**
  - avoB's fp64 value CE jumps between checkpoints: 0.733 at 21k, 0.891 at 101k, 0.725 at 141k, 0.978 at 161k, 0.951 at 201k, 0.723 at 277k. The shared value flips sign in between: −258, +7, −270, +494, +521 (`results/trajce_avoB.log`).
  - The trainer's own vLoss rose from ≈ 0.81–0.84 at 60k–140k to ≈ 0.95–1.05 at 180k–260k (`dcm_log_20260709-000611.txt`).
- **Inferred:** the trainer's bf16 forward sees the same quantized value logits, so value gradients are corrupted once the shared term is large. This isn't separable from corpus effects without a GPU run.

## Policy head

- **Normal lines:**
  - Legal logits sit ≈ +9 … +19, about 13 above the all-move mean. That gap is learned legality structure, not a shared artifact.
  - The spread of ~0.6 against a bf16 spacing of 0.0625 gives KL ≈ 1.5e-4, 0% top-1 changes, and policy CE change ≤ 0.003.
  - Exact top-2 ties occur in ~4–8% of positions. This is the source of the bot's 0.50/0.50 ties at τ = 0.01, and it costs almost nothing.
- **BAD lines:**
  - All 4,864 logits carry a shared offset (all-move mean −140 … −304).
  - Policy CE rises by up to +0.156 (0pTW) and +0.107 (Ejp0 at 1.397M).
  - The policy.conv mean-row norm is 3.3–8.0, against 0.3–0.7 on normal lines.
- **Fixes tried:**
  - Subtracting only the mean bias is exact but changes nothing (top-2 ties stay 39–66%).
  - An oracle that removes a per-position constant before rounding fixes it completely (KL ~1.4e-6 to 2e-6, top-2 ties ~0.3%). This is what an fp32 final projection achieves; readback is already fp32.

## Raw-logit consumers (why recentering is safe)

- **Every loss term is softmax-based,** so a shared shift has no effect in exact math:
  - policy CE: `ChessTrainer.swift`, `softMaxCrossEntropy(network.policyOutput, …)`
  - complement CE
  - entropy from `softMax(maskedLogits)`
  - illegal-mass penalty from `softMax(policyOutput)`
  - value CE `softMaxCrossEntropy(valueLogits, …)`
  - Total loss is only value + policy − entropy + illegal.
- **Value diagnostics use the derived softmax quantities.**
- **The only raw-logit readers are telemetry:** `policyLogitAbsMax` and the policy final-weight L2 norm. Inference readback is raw logits, but the CPU sampler masks and softmaxes per position.
- **Caveats for recentering during training:**
  - Recenter the fp32 master *and* the momentum velocity, or the velocity re-adds the component.
  - Weight decay then only shrinks the non-functional mean row.
  - The two telemetry values will change.

## Suspected cause

- **Measured:** the offset appears only under bf16 compute.
- **Deterministic contributor found:** both CE targets are built in bf16.
  - The value target (ε = 0.013) sums to 1.000854 after bf16 rounding.
  - The policy target (ε = 0.1) falls short of 1 by ~1e-3 on average for 20–40 legal moves.
  - If MPSGraph's fused `softMaxCrossEntropy` backward is `p − y` (not `p·Σy − y`), a target with Σy ≠ 1 puts a non-zero gradient on the shared shift. **This assumption is untested; no test in the repo pins it.**
- **The policy numbers agree:** the predicted bias-mean drift, lr × 1/(1−μ) × E[w·(1−Σy)]/76, matches the observed avoB rate of −6.2e-7 per step with (1−Σy) ≈ 1e-3 and a positive-advantage weight ≈ 0.45.
- **The value numbers don't fully agree:** the label mismatch alone predicts ≈ +7.9 of fc2 bias-mean drift over avoB's 277k steps. Observed was −0.11, then +2.2, and several lines drift negative. So there's a second, state-dependent contributor, most likely bf16 rounding of the softmax probabilities in the backward pass (consistent with the 512/1024 clustering).

## Fix options (none implemented)

1. Build the CE targets and compute both CE losses in fp32 (widen the logits before `softMaxCrossEntropy`).
2. Run the final value fc2 and policy projection in fp32 at inference.
3. Periodically recenter value fc2 (the mean row and mean bias, the fp32 master and the momentum velocity) during training.
4. Do a one-time exact recentering of `value.wdl_fc2` on existing checkpoints (verified here to restore the value head).
5. Add a one-case MPSGraph test pinning `softMaxCrossEntropy`'s gradient when Σy ≠ 1.

## Side notes

- NYAZ's file is named `…step978000…`, but its metadata `training_step` is 758000.
- The Ejp0 replay checkpoint at step 1,397,000 has an unusually high fp64 value CE: 1.20, against ≈ 0.72–0.76 for the other replay lines.
- Replay checkpoints in Models store bf16-rounded weights, so each resume re-seeds the fp32 masters from bf16 values.

## Reproducing

The scripts in `scripts/` read local paths under `~/Library/Application Support/DrewsChessMachine/`. They need numpy and python-chess (`pip install chess`; the survey used a local copy on `sys.path`).

1. `scan2.py` reads every Models header and writes `scan2.json`. `lines_list.py` then writes `latest.json` (latest file per `model_id`).
2. `posset.py` builds `posset.pkl` from the bot's game records and corpus shard 45. It's regenerable, so it isn't checked in (≈ 9.3 MB).
3. `analyze.py jobs.json all_out.jsonl` runs `fwd3.py`'s forward per model and appends one JSON row per model. `report.py` renders `report.txt`.
4. `traj2.py` and `trajce.py` produce the avoB trajectory. `validate2.py` and `variants.py` check the emulation against the recorded outputs. `recenter.py` checks the mean-row fix.
