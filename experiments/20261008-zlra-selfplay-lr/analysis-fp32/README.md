# R-elitecyc-fp32: what went wrong

R-elitecyc-fp32 resumed R-elitecyc exactly at trainer step 384,964, on the same schedule, batches and corpus feed,
but computing in fp32 instead of bf16 (launch: `../launch/elitecyc_fp32_launch.sh`, build 2669). It was stopped by
SIGINT on 2026-11-13 at 23:2x CST (abort save at 413,778) after it fell 330–380 pElo behind the bf16 line.

**Cause: MPSGraph computes a wrong fp32 convolution weight gradient when N·H·W·C_in·k_h·k_w > 2³¹ and the
convolution's output has another consumer.** In this architecture that is every block's `conv1` (batch 4096 ×
64 squares × 128 channels × 9×9 = 2.72·10⁹ elements), whose output feeds a training-mode BatchNorm (batch mean,
batch variance and normalize all read it). The wrong gradient has a large component along the weights, so fp32
training inflated every `conv1` weight norm (13 → 26 at 390k → 43 at 410k) and the network degraded. bf16 takes a
different kernel and is correct; the bf16 line (R-elitecyc) is unaffected.

## Evidence

### Matched checkpoints, large puzzle set (90,378 puzzles)

| trainer step | bf16 pElo | fp32 pElo | bf16 NLL | fp32 NLL |
|---:|---:|---:|---:|---:|
| 390,000 | 1765.1 | 1436.6 | 1.826 | 2.276 |
| 400,000 | 1755.0 | 1430.9 | 1.835 | 2.273 |
| 410,000 | 1767.6 | 1391.2 | 1.831 | 2.335 |

Files: `large_probe_step*.jsonl` (build 2669); bf16 from `../../20261108-elitecyc-heldout/large_probe.jsonl`.
Both lines trained on identical batches (`[BATCH-HASH]` equal at 400,000 and 413,000).

### The derive is not the cause

The fp32 start file holds every tensor bit-identical to its bf16 source (SHA-1 per tensor, all 102 equal) and scores
the same on the bundled puzzle sets (lichess-wide 1739.5 vs 1737.4). Step 1 of both lines has the same loss
(pLoss 2.5439 vs 2.5443).

### Step-1 gradients, same weights and batch

One-step exact resumes of the same state in fp32 and bf16; gradient recovered as v₁ − 0.9·v₀ from the saved
velocity (`repro/step1_gradient_compare.py`). Every tensor's fp32 gradient matches bf16 to cos ≥ 0.993 except:

| tensor | cos(g_fp32, g_bf16) | ‖g_fp32‖ / ‖g_bf16‖ |
|---|---:|---:|
| blocks.0.conv1.weight | 0.573 | 1.10 |
| blocks.1.conv1.weight | 0.443 | 1.40 |
| blocks.2.conv1.weight | 0.417 | 1.65 |
| blocks.*.conv2.weight (same shape, one consumer) | 0.999 | 1.00 |
| blocks.2.res_ln (feeds tower_final_bn) | — | fp32 ≈ 0 (correct: BN after it cancels γ/β) |

bn1/bn2 γ/β gradients agree (≥ 0.997), so the gradient arriving at `conv1`'s output is right; only its weight
gradient is wrong.

### BatchNorm scale invariance

A convolution feeding a training-mode BatchNorm cannot change the loss by scaling its weights, so its true gradient
is orthogonal to its weights per output channel. In the saved files (`repro/velocity_weight_orthogonality.py`) bf16
`conv1` velocity stays orthogonal (|cos| ≤ 0.009); fp32 `conv1` velocity reaches mean cos −0.3 to −0.37 (pointing to
grow the weights) by 390k, and its weight norm doubles, then triples. Every other conv→BN (stem, policy, value)
stays orthogonal in fp32.

### Standalone reproduction (no DCM code)

`repro/conv_weight_gradient_overflow.swift`: Y = conv2d(X, W) (NCHW, OIHW, same padding, 8×8 board),
loss = Σ R·Y, with Y also read as a graph target. The graph's dW is compared with an exact float64 CPU
correlation of the graph's own X and dY.

| batch | kernel | C_in = C_out | N·64·C_in·k² | relative error |
|---:|---:|---:|---:|---:|
| 3072 | 9 | 128 | 2.04·10⁹ | 1.1·10⁻⁶ |
| 3236 | 9 | 128 | 2.147·10⁹ (< 2³¹) | 1.1·10⁻⁶ |
| 3264 | 9 | 128 | 2.166·10⁹ (> 2³¹) | **0.169** |
| 4096 | 9 | 128 | 2.72·10⁹ | **0.747** |
| 5312 | 7 | 128 | 2.13·10⁹ (< 2³¹) | 1.3·10⁻⁶ |
| 5376 | 7 | 128 | 2.16·10⁹ (> 2³¹) | **0.094** |
| 4096 | 1 / 3 / 5 / 7 | 128 | < 2³¹ | ≤ 1.3·10⁻⁶ |
| 4096 | 9 | 24 / 64 | < 2³¹ | ≤ 1.3·10⁻⁶ |

The error starts exactly where N·H·W·C_in·k² passes 2³¹, for two kernel sizes, which points to a 32-bit element
index overflowing in the weight-gradient (im2col-style) path. With Y not read by anything else the same shape is
exact (5·10⁻⁶), and bf16 is at rounding level (1.7·10⁻³) everywhere. `graph.run` and a `.level1` compiled
executable give identical results. `repro/block_conv1_weight_gradient.swift` reproduces it in a 3-block
pre-activation tower: fp32 batch 4096 `conv1` dW error 73–80%, batch 512 4·10⁻⁷, bf16 1.7·10⁻³.

Measured on macOS 27.2 (26B123), Apple M5 Max. Not yet checked on another macOS version.

### What is and isn't affected (fp32, batch 4096, 9×9, 128 channels)

| quantity | relative error |
|---|---:|
| forward output Y | 1.4·10⁻⁶ (also at batch 8192) |
| input gradient dX | 1.3·10⁻⁶ |
| **weight gradient dW, output also read** | **0.747** |
| weight gradient dW, output read only by the loss | 5·10⁻⁶ |

Only the weight gradient is wrong, so inference (no gradients) is correct. `repro/conv_forward_and_input_gradient.swift`.

### Why bf16 is correct

| dtype | batch 4096 (2.7·10⁹ elements) | batch 8192 (5.4·10⁹, past 2³²) |
|---|---|---|
| fp32 | 0.75 | MPS aborts allocating a 43,486,543,872-byte buffer |
| bf16 | 1.6·10⁻³ | 1.7·10⁻³ |
| fp16 | 1.8·10⁻⁴ | 1.9·10⁻⁴ |

43,486,543,872 bytes is exactly 8 bytes per element of batch × 64 × C_in × k²: for fp32 MPS materializes a buffer of that
product for the weight gradient, and its indexing breaks past 2³¹ elements. bf16 and fp16 stay right past 2³², so they use a
different kernel without that buffer. The fp32 kernel is chosen only when the convolution's output has another consumer.

### Workarounds (`repro/conv_weight_gradient_workarounds.swift`)

| form (fp32, output also read) | batch 4096 | batch 8192 |
|---|---:|---:|
| plain | 0.747 | abort (40.5 GiB) |
| identity op on the second reader | 0.747 (optimized away) | |
| `+ 0` on the output | 0.747 (optimized away) | |
| batch split in 2 / 4, concatenated | 5.0·10⁻⁶ / 3.8·10⁻⁶ | 5.2·10⁻⁶ (4) |
| **input-channel split in 2, summed** | **7.5·10⁻⁶** | **8.7·10⁻⁶** |

Input-channel split, also at batch 16,384 (1.4·10⁻⁵) and 32,768 (1.9·10⁻⁵), and inside the real 3-block structure
(`repro/block_conv1_weight_gradient.swift`, 4th argument 2: 7·10⁻⁶ against 0.76 unsplit, ~19% slower in that
microbenchmark). The batch axis is dynamic in DCM's graphs, so the app uses the input-channel split
(`ConvolutionWeightGradientWorkaround`, build 2672+; `documentation/plans-active/FP32_CONV_WEIGHT_GRADIENT_WORKAROUND_PLAN.md`).

### Fix validated on the real model

1-step fp32 resume of the 384,964 state on build 2672: gNorm 0.745 (bf16 0.745; build 2669 fp32 0.877). Every
convolution weight's gradient now matches bf16 at cos ≥ 0.998 (`conv1` was 0.42–0.57). The small-gradient tensors that
differed before the fix (`res_ln`, `stem.bn.bias`, cos 0.6–0.94) differ by exactly the same amounts after it: they sit
before normalizations that cancel most of their gradient, so bf16 rounding is the likely difference; not yet proven.

## Audit of earlier fp32 runs

`audit_fp32_runs.py` over every model file (`--describe-model`, 5,748 safetensors + 21 legacy `.dcmmodel`) and every
`[RUN]` line in 10,227 session logs:

| line | precision | convs | batch | over 2³¹ from batch | affected |
|---|---|---|---:|---:|---|
| R-elitecyc-fp32 (E709 seg after 384,964; 32 files) | fp32 | 9×9 / 128 | 4096 | 3,237 | **yes** |
| v3 8-block 3×3 line (30 files, 2026-06, no lineage record) | fp32 | 3×3 / 128 | not recorded | 29,128 | no (no run used ≥ 29k) |
| KbHZ and legacy 8/16-block 3×3 `.dcmmodel` (17 files) | fp32 | 3×3 / 128 | ≤ 4096 | 29,128 | no |
| 2026-10-19 inference-dtype speed arm (GUI `--train`, 55 s) | fp32 | — | — | — | no training step |
| `--sharpness` (E-heldout, fp32 compute) | fp32 | 9×9 / 128 | 256 | 3,237 | no |

Every other model file (5,686 safetensors, 4 `.dcmmodel`) is bf16. The `[RUN]` lines that name a precision
(`policy_tail=does_not_apply` is fp32; the field exists from 2026-10-07) show five fp32 starts: the speed arm above (three
launches) and R-elitecyc-fp32 (and its 1-step diagnostic). Runs from before `[RUN]` are covered by their saved files only; a
deleted fp32 run with 9×9 kernels or a batch past the thresholds would be missed, but no 9×9 tower existed before the
elite-corpus (bf16) runs. Only R-elitecyc-fp32 trained on the defect.

## Reproduce

```sh
swiftc -O repro/conv_weight_gradient_overflow.swift -o /tmp/convgrad
/tmp/convgrad 4096 9 128 none none   # args: batch kernel channels pre(none|scale|silu|bn) post(none|bn)
```
