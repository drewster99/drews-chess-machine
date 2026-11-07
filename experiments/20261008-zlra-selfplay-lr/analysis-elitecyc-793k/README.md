# R-elitecyc at 793k: network and heads analysis (2026-11-07)

Checkpoints: elitecyc trainer steps 500,000, 600,000, 700,000 and 793,000 (segment 19, live run), beside the 2026-10-26 plateau analysis's 250k, 291k and 384,964 (`../analysis-elitecyc-plateau`). Same commands, build 2620 (the trainer's own frozen build). Reproduce: `./run.sh && python3 -I analyze.py && python3 -I loss_history.py`.

Architecture: basic24 input, 5x5 stem, 3 pre-activation residual blocks of 128 channels with 9x9 convs (SiLU, layer-norm output, no SE, no ReZero), intermediate-conv policy head (128 ch, policy tail fp32_from_pre_bn), W/D/L value head (24-ch conv, 128-unit FC). 8,271,279 parameters, bf16 compute.

LR schedule (`parameters-zlra-elitecyc-decay.json`, 10,000-step cycles, batch 4096, weight decay 3e-4, momentum 0.9 constant): peak 0.02 / trough 0.002 to step 386,000; geometric decay to 1e-3 / 4.47e-4 at 486,000; to 1e-3 / 1e-4 at 586,000, held after.

## Checkpoints

| metric | 250k | 291k | 385k | 500k | 600k | 700k | 793k |
|---|---:|---:|---:|---:|---:|---:|---:|
| wide pElo | 1753.4 | 1786.0 | 1737.4 | 1805.1 | 1806.1 | 1801.5 | 1814.4 |
| wide NLL | 1.8889 | 1.8695 | 1.9066 | 1.8420 | 1.8342 | 1.8412 | 1.8296 |
| wide top-1 | 2451 | 2514 | 2420 | 2551 | 2553 | 2544 | 2569 |
| wide top-5 | 3912 | 3927 | 3878 | 3959 | 3973 | 3956 | 3977 |
| 200 pElo | 1457.9 | 1468.9 | 1463.4 | 1502.4 | 1491.1 | 1485.5 | 1491.1 |
| 200 NLL | 1.6850 | 1.6705 | 1.6782 | 1.6048 | 1.5869 | 1.5988 | 1.5880 |
| 200 top-1 | 126 | 128 | 127 | 134 | 132 | 131 | 132 |
| 200 top-5 | 186 | 189 | 188 | 191 | 192 | 190 | 192 |
| wide avg prob right move | 0.2257 | 0.2282 | 0.2241 | 0.2330 | 0.2341 | 0.2333 | 0.2343 |
| wide mean policy entropy | 2.734 | 2.730 | 2.735 | 2.725 | 2.723 | 2.724 | 2.724 |
| wide mean top-1 prob | 0.2930 | 0.2947 | 0.2949 | 0.2990 | 0.3002 | 0.2994 | 0.2997 |
| wide mean illegal mass | 7.8e-04 | 6.9e-04 | 8.5e-04 | 4.9e-04 | 4.5e-04 | 4.5e-04 | 4.3e-04 |
| policy logit abs max (peak) | 24.5 (61.3) | 24.2 (61.1) | 23.6 (60.8) | 23.3 (62.5) | 23.2 (63.1) | 23.1 (63.2) | 23.0 (63.7) |
| puzzle value W/D/L (side to move) | 0.414/0.121/0.465 | 0.438/0.096/0.466 | 0.443/0.114/0.444 | 0.415/0.129/0.455 | 0.427/0.104/0.469 | 0.408/0.133/0.458 | 0.416/0.125/0.459 |
| numerics overall | degraded | degraded | degraded | degraded | degraded | degraded | degraded |
| policy offset x init | 4.582 | 4.588 | 4.653 | 4.690 | 4.696 | 4.700 | 4.705 |
| value offset x init | 0.0076 | 0.0070 | 0.0055 | 0.0049 | 0.0048 | 0.0047 | 0.0046 |
| bf16 policy KL vs fp32 | 2.1e-04 | 1.6e-04 | 1.4e-04 | 2.3e-04 | 2.1e-04 | 2.7e-04 | 2.7e-04 |
| bf16 value CE delta | -7.1e-04 | -8.2e-04 | +1.0e-03 | +1.1e-03 | +5.9e-04 | +9.8e-04 | -6.0e-04 |
| bf16 policy top-2 ties | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rv max/median (site) | 446 | 478 | 488 | 494 | 503 | 483 | 491 |
| worst BN |mean|/std | 48.64 at block2_bn1 | 44 at block2_bn1 | 46.28 at block2_bn1 | 56.39 at block2_bn1 | 61.12 at block2_bn1 | 61.81 at block2_bn1 | 61.46 at block2_bn1 |
| rv>=10x channels | 39 | 43 | 45 | 44 | 44 | 43 | 45 |
| value fc1 low-velocity units | 3 | 4 | 2 | 6 | 4 | 8 | 2 |
| value fc1 median unit velocity | 0.00758 | 0.008173 | 0.007263 | 0.01039 | 0.01129 | 0.0119 | 0.01049 |
| value fp32 CE (audit positions) | 0.9985 | 0.9947 | 0.9875 | 0.9995 | 0.9838 | 0.9952 | 0.9959 |
| value start-position W/D/L | 0.479/0.105/0.417 | 0.489/0.072/0.439 | 0.482/0.083/0.435 | 0.468/0.120/0.412 | 0.484/0.078/0.437 | 0.470/0.108/0.422 | 0.470/0.103/0.427 |
| policy legal spread median | 0.5749 | 0.5688 | 0.576 | 0.5742 | 0.5739 | 0.5715 | 0.5685 |
| degraded findings | 6 | 6 | 11 | 11 | 11 | 11 | 11 |

| BN site rv max/median (rv>=10x ch) | 250k | 291k | 385k | 500k | 600k | 700k | 793k |
|---|---:|---:|---:|---:|---:|---:|---:|
| stem.bn | 4.6 (0) | 4.6 (0) | 4.2 (0) | 4.2 (0) | 4.4 (0) | 4.2 (0) | 4.4 (0) |
| blocks.0.bn1 | 15.2 (4) | 16.2 (4) | 18.2 (4) | 19.3 (4) | 19.4 (4) | 19.6 (4) | 19.7 (4) |
| blocks.0.bn2 | 9.4 (0) | 9.9 (0) | 10.0 (1) | 10.4 (1) | 11.0 (1) | 10.9 (1) | 11.5 (1) |
| blocks.1.bn1 | 225.2 (14) | 248.0 (17) | 283.0 (18) | 319.5 (18) | 330.7 (18) | 330.5 (18) | 344.5 (19) |
| blocks.1.bn2 | 9.9 (0) | 10.1 (1) | 10.1 (1) | 10.1 (1) | 10.2 (1) | 10.2 (1) | 10.3 (1) |
| blocks.2.bn1 | 446.2 (18) | 478.5 (18) | 487.6 (18) | 494.0 (17) | 503.3 (17) | 483.2 (16) | 490.6 (17) |
| blocks.2.bn2 | 2.5 (0) | 2.5 (0) | 3.0 (0) | 3.1 (0) | 3.0 (0) | 3.1 (0) | 3.1 (0) |
| tower_final_bn | 35.7 (3) | 36.7 (3) | 35.9 (3) | 40.0 (3) | 41.7 (3) | 40.8 (3) | 41.8 (3) |
| policy.pre_bn | 7.6 (0) | 7.6 (0) | 8.1 (0) | 8.1 (0) | 8.0 (0) | 8.1 (0) | 8.0 (0) |
| value.bn | 2.5 (0) | 3.2 (0) | 4.3 (0) | 4.2 (0) | 4.5 (0) | 4.0 (0) | 4.1 (0) |
| BN site min/median pass-through P | 250k | 291k | 385k | 500k | 600k | 700k | 793k |
|---|---:|---:|---:|---:|---:|---:|---:|
| blocks.0.bn1 | 0.124555/0.337843 | 0.120785/0.332953 | 0.114077/0.326335 | 0.109973/0.326352 | 0.109382/0.326673 | 0.108929/0.326929 | 0.108588/0.327280 |
| blocks.0.bn2 | 0.155935/0.318081 | 0.156703/0.319154 | 0.157471/0.316703 | 0.156645/0.315536 | 0.156723/0.315815 | 0.156966/0.315772 | 0.156972/0.315956 |
| blocks.1.bn1 | 0.095803/0.220772 | 0.095247/0.220266 | 0.094588/0.218143 | 0.093685/0.217906 | 0.093500/0.217520 | 0.093236/0.216907 | 0.093195/0.217023 |
| blocks.1.bn2 | 0.092668/0.239060 | 0.092084/0.237994 | 0.090645/0.237493 | 0.091285/0.237005 | 0.091155/0.237103 | 0.091096/0.237582 | 0.091168/0.237611 |
| blocks.2.bn1 | 0.060630/0.201766 | 0.060326/0.199361 | 0.059222/0.195886 | 0.059167/0.194815 | 0.059056/0.194506 | 0.058994/0.194913 | 0.058968/0.195077 |
| blocks.2.bn2 | 0.069178/0.285114 | 0.067208/0.284585 | 0.062793/0.283837 | 0.061174/0.284268 | 0.060977/0.285013 | 0.060775/0.285153 | 0.060646/0.285636 |
| tower_final_bn | 0.253626/0.516622 | 0.252766/0.514443 | 0.251198/0.508838 | 0.246522/0.507186 | 0.245730/0.506931 | 0.245127/0.506740 | 0.244605/0.506527 |
| policy.pre_bn | 0.186972/0.588925 | 0.185293/0.589313 | 0.183108/0.590393 | 0.182564/0.591663 | 0.182524/0.591849 | 0.182504/0.592015 | 0.182810/0.592171 |
| value.bn | 0.126637/0.158873 | 0.128004/0.159956 | 0.129734/0.162746 | 0.130538/0.163491 | 0.130751/0.163901 | 0.130950/0.164230 | 0.131084/0.164396 |

| wide theme (correct) | 250k | 291k | 385k | 500k | 600k | 700k | 793k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deflection | 156 | 156 | 150 | 163 | 163 | 162 | 164 |
| DiscoveredAttack | 115 | 114 | 115 | 127 | 128 | 124 | 126 |
| Endgame | 189 | 196 | 187 | 195 | 195 | 196 | 192 |
| Fork | 183 | 184 | 185 | 190 | 192 | 192 | 196 |
| HangingPiece | 272 | 271 | 262 | 268 | 269 | 267 | 268 |
| MateIn1 | 303 | 302 | 288 | 313 | 314 | 317 | 319 |
| MateIn2 | 251 | 269 | 258 | 271 | 267 | 263 | 276 |
| Middlegame | 193 | 199 | 189 | 207 | 207 | 203 | 203 |
| Opening | 162 | 168 | 156 | 164 | 167 | 168 | 169 |
| Pin | 161 | 175 | 162 | 174 | 175 | 174 | 176 |
| Promotion | 203 | 203 | 207 | 203 | 206 | 205 | 203 |
| Sacrifice | 113 | 117 | 107 | 112 | 104 | 111 | 113 |
| Skewer | 150 | 160 | 154 | 164 | 166 | 162 | 164 |

| wide rating bucket: top-1 % / NLL | 250k | 291k | 385k | 500k | 600k | 700k | 793k |
|---|---:|---:|---:|---:|---:|---:|---:|
| 400-799 (n=566) | 75.1 / 1.419 | 77.0 / 1.384 | 73.9 / 1.445 | 77.6 / 1.361 | 77.7 / 1.356 | 77.6 / 1.364 | 78.6 / 1.345 |
| 800-1199 (n=818) | 74.4 / 1.509 | 75.8 / 1.487 | 73.8 / 1.527 | 76.8 / 1.460 | 77.1 / 1.452 | 76.5 / 1.455 | 77.0 / 1.449 |
| 1200-1599 (n=767) | 57.9 / 1.858 | 59.3 / 1.828 | 58.3 / 1.866 | 61.0 / 1.790 | 60.9 / 1.779 | 61.1 / 1.794 | 61.1 / 1.777 |
| 1600-1999 (n=781) | 45.7 / 2.103 | 46.9 / 2.087 | 45.1 / 2.120 | 48.8 / 2.049 | 48.5 / 2.037 | 48.4 / 2.045 | 49.2 / 2.029 |
| 2000-2399 (n=777) | 39.5 / 2.254 | 40.9 / 2.232 | 36.8 / 2.271 | 40.8 / 2.226 | 41.2 / 2.221 | 40.9 / 2.226 | 41.6 / 2.215 |
| 2400-2799 (n=646) | 43.7 / 2.071 | 44.6 / 2.082 | 44.3 / 2.098 | 44.4 / 2.048 | 44.0 / 2.046 | 44.0 / 2.046 | 44.9 / 2.046 |
| 2800-3199 (n=80) | 33.8 / 2.292 | 38.8 / 2.263 | 33.8 / 2.273 | 38.8 / 2.228 | 40.0 / 2.211 | 37.5 / 2.225 | 35.0 / 2.220 |

- 385k -> 500k: top-1 gained 262, lost 131 (net +131); per-puzzle NLL change mean -0.0646, mean |change| 0.2046
- 500k -> 600k: top-1 gained 64, lost 62 (net +2); per-puzzle NLL change mean -0.0077, mean |change| 0.0630
- 600k -> 700k: top-1 gained 48, lost 57 (net -9); per-puzzle NLL change mean +0.0070, mean |change| 0.0597
- 700k -> 793k: top-1 gained 66, lost 41 (net +25); per-puzzle NLL change mean -0.0116, mean |change| 0.0616
- 385k -> 793k: top-1 gained 273, lost 124 (net +149); per-puzzle NLL change mean -0.0770, mean |change| 0.2115

## Training step lines, whole run (per 50k trainer steps)

| trainer steps | lines | pLoss | vLoss | pEnt | pD | gNorm |
|---|---:|---:|---:|---:|---:|---:|
| 0k-50k | 445 | 2.574 | 0.849 | 2.827 | 0.127 | 0.395 |
| 50k-100k | 631 | 2.532 | 0.847 | 2.812 | 0.138 | 0.483 |
| 100k-150k | 471 | 2.501 | 0.835 | 2.804 | 0.151 | 0.542 |
| 150k-200k | 363 | 2.531 | 0.833 | 2.795 | 0.128 | 0.596 |
| 200k-250k | 515 | 2.513 | 0.836 | 2.794 | 0.134 | 0.653 |
| 250k-300k | 645 | 2.505 | 0.829 | 2.792 | 0.137 | 0.671 |
| 300k-350k | 685 | 2.517 | 0.828 | 2.788 | 0.130 | 0.709 |
| 350k-400k | 599 | 2.496 | 0.833 | 2.789 | 0.142 | 0.741 |
| 400k-450k | 369 | 2.495 | 0.827 | 2.785 | 0.137 | 0.758 |
| 450k-500k | 347 | 2.509 | 0.829 | 2.781 | 0.130 | 0.812 |
| 500k-550k | 350 | 2.490 | 0.832 | 2.781 | 0.137 | 0.857 |
| 550k-600k | 537 | 2.506 | 0.828 | 2.782 | 0.128 | 0.843 |
| 600k-650k | 866 | 2.509 | 0.830 | 2.780 | 0.128 | 0.875 |
| 650k-700k | 519 | 2.489 | 0.833 | 2.781 | 0.139 | 0.911 |
| 700k-750k | 419 | 2.500 | 0.826 | 2.781 | 0.132 | 0.888 |
| 750k-800k | 405 | 2.503 | 0.827 | 2.777 | 0.129 | 0.919 |

## Findings

- **Strength is on a plateau.** Wide pElo rose +68 from 385k to 500k (the LR drop from 0.02/0.002 to ~1e-3). Since 500k it has moved +9 over 293k steps (1805 -> 1814; NLL 1.842 -> 1.830). The 100k-step top-1 churn (about 50-65 puzzles gained and lost each way) is larger than the net change, so most of the movement is noise.
- **The training losses have been flat since about 150k.** pLoss is 2.49-2.53, vLoss 0.826-0.836 and pEnt about 2.78 in every 50k window. The step lines are measured on fresh corpus batches (epoch 5 of 14.6M games), so this is a fit limit, not memorization.
- **gNorm climbed steadily, from 0.40 to 0.92,** while the LR fell 20x. There are no clips. This is unexplained here. Weight norms, which would test whether weight decay at 3e-4 under the low LR explains it, were not measured.
- **No dead, parked or non-finite channels.** Every SiLU site has a minimum pass-through of at least 0.059. The value FC1 has no zero-velocity units (2 low).
- **Residual-stream outlier channels.**
  - blocks.2.bn1's running-variance max/median has been about 490 since 291k.
  - blocks.1.bn1's is still growing: 225 at 250k, 345 at 793k.
  - Channel 13 is the outlier at both blocks.2.bn1 and tower_final_bn.
  - The worst |mean|/std (block2_bn1) rose from 46 to 61.
  - All 11 "degraded" audit findings are these two BN inputs plus the policy offset.
  - bf16 still matches fp32: policy KL 2.7e-4, 0 top-2 ties, value CE delta under 1.1e-3.
- **Policy head:**
  - The shared offset is 4.7x init, rising slowly. This is the `policy_offset_drift` warning; softmax can't see it, and bf16 shows no ties.
  - Illegal mass halved since 385k, to 4.3e-4.
  - Entropy on puzzles is 2.72 nats, flat.
- **Value head:**
  - The shared offset is shrinking, now 0.0046x init (fine).
  - The start position reads W/D/L 0.47/0.10/0.43.
  - The batch pD (about 0.13) matches the corpus draw rate (13.1%).
  - On the audit's 4,096 positions from 63 of DCM's own Lichess games, value CE is 0.996. That is about what a constant guess at the base rates scores (about 0.99), so the value head says little about how DCM's own bot games end.
