# R-elitecyc plateau: checkpoint analysis (2026-10-26)

Checkpoints: elitecyc trainer steps 250,000 (LR 0.00249), 291,000 (trough 0.002, best wide pElo), 384,964 (abort save, LR ~0.013 rising), and R-elitecyc-anneal step 30,000 (LR 0.0002). Build 2620. Reproduce: `zsh run_analysis.sh && python3 -I analyze.py`.

| metric | 250k | 291k | 385k | anneal30k |
|---|---:|---:|---:|---:|
| wide pElo | 1753.4 | 1786.0 | 1737.4 | 1787.0 |
| wide NLL | 1.8889 | 1.8695 | 1.9066 | 1.8663 |
| wide top-1 | 2451/4435 | 2514/4435 | 2420/4435 | 2516/4435 |
| wide top-5 | 3912/4435 | 3927/4435 | 3878/4435 | 3935/4435 |
| wide avg prob of right move | 0.2257 | 0.2282 | 0.2241 | 0.2290 |
| wide avg rank | 3.013 | 2.920 | 3.225 | 2.927 |
| 200 pElo | 1457.9 | 1468.9 | 1463.4 | 1463.4 |
| 200 NLL | 1.6850 | 1.6705 | 1.6782 | 1.6410 |
| 200 top-1 | 126/200 | 128/200 | 127/200 | 127/200 |
| 200 top-5 | 186/200 | 189/200 | 188/200 | 189/200 |
| 200 avg prob of right move | 0.2474 | 0.2492 | 0.2503 | 0.2539 |
| 200 avg rank | 2.295 | 2.210 | 2.120 | 2.095 |
| wide mean entropy (nats) | 2.734 | 2.730 | 2.735 | 2.732 |
| wide mean top-1 prob | 0.2930 | 0.2947 | 0.2949 | 0.2955 |
| wide mean illegal mass | 7.84e-04 | 6.92e-04 | 8.54e-04 | 5.86e-04 |
| policy logit |max| (peak) | 24.5 (61.3) | 24.2 (61.1) | 23.6 (60.8) | 23.5 (61.4) |
| value mean W/D/L | 0.414/0.121/0.465 | 0.438/0.096/0.466 | 0.443/0.114/0.444 | 0.403/0.132/0.465 |
| dead_channels | 0 | 0 | 0 | 0 |
| mostly_off_channels | 0 | 0 | 0 | 0 |
| always_on_channels | 0 | 0 | 0 | 0 |
| running_variance_max_over_median | 446.2 | 478.5 | 487.6 | 484.1 |
| running_variance_worst_site | blocks.2.bn1 | blocks.2.bn1 | blocks.2.bn1 | blocks.2.bn1 |
| se_zero_velocity_units | 0 | 0 | 0 | 0 |
| value_fc1_zero_velocity_units | 0 | 0 | 0 | 0 |
| non_finite_values | 0 | 0 | 0 | 0 |

| wide theme | 250k | 291k | 385k | anneal30k | 291k→385k | 385k→anneal |
|---|---:|---:|---:|---:|---:|---:|
| Deflection | 156/305 | 156/305 | 150/305 | 160/305 | -6 | +10 |
| DiscoveredAttack | 115/311 | 114/311 | 115/311 | 120/311 | +1 | +5 |
| Endgame | 189/315 | 196/315 | 187/315 | 193/315 | -9 | +6 |
| Fork | 183/315 | 184/315 | 185/315 | 191/315 | +1 | +6 |
| HangingPiece | 272/309 | 271/309 | 262/309 | 265/309 | -9 | +3 |
| MateIn1 | 303/482 | 302/482 | 288/482 | 313/482 | -14 | +25 |
| MateIn2 | 251/548 | 269/548 | 258/548 | 261/548 | -11 | +3 |
| Middlegame | 193/315 | 199/315 | 189/315 | 200/315 | -10 | +11 |
| Opening | 162/290 | 168/290 | 156/290 | 169/290 | -12 | +13 |
| Pin | 161/310 | 175/310 | 162/310 | 166/310 | -13 | +4 |
| Promotion | 203/312 | 203/312 | 207/312 | 210/312 | +4 | +3 |
| Sacrifice | 113/315 | 117/315 | 107/315 | 111/315 | -10 | +4 |
| Skewer | 150/308 | 160/308 | 154/308 | 157/308 | -6 | +3 |

| wide rating | n | 250k top-1 / NLL | 291k top-1 / NLL | 385k top-1 / NLL | anneal30k top-1 / NLL |
|---|---:|---:|---:|---:|---:|
| 400-799 | 566 | 75.1% / 1.419 | 77.0% / 1.384 | 73.9% / 1.445 | 77.0% / 1.384 |
| 800-1199 | 818 | 74.4% / 1.509 | 75.8% / 1.487 | 73.8% / 1.527 | 76.2% / 1.494 |
| 1200-1599 | 767 | 57.9% / 1.858 | 59.3% / 1.828 | 58.3% / 1.866 | 61.1% / 1.823 |
| 1600-1999 | 781 | 45.7% / 2.103 | 46.9% / 2.087 | 45.1% / 2.120 | 47.4% / 2.076 |
| 2000-2399 | 777 | 39.5% / 2.254 | 40.9% / 2.232 | 36.8% / 2.271 | 39.9% / 2.237 |
| 2400-2799 | 646 | 43.7% / 2.071 | 44.6% / 2.082 | 44.3% / 2.098 | 43.0% / 2.063 |
| 2800-3199 | 80 | 33.8% / 2.292 | 38.8% / 2.263 | 33.8% / 2.273 | 37.5% / 2.257 |

- 250k → 291k: top-1 gained 198, lost 135 (net +63); per-puzzle NLL change mean -0.0194, mean |change| 0.1680
- 291k → 385k: top-1 gained 172, lost 266 (net -94); per-puzzle NLL change mean +0.0371, mean |change| 0.2288
- 385k → anneal30k: top-1 gained 189, lost 93 (net +96); per-puzzle NLL change mean -0.0403, mean |change| 0.1496
- 291k → anneal30k: top-1 gained 177, lost 175 (net +2); per-puzzle NLL change mean -0.0032, mean |change| 0.1721
