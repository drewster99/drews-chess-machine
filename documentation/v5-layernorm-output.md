# v5 architecture — per-block output LayerNorm (`v5_5block_7x7_lnout`)

A running log + post-mortem of the **v5** architecture experiment: `v4` plus a
channel-wise **LayerNorm on each residual block's output**, trained from a fresh
net by offline `--replay-corpus` over the lichess `2026-05` corpus. Written
2026-06-28 while the run is in progress (≈ step 43k); the data table is a
snapshot and grows as the run continues.

## What v5 is

`v5_5block_7x7_lnout` = the `v4_5block_7x7` recipe **byte-for-byte**, plus one
orthogonal addition: each block returns `LayerNorm(merge)` instead of `merge` —
a channel-wise LayerNorm over the C dimension at each board square (ConvNeXt
convention), with per-channel learnable γ/β and **no running statistics**.

- 5 blocks, 7×7+7×7 convs, 128 channels, scale-and-bias SE, pre-activation
  (ResNet-v2), clean-add skip, ReZero branch scalar (tanh soft-bound, see below),
  bf16 compute. **8,447,028 params** (v4 + 2·128 γ/β per block = +1,280).
- The LayerNorm adds **no train/eval gap** — it recomputes mean/variance every
  forward, identically at train and inference. That is the entire point (below).

**Run config:** fresh minted start net (`20260628-1-tWtk`, fresh-probe pElo ≈ 497),
offline corpus-replay over `/Volumes/20260624-192615-w3aA5b` (lichess
`2026-05`: 46 sealed shards, **20,935,171 games / 1,386,486,078 plies**, ≈ 66
plies/game). lr 0.01 flat (500-step warm-up, no decay), weight decay 1e-4
(conv/linear only), batch 4096, momentum 0.9, replay-ratio 0.48, buffer cap 1M,
grad-clip 30, epoch limit 1 (one pass; still in epoch 0 throughout). Probes are
the `--probe-model --probe-set wide` battery (4,435 lichess puzzle positions);
`pElo` is the battery's derived rating, `nll` its mean negative log-likelihood.

## Outcome (so far)

v5 **cleared the entire 5–9k inference-collapse zone that every v4 variant died
in** (see below) with inference *improving*. The base run (`wd 1e-4`) climbed to
**pElo ~1502 by 44k** — ~230 above v3's all-time peak of 1269 (which v3 only
reached at ~27k). Two hyperparameter **continuation runs** followed (warm-started
from the prior run's weights, corpus position continuous — see "Continuation
runs" below): `wd 5e-4` held a ~1493–1529 high band (peak **1529 @ 13k**), and
`wd 2.5e-4 + momentum 0.93` (higher effective LR, ~14.3× steady-state vs 0.9's
10×) reached a new **all-time high 1577 @ 7k** with a recent band center ~1525.
v5 is by a wide margin the strongest net produced on this corpus.

The WD sweep also answered a side question: **weight decay is not a strength
lever here.** 5e-4 was harmless (didn't hurt strength) but did *not* pull the
`pLogitAbsMax` *mean* down out of its ~15.4–16.2 band — it only modestly lowered
the peak. The strength lever is LR / momentum (the `m0.93` run is the only one to
break the ~1529 ceiling, if marginally). `bn1Mean` and `Σαeff²` are unaffected by
WD (they're driven by the undecayed LN/BN γ and the ReZero α), and keep their
slow pre-WD linear creep straight across both cutovers.

## Data table (probed every 1000-step mark; out-model frozen + kept at each)

Columns: probe `pElo` / `nll`; train `pLoss` / `vLoss` / `legalMass` (1 − illegal
mass); `bn1Mean` = max\|bn1 running-mean\| across blocks (highway-drift proxy,
parsed from the frozen checkpoint); `gNorm` = pre-clip grad L2 at that step line;
`Σαeff²` = Σ of squared effective ReZero α (= `C·tanh(α/C)`), the
residual-stream-variance quantity; `pLogitAbsMax` = mean / peak of per-position
max\|raw policy logit\| over the wide set.

The three runs chain continuously through the corpus (each warm-started from the
previous run's weights at the next corpus game; per-run step counters reset to 0
at each cutover). Rows are labeled by run; within each run the `step` is that
run's own step counter.

| run | step | pElo | nll | pLoss | vLoss | legalMass | bn1Mean | gNorm | Σαeff² | pLogitAbsMax (mean/peak) |
|---|---|---|---|---|---|---|---|---|---|---|
| wd1e-4 | 1000 | 857 | 3.270 | 3.14 | 0.81 | 0.904 | 3.70 | 2.016 | 0.785 | 14.0 / 18.75 |
| wd1e-4 | 2000 | 1009 | 2.910 | 3.03 | 0.82 | 0.940 | 4.72 | 1.594 | — | 14.8 / 19.9 |
| wd1e-4 | 3000 | 1055 | 2.780 | 2.98 | 0.81 | 0.957 | 5.16 | 1.523 | — | 15.2 / 20.6 |
| wd1e-4 | 4000 | 1033 | 2.860 | 2.91 | 0.82 | 0.959 | 5.38 | 1.438 | — | 15.5 / 21.0 |
| wd1e-4 | 5000 | 1089 | 2.720 | 2.91 | 0.81 | 0.966 | 5.63 | 1.531 | 0.745 | 15.6 / 21.1 |
| wd1e-4 | 6000 | 1123 | 2.684 | 2.88 | 0.83 | 0.969 | 5.84 | 1.391 | 0.742 | 15.9 / 21.6 |
| wd1e-4 | 7000 | 1121 | 2.715 | 2.88 | 0.81 | 0.971 | 6.13 | 1.188 | 0.741 | 15.9 / 21.9 |
| wd1e-4 | 8000 | 1105 | 2.720 | 2.88 | 0.82 | 0.974 | 6.28 | 1.289 | 0.742 | 16.0 / 22.4 |
| wd1e-4 | 9000 | 1149 | 2.635 | 2.86 | 0.80 | 0.977 | 6.41 | 1.281 | 0.744 | 16.1 / 22.6 |
| wd1e-4 | 10000 | 1223 | 2.515 | 2.86 | 0.81 | 0.978 | 6.50 | 1.188 | 0.749 | 15.9 / 22.1 |
| wd1e-4 | 11000 | 1134 | 2.663 | 2.83 | 0.82 | 0.980 | 6.66 | 1.344 | 0.755 | 15.9 / 23.1 |
| wd1e-4 | 12000 | 1195 | 2.562 | 2.84 | 0.82 | 0.982 | 6.81 | 1.391 | 0.758 | 16.2 / 23.4 |
| wd1e-4 | 13000 | 1224 | 2.501 | 2.86 | 0.83 | 0.984 | 6.94 | 1.953 | 0.765 | 15.9 / 23.8 |
| wd1e-4 | 14000 | 1222 | 2.524 | 2.83 | 0.81 | 0.985 | 7.09 | 1.289 | 0.770 | 15.8 / 24.5 |
| wd1e-4 | 15000 | 1208 | 2.568 | 2.81 | 0.83 | 0.986 | 7.16 | 1.141 | 0.772 | 16.0 / 26.5 |
| wd1e-4 | 16000 | 1207 | 2.570 | 2.78 | 0.82 | 0.987 | 7.28 | 1.531 | 0.779 | 16.3 / 27.6 |
| wd1e-4 | 17000 | 1277 | 2.487 | 2.75 | 0.82 | 0.988 | 7.41 | 1.672 | 0.782 | 16.1 / 27.9 |
| wd1e-4 | 18000 | 1238 | 2.513 | 2.77 | 0.82 | 0.988 | 7.66 | 1.125 | 0.786 | 16.0 / 26.9 |
| wd1e-4 | 19000 | 1312 | 2.419 | 2.78 | 0.80 | 0.989 | 7.78 | 1.211 | 0.785 | 15.9 / 27.5 |
| wd1e-4 | 20000 | 1345 | 2.367 | 2.77 | 0.81 | 0.989 | 8.00 | 1.031 | 0.789 | 16.0 / 27.6 |
| wd1e-4 | 21000 | 1293 | 2.461 | 2.81 | 0.80 | 0.990 | 8.06 | 1.055 | 0.791 | 16.2 / 27.5 |
| wd1e-4 | 22000 | 1289 | 2.453 | 2.79 | 0.80 | 0.990 | 8.19 | 1.133 | 0.794 | 16.3 / 28.5 |
| wd1e-4 | 23000 | 1318 | 2.428 | 2.81 | 0.80 | 0.991 | 8.38 | 1.062 | 0.799 | 16.1 / 27.3 |
| wd1e-4 | 26000 | 1338 | 2.430 | 2.71 | 0.81 | 0.992 | 8.69 | 1.094 | 0.808 | 15.8 / 26.9 |
| wd1e-4 | 29000 | 1284 | 2.443 | 2.78 | 0.81 | 0.992 | 9.06 | 0.949 | 0.818 | 16.1 / 26.5 |
| wd1e-4 | 30000 | 1352.6 | 2.340 | 2.78 | 0.80 | 0.993 | 9.13 | 1.125 | 0.822 | 16.2 / 26.6 |
| wd1e-4 | 31000 | 1331.0 | 2.401 | 2.75 | 0.81 | 0.992 | 9.19 | 1.023 | 0.826 | 15.9 / 26.75 |
| wd1e-4 | 32000 | 1285.6 | 2.401 | 2.73 | 0.80 | 0.993 | 9.375 | 1.008 | 0.827 | 16.65 / 26.875 |
| wd1e-4 | 33000 | 1415.3 | 2.288 | 2.66 | 0.81 | 0.994 | 9.375 | 2.797 | 0.829 | 16.1 / 26.6 |
| wd1e-4 | 34000 | 1377.8 | 2.346 | 2.75 | 0.80 | 0.994 | 9.438 | 0.945 | 0.832 | 15.9 / 25.75 |
| wd1e-4 | 35000 | 1338.2 | 2.367 | 2.72 | 0.83 | 0.994 | 9.562 | 1.648 | 0.835 | 16.5 / — |
| wd1e-4 | 36000 | 1407.6 | 2.262 | 2.77 | 0.80 | 0.994 | 9.750 | 1.086 | 0.838 | 15.97 / 25.0 |
| wd1e-4 | 37000 | 1376.8 | 2.298 | 2.78 | 0.82 | 0.994 | 9.812 | 1.109 | 0.839 | 16.21 / 25.75 |
| wd1e-4 | 38000 | 1378.8 | 2.358 | 2.77 | 0.81 | 0.994 | 9.875 | 1.000 | 0.843 | 16.06 / 25.375 |
| wd1e-4 | 39000 | 1381.4 | 2.313 | 2.78 | 0.82 | 0.995 | 10.000 | 0.996 | 0.843 | 16.24 / 25.875 |
| wd1e-4 | 40000 | 1404.5 | 2.288 | 2.61 | 0.84 | 0.995 | 9.938 | 3.453 | 0.848 | 15.98 / 24.75 |
| wd1e-4 | 41000 | 1422.0 | 2.306 | 2.83 | 0.81 | 0.994 | 10.062 | 2.391 | 0.852 | 15.70 / 24.75 |
| wd1e-4 | 42000 | 1468.7 | 2.190 | 2.83 | 0.85 | 0.994 | 10.125 | 4.094 | 0.856 | 16.16 / 24.75 |
| wd1e-4 | 43000 | 1482.6 | 2.216 | 2.77 | 0.83 | 0.994 | 10.125 | 1.078 | 0.858 | 15.72 / 24.25 |
| wd1e-4 | 44000 | 1501.6 | 2.214 | 2.72 | 0.81 | 0.995 | 10.250 | 1.062 | 0.860 | 15.93 / 25.125 |
| wd1e-4 | 45000 | 1422.0 | 2.309 | 2.75 | 0.80 | 0.995 | 10.312 | 0.996 | 0.864 | 15.91 / 25.25 |
| wd5e-4 | 1000 | 1434.3 | 2.290 | 2.75 | 0.79 | 0.9950 | 10.375 | 0.887 | 0.866 | 15.75 / 25.375 |
| wd5e-4 | 2000 | 1506.2 | 2.203 | 2.80 | 0.81 | 0.9945 | 10.625 | 1.172 | 0.870 | 15.75 / 24.375 |
| wd5e-4 | 3000 | 1435.9 | 2.287 | 2.69 | 0.80 | 0.9950 | 10.688 | 1.062 | 0.872 | 15.81 / 24.125 |
| wd5e-4 | 4000 | 1444.6 | 2.301 | 2.70 | 0.80 | 0.9950 | 10.750 | 1.180 | 0.876 | 15.65 / 24.375 |
| wd5e-4 | 5000 | 1452.8 | 2.262 | 2.75 | 0.80 | 0.9945 | 10.688 | 1.086 | 0.879 | 15.95 / 24.25 |
| wd5e-4 | 6000 | 1411.2 | 2.257 | 2.69 | 0.83 | 0.9952 | 10.750 | 1.156 | 0.884 | 15.94 / 24.625 |
| wd5e-4 | 7000 | 1403.0 | 2.363 | 2.75 | 0.82 | 0.9950 | 10.938 | 1.438 | 0.887 | 15.40 / 23.5 |
| wd5e-4 | 8000 | 1508.8 | 2.181 | 2.70 | 0.80 | 0.9946 | 10.938 | 0.980 | 0.890 | 15.92 / 23.875 |
| wd5e-4 | 9000 | 1453.8 | 2.290 | 2.55 | 0.84 | 0.9949 | 10.875 | 4.344 | 0.892 | 15.50 / 23.5 |
| wd5e-4 | 10000 | 1501.1 | 2.177 | 2.69 | 0.82 | 0.9950 | 11.062 | 1.141 | 0.895 | 15.51 / 23.5 |
| wd5e-4 | 11000 | 1448.2 | 2.222 | 2.52 | 0.84 | 0.9945 | 11.000 | 3.719 | 0.898 | 15.77 / 23.625 |
| wd5e-4 | 12000 | 1434.8 | 2.244 | 2.64 | 0.82 | 0.9946 | 11.312 | 1.469 | 0.9004 | 16.14 / — |
| wd5e-4 | 13000 | 1528.8 | 2.171 | 2.77 | 0.82 | 0.9949 | 11.188 | 1.164 | 0.902 | 15.91 / 24.0 |
| wd5e-4 | 14000 | 1515.4 | 2.170 | 2.66 | 0.82 | 0.9955 | 11.188 | 1.320 | 0.905 | 15.75 / 23.875 |
| wd5e-4 | 15000 | 1493.4 | 2.231 | 2.73 | 0.80 | 0.9957 | 11.312 | 1.000 | 0.907 | 15.41 / 23.5 |
| m0.93 | 1000 | 1462.6 | 2.228 | 2.69 | 0.82 | 0.9948 | 11.375 | 0.988 | 0.911 | 15.65 / 22.625 |
| m0.93 | 2000 | 1556.0 | 2.121 | 2.73 | 0.81 | 0.9957 | 11.438 | 1.359 | 0.914 | 15.70 / 23.75 |
| m0.93 | 3000 | 1467.2 | 2.215 | 2.69 | 0.83 | 0.9957 | 11.438 | 1.789 | 0.917 | 15.55 / 23.625 |
| m0.93 | 4000 | 1491.8 | 2.200 | 2.77 | 0.83 | 0.9957 | 11.562 | 1.734 | 0.918 | 15.65 / 23.125 |
| m0.93 | 5000 | 1489.3 | 2.180 | 2.75 | 0.81 | 0.9957 | 11.625 | 1.102 | 0.921 | 15.60 / 23.125 |
| m0.93 | 6000 | 1524.7 | 2.158 | 2.80 | 0.79 | 0.9955 | 11.750 | 1.406 | 0.922 | 15.70 / 23.75 |
| m0.93 | 7000 | 1577.0 | 2.138 | 2.72 | 0.82 | 0.9957 | 11.625 | 0.980 | 0.924 | 15.46 / 22.5 |
| m0.93 | 8000 | 1506.7 | 2.187 | 2.73 | 0.82 | 0.9955 | 11.625 | 0.902 | 0.925 | 15.38 / 22.625 |
| m0.93 | 9000 | 1538.0 | 2.164 | 2.77 | 0.79 | 0.9960 | 11.875 | 0.910 | 0.927 | 15.33 / 22.25 |
| m0.93 | 10000 | 1475.9 | 2.160 | 2.73 | 0.83 | 0.9953 | 11.812 | 1.992 | 0.929 | 15.92 / 23.5 |
| m0.93 | 11000 | 1511.3 | 2.232 | 2.75 | 0.80 | 0.9959 | 11.938 | 0.867 | 0.930 | 15.36 / 22.5 |
| m0.93 | 12000 | 1550.3 | 2.170 | 2.72 | 0.80 | 0.9962 | 11.875 | 0.863 | 0.932 | 15.20 / 22.375 |
| m0.93 | 13000 | 1514.9 | 2.256 | 2.70 | 0.80 | 0.9956 | 11.812 | 1.133 | 0.934 | 14.98 / 22.0 |
| m0.93 | 14000 | 1521.6 | 2.224 | 2.72 | 0.79 | 0.9958 | 12.062 | 0.828 | 0.935 | 15.22 / 22.375 |
| m0.93 | 15000 | 1572.4 | 2.125 | 2.70 | 0.82 | 0.9959 | 12.125 | 0.871 | 0.937 | 15.44 / 22.0 |
| m0.93 | 16000 | 1496.9 | 2.245 | 2.70 | 0.81 | 0.9964 | 12.125 | 0.957 | 0.937 | 15.18 / 22.0 |
| m0.93 | 17000 | 1589.3 | 2.130 | 2.73 | 0.82 | 0.9961 | 12.250 | 0.977 | 0.939 | 15.19 / 22.625 |
| m0.93 | 18000 | 1525.2 | 2.201 | 2.72 | 0.81 | 0.9964 | 12.375 | 0.809 | 0.939 | 15.07 / 21.625 |
| m0.93 | 19000 | 1536.0 | 2.180 | 2.70 | 0.82 | 0.9964 | 12.125 | 0.855 | 0.941 | 15.26 / 22.625 |
| m0.93 | 20000 | 1550.3 | 2.198 | 2.56 | 0.82 | 0.9964 | 12.062 | 1.805 | 0.943 | 15.11 / 21.875 |
| m0.93 | 21000 | 1520.6 | 2.182 | 2.70 | 0.80 | 0.9965 | 12.312 | 0.758 | 0.943 | 15.16 / 22.5 |
| m0.93 | 22000 | 1508.8 | 2.167 | 2.70 | 0.81 | 0.9958 | 12.125 | 0.867 | 0.944 | 15.19 / 21.625 |
| m0.93 | 23000 | 1576.5 | 2.167 | 2.56 | 0.84 | 0.9961 | 12.375 | 1.812 | 0.945 | 14.99 / 21.375 |
| m0.93 | 24000 | 1519.0 | 2.227 | 2.77 | 0.80 | 0.9965 | 12.375 | 0.922 | 0.946 | 15.06 / 21.875 |
| m0.93 | 25000 | 1587.3 | 2.097 | 2.73 | 0.80 | 0.9967 | 12.500 | 0.871 | 0.947 | 15.26 / 22.0 |
| m0.93 | 26000 | 1586.2 | 2.155 | 2.70 | 0.80 | 0.9969 | 12.562 | 0.773 | 0.948 | 15.16 / 21.5 |
| m0.93 | 27000 | 1549.8 | 2.114 | 2.70 | 0.80 | 0.9968 | 12.625 | 0.773 | 0.949 | 15.37 / 22.5 |
| m0.93 | 28000 | 1564.2 | 2.141 | 2.77 | 0.80 | 0.9964 | 12.500 | 0.863 | 0.950 | 15.24 / 22.25 |
| m0.93 | 29000 | 1601.7 | 2.071 | 2.72 | 0.80 | 0.9966 | 12.500 | 0.836 | 0.951 | 15.19 / 21.0 |
| m0.93 | 30000 | 1515.9 | 2.232 | 2.66 | 0.81 | 0.9967 | 12.438 | 1.391 | 0.952 | 14.91 / 21.25 |
| m0.93 | 31000 | 1594.5 | 2.115 | 2.73 | 0.78 | 0.9964 | 12.438 | 0.848 | 0.952 | 14.96 / 21.5 |
| m0.93 | 32000 | 1541.1 | 2.190 | 2.75 | 0.82 | 0.9966 | 12.812 | 0.809 | 0.953 | 14.83 / 21.625 |
| m0.93 | 33000 | 1584.7 | 2.187 | 2.78 | 0.81 | 0.9964 | 12.812 | 0.984 | 0.953 | 14.59 / 20.75 |
| m0.93 | 34000 | 1595.5 | 2.119 | 2.375 | 0.945 | 0.9965 | 12.500 | 8.562 | 0.954 | 14.89 / 21.375 |
| m0.93 | 35000 | 1555.5 | 2.180 | 2.69 | 0.84 | 0.9964 | 12.438 | 0.875 | 0.955 | 14.79 / 21.375 |
| m0.93 | 36000 | 1554.4 | 2.137 | 2.72 | 0.79 | 0.9971 | 12.312 | 0.750 | 0.956 | 15.04 / 21.25 |
| m0.93 | 37000 | 1604.2 | 2.103 | 2.70 | 0.80 | 0.9972 | 12.375 | 0.684 | 0.956 | 14.94 / 20.875 |
| m0.93 | 38000 | 1602.7 | 2.093 | 2.67 | 0.80 | 0.9968 | 12.500 | 0.836 | 0.957 | 14.88 / 20.75 |
| **m0.93** | **39000** | **1583.2** | **2.149** | **2.72** | **0.80** | **0.9971** | **12.500** | **0.770** | **0.958** | **14.84 / 21.0** |

Notes: the wd1e-4 17k row's `Σαeff²` and 2–4k `Σαeff²` cells, the wd1e-4 35k
peak and wd5e-4 12k peak `pLogitAbsMax`, are gaps in collection, not data — left
as `—`. wd1e-4 marks 24/25/27/28k were missed during monitoring lulls (the
rolling out-model overwrites each 1000, so an un-frozen mark is unrecoverable).
`gNorm` is the single value on that step's log line — a spiky per-step quantity
(e.g. wd1e-4's 33k 2.797 / 40k 3.453 / 42k 4.094, wd5e-4's 9k 4.344 / 11k 3.719
are isolated single-step spikes that recovered the next line), not a trend; all
sit well under the clip threshold of 30. Across both run cutovers `bn1Mean`
(10.1 → 10.4 → 11.6) and `Σαeff²` (0.858 → 0.866 → 0.925) continue their pre-WD
linear creep without a kink — confirming both are WD-independent.

## Continuation runs (hyperparameter experiments)

The corpus-replay CLI has **no optimizer / buffer / step resume** — a "continue"
is a warm-start: `--start-model <prev>-replay-latest` (weights only) plus
`--start-game-index <N>` to pick up at the next corpus game. The buffer
cold-refills (500k prefill) and the LR warmup (500 steps) re-runs each time, so
each run's first ~1k steps are a transient; the step counter restarts at 0. The
three runs are otherwise one continuous pass over the corpus (games 0 → ~8.9M,
~43% of one epoch), same net lineage throughout.

- **wd1e-4** (base) — `wd 1e-4`, `lr 0.01` flat, `momentum 0.9`. Stopped via
  SIGINT at step 45441 (corpus game 5,852,793). pElo arc topped 1501.6 @ 44k.
- **wd5e-4** — stopped wd1e-4, bumped weight decay 1e-4 → **5e-4**, everything
  else unchanged; resumed at game 5,852,793. Ran to step 15460 (game 7,851,549).
  **Finding:** harmless to strength (held the ~1493–1529 high band, peak 1529 @
  13k) but did *not* pull the `pLogitAbsMax` mean down out of ~15.4–16.2 — it
  only modestly lowered the peak (24–25 vs the pre-WD 25–28). So 5e-4 is too weak
  to be the containment lever it was tried as.
- **m0.93** — `wd 2.5e-4` (lighter than 5e-4), `lr 0.01`, **`momentum 0.93`**
  (steady-state step amplification 1/(1−μ) = 14.3× vs 0.9's 10× — i.e. a higher
  *effective* LR without touching the warmup schedule). Resumed at game
  7,851,549. **Finding (29k in — a confirmed win):** early on (≤16k) this looked
  like only a marginal, noisy ~+20 edge that wouldn't consolidate. It then
  resolved cleanly: the plateau stepped up — center ~1535 (18–24k) → ~1555–1570
  (22–29k), the floor lifted from the early ~1476 to ~1510–1550, and the *ceiling
  itself kept rising* across the run: 1577 @ 7k → 1589 @ 17k → 1587 @ 25k → **new
  all-time high 1601.7 @ 29k**. Crucially the highs land with record `nll`
  (2.097 @ 25k, then **2.071 @ 29k**, both run-bests) and record `legalMass`
  (0.9969 @ 26k) — pElo, nll, and legal-mass agreeing rules out scoring noise.
  Back-to-back ceiling marks (1587/1586 @ 25/26k) and the >1600 cross put m0.93
  at ~+65 over wd5e-4's ~1505 center and ~+95 over its 1529 best — decisively
  past the noise band, and still slowly climbing at 29k (not yet plateaued).
  μ=0.93 stayed stable throughout (gNorm calm, no instability from the larger
  effective step). Side note: across the m0.93 run the `pLogitAbsMax` *peak*
  drifts to new run-lows (21.0 @ 29k) and the mean bounces ~15.0–15.4 — so the
  lighter 2.5e-4 WD eases the logits slowly over many steps, contra the flat read
  the shorter wd5e-4 window gave.

Takeaway: **weight decay is not a strength lever** on this corpus at this scale —
**LR / momentum is, and decisively so**: the only change between the flat-plateau
wd runs (~1505) and the climbing m0.93 run (~1570 and rising, peak 1601) was the
higher effective LR from μ=0.9→0.93. WD's only clean effect was a slightly lower
logit peak.

## Failed runs — the v4 lineage that motivated v5

All of these are the **v4 architecture** (pre-activation ResNet-v2 + ReZero
branch scalar), same lichess `2026-05` corpus, same hyperparameters as v5 (lr
0.01, batch 4096, etc.); they differ only in how the ReZero α is bounded and in
the start weights. Runs 1–4 warm-started from the same `q2Bb-manual` v4 net; run
5 is a fresh-init control. The arc is: fix the α runaway (runs 1→4), discover the
α fix is **not sufficient** (run 4 still breaks), confirm it's architectural (run
5), then fix the real cause in v5. `bn1Mean` = max\|bn1 running-mean\|; "broke" =
inference probe → noise (`nll` ≫ `ln 4864 ≈ 8.49`).

### Run 1 — unbounded α (raw, no transform). Broke ~4k.

| step | pElo | nll | bn1Mean | block-0 α |
|---|---|---|---|---|
| 1000 | 818 | 3.43 | small | (healthy) |
| 2000 | 911 | 3.13 | small | |
| 3000 | 896 | 3.10 | small | |
| 4000 | 448 | 15.34 | ~46 | **1.37** ← runaway begins |
| 5000 | 463 | 15.86 | ~70,000 | |
| 6000 | 455 | 16.11 | ~750,000 | |
| 7000 | 409 | 16.51 | ~2,000,000 | |
| 8000 | 524 | 12.40 | ~3,900,000 | 28.9 |
| 9000 | 422 | 16.65 | ~7,100,000 | 31.8 |

α ratchets monotonically to 30+, drowning the identity skip; stream + BN
running-mean explode to ~10⁶–10⁷; inference is noise from ~4k on.

### Run 2 — hard clamp `α ← max(min(α, C), 0)`, `C = 4.5·α₀ ≈ 2.0`. Degraded 4k–17k, never recovers past mediocre.

| step | pElo | nll | pLoss | legalMass | bn1Mean |
|---|---|---|---|---|---|
| 1000 | 813 | 3.35 | 3.14 | 0.905 | ~4 |
| 2000 | 923 | 3.06 | 3.02 | 0.937 | ~9 |
| 3000 | 794 | 3.48 | 2.97 | 0.951 | ~20 |
| 4000 | 559 | 10.47 | 2.94 | 0.955 | ~45 (α 1.20, not yet at cap) |
| 5000 | 577 | 6.06 | 3.98 | 0.744 | 6,208 (clamp engaged) |
| 6000 | 392 | 16.33 | 4.88 | 0.504 | 175,000 |
| 9000 | 435 | 11.5 | 3.89 | 0.56 | 1,150,000 |
| 10000 | 582 | 14.43 | 3.89 | 0.66 | 1,262,000 |
| 11000 | 417 | 14.01 | 3.89 | 0.67 | 1,303,000 |
| 12000 | 425 | 12.92 | 3.98 | 0.66 | 1,368,000 |
| 13000 | 596 | 5.64 | 3.84 | 0.69 | 1,450,000 |
| 14000 | 418 | 10.48 | 3.80 | 0.69 | 1,540,000 |
| 15000 | 641 | 5.87 | 3.66 | 0.72 | 1,647,000 |
| 16000 | 433 | 16.18 | 3.61 | 0.73 | 1,696,000 |
| 17000 | 373 | 14.56 | 3.58 | 0.74 | 1,778,000 (worst pElo) |
| 18000 | 620 | 4.60 | 3.47 | 0.77 | 1,884,000 |
| 19000 | 560 | 5.22 | 3.47 | 0.78 | 1,950,000 |
| 20000 | 648 | 4.35 | 3.44 | 0.80 | 2,023,000 |
| 21000 | 564 | 4.08 | 3.47 | 0.79 | 2,073,000 |
| 22000 | 677 | 4.31 | 3.45 | 0.81 | 2,130,000 (best — still mediocre vs ~900 early) |
| 23000 | 399 | 8.64 | 3.47 | 0.82 | 2,195,000 |
| 24000 | 480 | 9.47 | 3.45 | 0.82 | 2,228,000 |
| 25000 | 647 | 3.98 | 3.44 | 0.82 | 2,261,000 |
| 26000 | 596 | 5.14 | 3.44 | 0.82 | 2,310,000 |

Two lessons: the dead zone past C let stored α drift while pinned, and **C ≈ 2.0
is already in the degraded regime** (degradation set in at α ≈ 1.3–1.5, below the
cap), so the tower spent ~17k steps broken and only clawed back to a noisy ~650.

### Run 3 — tanh soft-bound `C·tanh(α/C)`, `C = 1.0`. Best peak yet, then broke ~5.8k.

| step | pElo | nll | pLoss | legalMass | bn1Mean | max α (stored → eff) |
|---|---|---|---|---|---|---|
| 1000 | 823 | 3.32 | 3.13 | 0.905 | 3.5 | 0.82 |
| 2000 | 916 | 3.08 | 3.06 | 0.938 | 6.75 | 0.91 |
| 3000 | 985 | 2.94 | 2.98 | 0.948 | 10.12 | 0.97 |
| 4000 | **1015** | **2.88** | 2.94 | 0.959 | 17.5 | 1.047 → 0.78 |
| 5000 | 471 | 8.97 | 2.91 | 0.965 | 43.25 | 1.211 → 0.84 |
| ~6000 | ~442 | (broke) | 6.09 | 0.18 | ~1,384 | (fully broken) |

tanh killed the dead zone (live gradient everywhere) and gave the best peak of
any run to date (1015 @4k) — but the raw α keeps ratcheting regardless of the
bound, so the **effective** α saturates *at* C on every block; `C = 1.0` ⇒
`Σα² ≈ 4.5` (4.5× the variance-preserving target), the stream mean compounds, and
it breaks ~5.8k.

### Run 4 — tanh `C = α₀ = 1/√N` (C ≈ 0.447), q2Bb start. α fully solved — and it STILL broke ~8.8k.

| step | pElo | nll | pLoss | legalMass | bn1Mean | eff α (Σα²) |
|---|---|---|---|---|---|---|
| 1000 | 835 | 3.36 | 3.13 | 0.90 | 2.4 | 0.40–0.42 (0.84) |
| 2000 | 951 | 2.98 | 3.00 | 0.93 | 4.1 | 0.41 (0.86) |
| 3000 | 976 | 2.95 | 2.97 | 0.95 | 5.4 | 0.42 (0.875) |
| 4000 | 1032 | 2.80 | 2.97 | 0.95 | 6.9 | 0.42 (0.889) |
| 5000 | 1006 | 2.88 | 2.91 | 0.96 | 10.1 | 0.42 (0.90) |
| 6000 | **1050** | 2.79 | 2.94 | 0.96 | 15.8 | 0.42 (0.91) |
| 7000 | 1038 | 2.88 | 2.89 | 0.967 | 28 | 0.43 (0.927) |
| 8000 | 423 | 9.50 | 2.89 | 0.971 | 58 | 0.44 (0.944) ← inference dip, train still pristine |
| 9000 | 580 | 3.37 | 3.88→4.66 | 0.78→0.56 | 408 | 0.44 ← train now degrading |
| 10000 | — | — | ~4.8 | ~0.52 | (broken) | |

**This is the pivotal run.** The α fix is fully working — effective α flat at
~0.42, `Σα² ≈ 0.9` (variance-safe), no runaway. Yet `bn1Mean` still creeps
(2.4 → 58 → 408) and the run breaks ~8.8k with the *exact same* signature as the
α-runaway runs, just delayed and with α innocent. **Conclusion: capping α is
necessary but not sufficient — the residual-stream MEAN drift + BatchNorm
train/eval gap is the real cause**, independent of α. (This is the α bound v5
keeps; v5 adds the LayerNorm that finally fixes the mean drift.)

### Run 5 — fresh-init control, same `C = 1/√N` tanh. Terminated ~4.1k, breaking faster.

| step | pElo | nll | pLoss | legalMass | bn1Mean | eff α |
|---|---|---|---|---|---|---|
| 1000 | 781 | 3.39 | 3.16 | 0.90 | 4.9 | 0.40–0.41 |
| 2000 | 860 | 3.18 | 3.02 | 0.93 | 11.8 | 0.41–0.42 |
| 3000 | 679 | 4.50 | 2.94 | 0.95 | 28.5 | 0.42–0.43 |
| ~4000 | ~582 | — | (3.45 @4.1k) | (0.87 @4.1k) | **213** | (terminated) |

`bn1Mean` climbs ~5× faster than run 4 (28.5 by 3k vs run 4's 5.4) — a fresh
random init breaks the **same way**, only sooner. This rules out the start
weights: the failure is **architectural**, not a property of the `q2Bb` net.

### Healthy contrast — v3 (`v3_8block_3x3`, fp32, post-activation, `activation_gated` skip, NO ReZero, 2.48M)

Not a failed run — the comparator that pointed at the fix. The post-activation
`activation_gated` merge (`ReLU(x + F(x))`) **re-centers the highway every
block**, so it has no mean drift: `bn1Mean` stays flat (3.79 → 4.24 → 4.46 → … →
~5.3 at 13k) and the run sails past the 5–9k zone (pElo 865 → 1137 @13k, all-time
27k peak 1269). v3 demonstrated that *re-centering the highway* is what keeps
`bn1Mean` bounded — v5 achieves the same with a LayerNorm on the output while
keeping v4's stronger pre-activation tower and ReZero.

**Putting it together:** runs 1–3 are α problems; run 4 proves α isn't the whole
story (highway mean drift breaks it even with α perfectly bounded); run 5 proves
it's architectural; v3 shows a re-centered highway stays flat. **v5 = v4 + the
`C = 1/√N` tanh α-bound (from run 4) + a per-block output LayerNorm that
re-centers the highway with no train/eval gap.** Result is the table at the top:
the entire 5–9k death zone cleared, new highs carried through 45k and into the
continuation runs (peak 1577).

## Issues encountered and resolved

### 1. The motivating failure: v4 inference collapse (highway mean-drift × BN train/eval gap)

Every v4 corpus-replay attempt trained cleanly for a few thousand steps, then the
**inference probe collapsed** (probe pElo cratering, `nll` past `ln 4864 ≈ 8.49`
= the saved model evaluating to noise) while the *training-mode* metrics stayed
only mildly degraded. The split is the signature: a **statistics** mismatch, not
a precision one.

Mechanism: v4's pre-activation clean-add highway (`out = x + α·F(x)`) is never
re-centered, so the residual stream accumulates a drifting **mean** down the
tower and across steps. BatchNorm inside the blocks normalizes against its
**running** statistics at inference but **fresh batch** statistics at train — so
as the stream's mean inflates, the stale running stats and the live batch stats
diverge, and the eval path (running stats) produces garbage while the train path
(batch stats) limps along. The proxy `bn1Mean` (max\|bn1 running-mean\|) exploded
in the dying runs (e.g. a fresh-net v4 went 4.9 → 213 and was dead by ~4k; other
v4 variants died by ~8.8k). bf16 precision was tested and **ruled out** — forcing
full fp32 compute on the broken weights left them broken.

**Resolution → v5.** Put a **LayerNorm on each block's output**. LayerNorm
re-centers (and re-scales) the residual stream every block, killing the
mean-drift — and crucially it has **no train/eval statistics gap** (it recomputes
per-forward), so it does not reintroduce the very failure mode it's fixing.
Result: v5 sailed through the entire 5–9k death zone with `bn1Mean` rising gently
and *linearly* (v3-like) instead of exploding, and inference setting new highs
(pElo 1149 → new high at 9k, 1223 at 10k = v3's 25k value at 2.5× the
sample-efficiency).

### 2. The ReZero α runaway and the tanh soft-bound (carried into v5)

The per-block ReZero scalar `α` (`*_res_scale`, init `1/√N`) is trainable,
**un-decayed**, and was used raw — the one parameter with a one-way ratchet and
no restoring force. Unbounded it ran from `1/√N` to **~30+** over a few k steps,
drowning the identity skip and compounding the stream to ~10⁶. The fix went
through two failed forms before the one v5 uses (full write-up:
[`rezero-alpha-clamp.md`](rezero-alpha-clamp.md)):

- **Hard clamp `[0, C]`, `C = 4.5·α₀ ≈ 2.0`** — bounded the magnitude but had a
  dead zone past C (stored α drifted while pinned) and C ≈ 2.0 was *already* in
  the degraded regime; the tower broke and only clawed back to a mediocre ~650.
- **`C = 1.0` tanh** — fixed the dead zone (`C·tanh(α/C)` has live gradient
  everywhere), trained beautifully (pElo 1015 @4k, best of any run at the time),
  then **broke ~5.8k**: the raw α ratchets regardless of the bound, so the
  *effective* α saturates **at** C on every block, and `C = 1.0` gives
  `Σα² ≈ 4.5` — 4.5× the variance-preserving target.
- **`C = α₀ = 1/√N` tanh (shipped)** — with all N blocks saturated at C,
  `Σα² = N·C² = 1`, so the fully-saturated state is itself variance-safe. This is
  what v5 uses; the run's effective α's sit at 0.37–0.44 (ceiling 0.4472),
  Σαeff² ≈ 0.86, none pinned.

### 3. The benign internal "warming" — and why weight decay can't touch it

v5 shows a slow, **linear** internal creep that v3 didn't have: `bn1Mean`
3.7 → 10.1 (≈ 0.12/1k after an initial settling transient) and `Σαeff²`
0.785 → 0.86. It looks alarming but is **decoupled from inference** — pElo set new
highs *at* `bn1Mean` 10. Investigated directly from the frozen checkpoints:

- It is **not** conv-weight growth. Block-4 conv L2 is flat (16.0 → 16.3 over
  40k steps — the existing 1e-4 WD holds it).
- It is growth in the **undecayed scale/affine params**: LayerNorm γ
  (1.0 → 1.85), BN γ (1.10 → 1.65), ReZero α (raw 0.65 → 0.96), and most
  strikingly `bn1.running_var` (1.4 → 24). The block-input activation scale
  compounds through these, and `bn1Mean` (an EMA of the per-channel input mean)
  tracks it.

Because BN/LN renormalize this scale growth **out of the forward pass**, it never
reaches the output — which is why `pLogitAbsMax` mean stays flat ~16 and pElo
keeps rising. **Consequence for tuning:** increasing weight decay would **not**
bound `bn1Mean` — WD acts on conv/linear weights, which are already flat; the
drivers (γ, α, running stats) are un-decayed. The lever that *would* slow it is a
lower learning rate (slows all parameter movement). This corrected an initial
instinct that more WD could rein in `bn1Mean`.

> Terminology caution carried over from the project: a high *initial* draw rate
> or a slowly-rising internal statistic is **not** "collapse." The v4 failure was
> a genuine inference collapse (eval → noise); v5's creep is a bounded,
> inference-invisible internal drift. The distinction is the whole point.

### 4. `pLogitAbsMax` — the output-logit magnitude (the one WD *can* bound)

Separately tracked because large policy logits are a known pre-blow-up signal
(rule of thumb: comfortable < 15, watch > 15, ~18 is the path to blow-up). v5's
**mean** `pLogitAbsMax` rose 14 → 16 over the first ~8k (healthy policy
sharpening) then sat **flat ~16** for the rest of the run — slightly above the
12–15 comfort band but not climbing toward 18; the **peak** actually trended
*down* late (28.5 → 24.25). Unlike `bn1Mean`, this quantity **is** produced by a
weight-decayed parameter — `policy.conv.weight` (the final 76-channel
projection), which grew 13.0 → 14.45 — so raising WD (e.g. → 4e-4/5e-4) is a
valid lever to pull it down into band, at the usual regularization trade-off.

### 5. gamesFed vs positionsFed (a counting clarification, not a bug)

The replay log's `games=` counts every game pulled from the stream **including**
ones the feeder skips (empty / FEN-setup / illegal-on-replay → 0 plies), so over
a full epoch it equals the corpus total; `plies=` (`positionsFed`) counts only
plies actually staged. Empirically the skip rate is negligible — runtime
66.24 ply/game vs the corpus's recorded 66.23 — so the totals are consistent, not
off. (This distinction mattered when building the resume tooling, which counts in
fed-plies, not games.)

## Status / open

The v5 experiment is **concluded.** The base run topped ~1502 @ 44k; the `m0.93`
continuation became the strongest lineage by a clear margin, crossing **1604 @
37k** (its all-time high) with run-best `nll` 2.07–2.10 and back-to-back 1600+
marks (37k/38k), on a plateau that climbed from ~1535 (18–24k) to ~1575–1600
(37–39k). It was **stopped manually at step 39419** (final mark 1583 @ 39k; peak
1604 @ 37k) to free the GPU for a fresh mini-net experiment. Internals stayed
benign to the end — the §3 linear warming leveled off (`bn1Mean` flat ~12.5,
`Σαeff²` ~0.958 inching toward its by-design 1.0 cap, variance-preserving not a
runaway), and `pLogitAbsMax` *fell* over the long run to peak run-lows (~20.75)
with the mean easing to ~14.9 — the lighter 2.5e-4 WD slowly compressing the
logits exactly as §4 predicted over a long enough window.

The WD experiment (§4) is **resolved**: weight decay at 5e-4 / 2.5e-4 is harmless
and is not a strength lever; **LR / momentum is, decisively** — the μ=0.9→0.93
bump (higher effective LR) is the sole change that turned the flat ~1505 wd
plateau into the climbing m0.93 run (peak 1604). The "used-up capacity" findings
from the mid-run weight analysis (vestigial SE, conv2/deep-block rank headroom)
fed the next experiment: a fresh **2-block, 2.25M mini net with SE removed**
(`20260629-3-3MIV`), now training on the same corpus from scratch and tracked
separately (not in this v5-specific doc). Open question it probes: how high a
¼-size net climbs vs this v5's ~1604 — i.e. how capacity-bound the corpus-fit
ceiling actually is.
