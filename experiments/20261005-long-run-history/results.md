## 1. Training settings per segment ([REPLAY-HPARAMS], first line of each surviving log)

### v5
- cum 0: dcm_log_20260627-201127.txt — log missing; settings unknown
- cum 45,441: dcm_log_20260628-114648.txt — log missing; settings unknown
- cum 60,901: dcm_log_20260628-183831.txt — log missing; settings unknown
- cum 100,320: dcm_log_20260702-201756.txt
  - HPARAMS: lr=0.01 batch=4096 wd=0.00025 momentum=0.93 gradClip=30 entropyBonus=0 drawPenalty=0 policyW=1 valueW=1 illegalW=1 pLabelSmooth=0.1 vLabelSmooth=0.013 lrWarmup=500 bufCap=1000000 replayRatio=0.48 minPrefill=500000 complementCE=on sqrtBatchLR=on
  - CYCLE: none logged
  - logged step LRs: lr=0.01 ×5370, lr=0.0002 ×1
- cum 368,826: dcm_log_20260713-195047.txt
  - HPARAMS: lr=0.01 batch=4096 wd=0.00025 momentum=0.93 gradClip=30 entropyBonus=0 drawPenalty=0 policyW=1 valueW=1 illegalW=1 pLabelSmooth=0.1 vLabelSmooth=0.013 lrWarmup=500 bufCap=1000000 replayRatio=0.48 minPrefill=500000 complementCE=on sqrtBatchLR=on
  - CYCLE: none logged
  - logged step LRs: lr=0.01 ×6732, lr=0.0002 ×1
- cum 705,436: dcm_log_20260728-223132.txt
  - HPARAMS: lr=0.01 batch=4096 wd=0.00025 momentum=0.93 gradClip=30 entropyBonus=0 drawPenalty=0 policyW=1 valueW=1 illegalW=1 pLabelSmooth=0.1 vLabelSmooth=0.013 lrWarmup=500 bufCap=1000000 replayRatio=0.48 minPrefill=500000 complementCE=on sqrtBatchLR=on
  - CYCLE: none logged
  - logged step LRs: lr=0.01 ×2126, lr=0.0002 ×1
- cum 811,769: dcm_log_20260802-152830.txt
  - HPARAMS: lr=0.01 batch=4096 wd=0.00025 momentum=0.93 gradClip=30 entropyBonus=0 drawPenalty=0 policyW=1 valueW=1 illegalW=1 pLabelSmooth=0.1 vLabelSmooth=0.013 lrWarmup=500 bufCap=1000000 replayRatio=0.48 minPrefill=500000 complementCE=on sqrtBatchLR=on
  - CYCLE: none logged
  - logged step LRs: lr=0.01 ×978, lr=2e-05 ×1, lr=0.001 ×1
- cum 857,769: dcm_log_20260804-221739.txt
  - HPARAMS: lr=0.01 batch=4096 wd=0.00025 momentum=0.93 gradClip=30 entropyBonus=0 drawPenalty=0 policyW=1 valueW=1 illegalW=1 pLabelSmooth=0.1 vLabelSmooth=0.013 lrWarmup=500 bufCap=1000000 replayRatio=0.48 minPrefill=500000 complementCE=on sqrtBatchLR=on
  - CYCLE: none logged
  - logged step LRs: lr=0.01 ×41, lr=0.0002 ×1

### qeu8
- cum 0: dcm_log_20260702-095124.txt
  - HPARAMS: lr=0.01 batch=4096 wd=0.00025 momentum=0.93 gradClip=30 entropyBonus=0 drawPenalty=0 policyW=1 valueW=1 illegalW=1 pLabelSmooth=0.1 vLabelSmooth=0.013 lrWarmup=500 bufCap=1000000 replayRatio=0.48 minPrefill=500000 complementCE=on sqrtBatchLR=on
  - CYCLE: none logged
  - logged step LRs: lr=0.01 ×819, lr=2e-05 ×1, lr=0.001 ×1
- cum 41,407: dcm_log_20260702-182922.txt
  - HPARAMS: lr=0.01 batch=4096 wd=0.00025 momentum=0.93 gradClip=30 entropyBonus=0 drawPenalty=0 policyW=1 valueW=1 illegalW=1 pLabelSmooth=0.1 vLabelSmooth=0.013 lrWarmup=500 bufCap=1000000 replayRatio=0.48 minPrefill=500000 complementCE=on sqrtBatchLR=on
  - CYCLE: none logged
  - logged step LRs: lr=0.01 ×1341, lr=2e-05 ×1, lr=0.001 ×1
- cum 108,915: dcm_log_20260706-002713.txt
  - HPARAMS: lr=0.01 batch=4096 wd=0.00025 momentum=0.93 gradClip=30 entropyBonus=0 drawPenalty=0 policyW=1 valueW=1 illegalW=1 pLabelSmooth=0.1 vLabelSmooth=0.013 lrWarmup=500 bufCap=1000000 replayRatio=0.48 minPrefill=500000 complementCE=on sqrtBatchLR=on
  - CYCLE: none logged
  - logged step LRs: lr=0.01 ×1336, lr=2e-05 ×1, lr=0.001 ×1
- cum 175,915: dcm_log_20260727-094049.txt
  - HPARAMS: lr=0.01 batch=4096 wd=0.0005 momentum=0.9 gradClip=30 entropyBonus=0 drawPenalty=0 policyW=1 valueW=1 illegalW=1 pLabelSmooth=0.1 vLabelSmooth=0.013 lrWarmup=500 bufCap=1000000 replayRatio=0.48 minPrefill=500000 complementCE=on sqrtBatchLR=on
  - CYCLE: none logged
  - logged step LRs: lr=0.01 ×27943, lr=2e-05 ×1, lr=0.001 ×1

### nt8y
- cum 0: dcm_log_20260701-091259.txt — log missing; settings unknown
- cum 65,883: dcm_log_20260701-152447.txt
  - HPARAMS: lr=0.01 batch=4096 wd=0.00025 momentum=0.93 gradClip=30 entropyBonus=0 drawPenalty=0 policyW=1 valueW=1 illegalW=1 pLabelSmooth=0.1 vLabelSmooth=0.013 lrWarmup=500 bufCap=1000000 replayRatio=0.48 minPrefill=500000 complementCE=on sqrtBatchLR=on
  - CYCLE: none logged
  - logged step LRs: lr=0.01 ×1406, lr=2e-05 ×1, lr=0.001 ×1
- cum 136,662: dcm_log_20260706-125010.txt
  - HPARAMS: lr=0.01 batch=4096 wd=0.00025 momentum=0.93 gradClip=30 entropyBonus=0 drawPenalty=0 policyW=1 valueW=1 illegalW=1 pLabelSmooth=0.1 vLabelSmooth=0.013 lrWarmup=500 bufCap=1000000 replayRatio=0.48 minPrefill=500000 complementCE=on sqrtBatchLR=on
  - CYCLE: none logged
  - logged step LRs: lr=0.01 ×302, lr=2e-05 ×1, lr=0.001 ×1
- cum 151,662: dcm_log_20260706-193601.txt
  - HPARAMS: lr=0.01 batch=4096 wd=0.00025 momentum=0.93 gradClip=30 entropyBonus=0 drawPenalty=0 policyW=1 valueW=1 illegalW=1 pLabelSmooth=0.1 vLabelSmooth=0.013 lrWarmup=500 bufCap=1000000 replayRatio=0.48 minPrefill=500000 complementCE=on sqrtBatchLR=on
  - CYCLE: none logged
  - logged step LRs: lr=0.01 ×1283, lr=2e-05 ×1, lr=0.001 ×1
- cum 291,662: dcm_log_20260707-233038.txt
  - HPARAMS: lr=0.01 batch=4096 wd=0.0005 momentum=0.9 gradClip=30 entropyBonus=0 drawPenalty=0 policyW=1 valueW=1 illegalW=1 pLabelSmooth=0.1 vLabelSmooth=0.013 lrWarmup=500 bufCap=1000000 replayRatio=0.48 minPrefill=500000 complementCE=on sqrtBatchLR=on
  - CYCLE: none logged
  - logged step LRs: lr=0.01 ×412, lr=2e-05 ×1, lr=0.001 ×1

### coxw
- cum 0: dcm_log_20260629-112618.txt — log missing; settings unknown
- cum 55,550: dcm_log_20260709-000611.txt
  - HPARAMS: lr=0.01 batch=4096 wd=0.0005 momentum=0.9 gradClip=30 entropyBonus=0 drawPenalty=0 policyW=1 valueW=1 illegalW=1 pLabelSmooth=0.1 vLabelSmooth=0.013 lrWarmup=500 bufCap=1000000 replayRatio=0.48 minPrefill=500000 complementCE=on sqrtBatchLR=on
  - CYCLE: none logged
  - logged step LRs: lr=0.01 ×5533, lr=2e-05 ×1, lr=0.001 ×1

## 2. Head configurations (registry arch_heads)

- v5: policy intermediate_conv (pre-conv 128) · value WDL(16→FC128)
- qeu8: policy intermediate_conv (pre-conv 512) · value WDL(16→FC64)
- nt8y: policy intermediate_conv (pre-conv 512) · value WDL(16→FC64)
- coxw: policy intermediate_conv (pre-conv 128) · value WDL(16→FC128)
- qeu8b1128: policy intermediate_conv (pre-conv 512) · value WDL(16→FC64)
- se_none: policy intermediate_conv (128) · value WDL(16→FC128)

## 3. Head behaviour by step window (means of the dashboard CSV rows in the window)

| run | window | rows | pElo | nll | vLoss | pLoss | pIllM | gNorm | pLogit_mean | pLogit_peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nt8y | 100–150k | 49 | 1506 | 2.228 | 0.8157 | 2.727 | 0.004776 | 1.29 | 24.3 | 47.14 |
| nt8y | 200–250k | 50 | 1553 | 2.174 | 0.818 | 2.714 | 0.003307 | 0.9797 | 35.95 | 74.88 |
| nt8y | 250–312k | 64 | 1570 | 2.158 | 0.8325 | 2.682 | 0.0029 | 1.356 | 38.06 | 75.68 |
| qeu8 | 100–150k | 51 | 1561 | 2.132 | 0.8112 | 2.701 | 0.003243 | 1.124 | 19.62 | 40.29 |
| qeu8 | 200–250k | 50 | 1561 | 2.147 | 0.9788 | 2.555 | 0.003492 | 3.551 | 27.33 | 49.72 |
| qeu8 | 250–312k | 62 | 1600 | 2.101 | 1.104 | 2.468 | 0.002594 | 3.736 | 29.41 | 50.49 |
| qeu8 | 400–450k | 50 | 1630 | 2.067 | 1.057 | 2.521 | 0.002048 | 2.354 | 34.77 | 59.87 |
| qeu8 | 600–650k | 50 | 1651 | 2.043 | 1.074 | 2.476 | 0.001544 | 2.592 | 47.76 | 73.95 |
| qeu8 | 800–860k | 60 | 1656 | 2.038 | 1.075 | 2.467 | 0.001322 | 2.829 | 69.87 | 94.95 |
| v5 | 100–150k | 49 | 1576 | 2.137 | 0.8499 | 2.669 | 0.003282 | 1.739 | 14.44 | 21.9 |
| v5 | 200–250k | 50 | 1617 | 2.078 | 0.8521 | 2.688 | 0.002812 | 1.121 | 18.69 | 35.82 |
| v5 | 250–312k | 63 | 1495 | 2.241 | 0.9112 | 2.717 | 0.004168 | 1.545 | 25.84 | 49.93 |
| v5 | 400–450k | 50 | 1692 | 1.991 | 1.096 | 2.425 | 0.00143 | 2.573 | 43.07 | 59.45 |
| v5 | 600–650k | 50 | 1720 | 1.945 | 1.107 | 2.428 | 0.00133 | 2.593 | 136.4 | 152.7 |
| v5 | 800–860k | 61 | 1557 | 2.066 | 1.089 | 2.702 | 0.001675 | 4.649 | 324.3 | 357.4 |

## 4. nt8y at matched wall time: nt8y step = R7/R8 step × 1.833 (measured step times; window means of probes)

| R7/R8 step | R7/R8 avg (±w, n) | nt8y step | nt8y (±2k, n) | nt8y − R7/R8 |
|---:|---:|---:|---:|---:|
| 2,000 | 1041.9 (n=3) | 3,666 | 946.8 (n=4) | -95.1 |
| 5,000 | 1260.9 (n=3) | 9,164 | 1087.7 (n=4) | -173.2 |
| 10,000 | 1321.0 (n=5) | 18,328 | 1218.3 (n=4) | -102.7 |
| 15,000 | 1322.4 (n=5) | 27,492 | 1281.6 (n=4) | -40.7 |
| 20,000 | 1329.1 (n=5) | 36,655 | 1340.4 (n=4) | +11.3 |
| 25,000 | 1452.7 (n=5) | 45,819 | 1377.8 (n=4) | -74.9 |
| 30,000 | 1496.6 (n=5) | 54,983 | 1410.5 (n=4) | -86.1 |
| 33,000 | 1500.4 (n=3) | 60,481 | 1411.0 (n=4) | -89.4 |

