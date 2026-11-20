#!/usr/bin/env python3
"""Compare R-elitecyc-fp32fix with R-elitecyc (bf16) at one trainer step, from both runs' [REPLAY] step lines."""
import glob, re, sys
X = "/Users/andrew/cursor/drews-chess-machine/experiments/20261008-zlra-selfplay-lr"
STEP = int(sys.argv[1])
KEYS = ["loss", "pLoss", "vLoss", "pEnt", "playedP", "pW", "pD", "pL", "vAbs", "gNorm", "lr", "ms"]


def seg(path):
    m = re.search(r"seg(\d+)\.stdout$", path)
    return int(m.group(1)) if m else -1


def line_at(pattern):
    found = None
    for path in sorted(glob.glob(f"{X}/{pattern}"), key=seg):  # a later segment's line wins (it is the continuation)
        for line in open(path, errors="replace"):
            if line.startswith("[REPLAY] step=") and re.search(rf"\btrainerStep={STEP}\b", line):
                found = dict(re.findall(r"(\w+)=([-\d.e+]+)", line))
    return found


f = line_at("train-elitecyc-fp32fix-seg*.stdout")
b = line_at("train-elitecyc-seg*.stdout")
if not f:
    sys.exit(f"no fp32fix step line at trainerStep={STEP}")
print(f"{'':6}" + "".join(f"{k:>10}" for k in KEYS))
print(f"{'fp32':6}" + "".join(f"{float(f[k]):>10.4g}" for k in KEYS))
if b:
    print(f"{'bf16':6}" + "".join(f"{float(b[k]):>10.4g}" for k in KEYS))
    print(f"{'diff':6}" + "".join(f"{float(f[k]) - float(b[k]):>+10.3g}" for k in KEYS))
else:
    print("bf16: no step line at this trainer step")
