#!/usr/bin/env python3
"""Lightweight checkpoint freezer — a backstop against monitoring gaps.

DEPRECATED for runs launched with the app's --enumerate-checkpoints flag. The
corpus-replay runner now writes a step-numbered <stem>-replay-step<N>.safetensors
on every save, so preservation is already guaranteed by the app itself and
probe_backfill reads those files directly (see replay.py::enum_glob). Do NOT run
this alongside an --enumerate-checkpoints run — it only duplicates weights under a
second (cum-named) filename and confuses nothing but the disk. Kept only to
backstop legacy runs that do NOT enumerate.

The per-mark tracker (tick.py) only freezes a checkpoint when IT runs, so if
ticking lags the run overwrites its -latest and those checkpoints are lost for
good (pElo becomes unrecoverable). This decouples *preservation* from *probing*:
every 60s it copies the active run's -latest to a step<cum>-frozen file at each
new 1000-mark. No GPU, no probe — just durable copies, so pElo can always be
backfilled later even after a gap. Aligns with the keep-all-frozen policy.

Run detached (legacy non-enumerated runs only):
    nohup /usr/bin/python3 freezer.py >/dev/null 2>&1 &
"""
import time, os, shutil, subprocess, replay

def active_run():
    try:
        ps = subprocess.check_output(["ps", "-Ao", "args="], text=True)
    except Exception:
        return None
    for line in ps.splitlines():
        if "replay-corpus" in line and "--out-model" in line:
            for r, cfg in replay.REG["runs"].items():
                if cfg.get("out_model") and cfg["out_model"] in line:
                    return r
    return None

LOGF = os.path.join(replay.HERE, "freezer.log")
def log(msg):
    with open(LOGF, "a") as f:
        f.write(msg + "\n")

last = {}   # run -> last 1000-mark frozen
log(f"freezer started")
while True:
    try:
        run = active_run()
        if run:
            cfg = replay.REG["runs"][run]
            base = cfg["segments"][-1]["cumstep_base"]
            out = os.path.join(replay.MODELS, cfg["out_model"])
            if os.path.exists(out):
                meta = replay.meta_step_of(out)
                mark = (meta // 1000) * 1000
                if mark > 0 and mark > last.get(run, -1):
                    cum = base + mark
                    dst = os.path.join(replay.MODELS, cfg["frozen_glob"].replace("step*", f"step{cum}"))
                    if not os.path.exists(dst):
                        shutil.copy2(out, dst)
                        log(f"{time.strftime('%Y-%m-%d %H:%M:%S')} froze {run} cum {cum} (meta {mark})")
                    last[run] = mark
    except Exception as e:
        log(f"{time.strftime('%Y-%m-%d %H:%M:%S')} err: {e}")
    time.sleep(60)
