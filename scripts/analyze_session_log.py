#!/usr/bin/env python3
"""Slice a DCM session log to surface UI-responsiveness and training-throughput
trajectories. Designed for one specific question: 'this run got slow over time —
what changed, when?'

Usage: analyze_session_log.py [path]   (defaults to most-recent dcm_log_*.txt)
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from glob import glob
from statistics import median


LOG_DIR = os.path.expanduser("~/Library/Logs/DrewsChessMachine")


def pick_log() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    files = sorted(glob(os.path.join(LOG_DIR, "dcm_log_*.txt")), key=os.path.getmtime)
    if not files:
        sys.exit(f"no logs under {LOG_DIR}")
    return files[-1]


# Timestamps are HH:MM:SS.SSS at line start. Convert to an absolute hour index
# from start-of-log so day boundaries roll cleanly.
TS_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s+(.*)$")


def parse_ts(line: str):
    m = TS_RE.match(line)
    if not m:
        return None, line
    hh, mm, ss, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return hh * 3600 + mm * 60 + ss + ms / 1000.0, m.group(5)


TICK_SLOW_RE = re.compile(r"\[TICK-SLOW\] tickMs=([0-9.]+) mainActorEnqueueWaitMs=([0-9.]+)")
ALARM_RE = re.compile(r"\[ALARM\] (.+)")

# STATS line is one giant key=value soup. Pull out a handful of fields. We
# accept either '+0.6101' or '0.6101' or '12.56908' or '24.0MB' etc.
def kv(line: str, key: str, num_re: str = r"[+-]?\d+\.?\d*"):
    m = re.search(rf"{re.escape(key)}=({num_re})", line)
    return float(m.group(1)) if m else None


def kv_str(line: str, key: str):
    m = re.search(rf"{re.escape(key)}=(\S+)", line)
    return m.group(1) if m else None


def parse_rss_gb(line: str):
    """mem=(rss=20.70GB drss=+24.0MB) — pull rss number in GB."""
    m = re.search(r"mem=\(rss=([\d.]+)GB", line)
    return float(m.group(1)) if m else None


def parse_step(line: str):
    m = re.search(r"\bsteps=(\d+)", line)
    return int(m.group(1)) if m else None


def parse_buffer_unique(line: str):
    m = re.search(r"bufUniq=([\d.]+)", line)
    return float(m.group(1)) if m else None


def parse_legalcost(line: str):
    """[LEGAL-COST] step=N batch=B window=(...)  p1ms=(p50=.. p99=..) ...
       interStepMs=(p50=.. p99=..) gapMs=(p50=..)
    """
    fields = {}
    for key in ("p1ms", "p2ms", "p3ms", "interStepMs"):
        m = re.search(rf"{key}=\(p50=([\d.]+) p99=([\d.]+)\)", line)
        if m:
            fields[key + "_p50"] = float(m.group(1))
            fields[key + "_p99"] = float(m.group(2))
    m = re.search(r"gapMs=\(p50=([\d.+-]+)\)", line)
    if m:
        fields["gapMs_p50"] = float(m.group(1))
    return fields


def fmt_dur(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> None:
    path = pick_log()
    print(f"analyzing: {path}  ({os.path.getsize(path) / 1e6:.1f} MB)\n")

    # Hourly tick latency buckets (UI-thread responsiveness).
    tick_by_bucket = defaultdict(list)
    enqueue_wait_by_bucket = defaultdict(list)

    # STATS time series for trajectories.
    stats_rows = []  # (t_sec_from_start, step, rss_gb, cons, prod, spRate, trainRate, ratio_cur, step_ms, gpu_ms, pEnt, pLoss, vLoss, pD, gNorm)

    # LEGAL-COST time series.
    legalcost_rows = []  # (t_sec_from_start, dict)

    # ALARM counts by message head.
    alarm_counts = defaultdict(int)

    # Hour bucket size, in seconds from start of log.
    BUCKET_SEC = 3600

    t0 = None
    last_t = None
    wrap_offset = 0.0
    prev_raw_t = None

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            t_raw, body = parse_ts(line)
            if t_raw is None:
                continue
            # Day rollover detection: if timestamp jumped backward by more
            # than 12 hours since the last line, add 86400.
            if prev_raw_t is not None and t_raw + 12 * 3600 < prev_raw_t:
                wrap_offset += 86400
            prev_raw_t = t_raw
            t_abs = t_raw + wrap_offset
            if t0 is None:
                t0 = t_abs
            t = t_abs - t0
            last_t = t
            bucket = int(t // BUCKET_SEC)

            if "[TICK-SLOW]" in body:
                m = TICK_SLOW_RE.search(body)
                if m:
                    tick_by_bucket[bucket].append(float(m.group(1)))
                    enqueue_wait_by_bucket[bucket].append(float(m.group(2)))
            elif "[STATS]" in body and "[ARENA]" not in body and "[BATCH-STATS]" not in body:
                step = parse_step(body)
                rss = parse_rss_gb(body)
                # ratio=(target=1.00 cur=0.66 prod=4338.3 prodRaw=7221.7 cons=2860.3 spRate=15617790/hr trainRate=10297239/hr ...)
                cur = re.search(r"ratio=\(target=[\d.]+\s+cur=([\d.]+)", body)
                prod = re.search(r"\bprod=([\d.]+)", body)
                cons = re.search(r"\bcons=([\d.]+)", body)
                sp_rate = re.search(r"\bspRate=(\d+)/hr", body)
                tr_rate = re.search(r"\btrainRate=(\d+)/hr", body)
                # timing=(step=899.0 gpu=699.6 prep=44.68 read=0.05 wait=0.02 n=66)
                step_ms = re.search(r"timing=\(step=([\d.]+)", body)
                gpu_ms = re.search(r"timing=\([^)]*\bgpu=([\d.]+)", body)
                stats_rows.append({
                    "t": t,
                    "step": step,
                    "rss": rss,
                    "ratio_cur": float(cur.group(1)) if cur else None,
                    "prod": float(prod.group(1)) if prod else None,
                    "cons": float(cons.group(1)) if cons else None,
                    "spRate": int(sp_rate.group(1)) if sp_rate else None,
                    "trainRate": int(tr_rate.group(1)) if tr_rate else None,
                    "step_ms": float(step_ms.group(1)) if step_ms else None,
                    "gpu_ms": float(gpu_ms.group(1)) if gpu_ms else None,
                    "pEnt": kv(body, "pEnt"),
                    "pLoss": kv(body, "pLoss"),
                    "vLoss": kv(body, "vLoss"),
                    "pD": kv(body, "pD"),
                    "gNorm": kv(body, "gNorm"),
                    "bufUniq": parse_buffer_unique(body),
                    "trMs": kv(body, "trMs"),
                    "spMs": kv(body, "spMs"),
                    "champion": kv_str(body, "champion"),
                })
            elif "[LEGAL-COST]" in body:
                fields = parse_legalcost(body)
                if fields:
                    fields["t"] = t
                    legalcost_rows.append(fields)
            elif "[ALARM]" in body:
                m = ALARM_RE.search(body)
                if m:
                    # Bucket alarms by their first sentence/phrase.
                    head = m.group(1).split(" step=")[0].split(":")[0].strip()
                    alarm_counts[head] += 1

    total_duration = last_t or 0
    print(f"total runtime: {fmt_dur(total_duration)}  ({len(stats_rows)} STATS, {len(legalcost_rows)} LEGAL-COST)\n")

    # --- UI responsiveness: hourly tick stats --------------------------
    print("=== UI: hourly TICK-SLOW (slow-tick log; threshold ≈ a few ms) ===")
    print(f"{'hour':>5}  {'n':>6}  {'mean ms':>9}  {'p50 ms':>9}  {'p95 ms':>9}  {'max ms':>9}  {'enqWait p50':>11}")
    def p(xs, q):
        if not xs: return float('nan')
        ys = sorted(xs)
        return ys[min(len(ys) - 1, int(q * len(ys)))]
    for b in sorted(tick_by_bucket):
        v = tick_by_bucket[b]
        ew = enqueue_wait_by_bucket[b]
        print(f"{b:>5}  {len(v):>6}  {sum(v)/len(v):>9.1f}  {median(v):>9.1f}  {p(v,0.95):>9.1f}  {max(v):>9.1f}  {median(ew) if ew else 0:>11.2f}")

    # --- Memory + throughput per hour --------------------------------
    print("\n=== Training: hourly STATS samples ===")
    print(f"{'hour':>5}  {'samples':>7}  {'rss GB end':>10}  {'cons/s p50':>10}  {'spRate /hr':>11}  {'step ms p50':>11}  {'pEnt p50':>9}  {'pLoss p50':>10}")
    hourly = defaultdict(list)
    for row in stats_rows:
        hourly[int(row["t"] // BUCKET_SEC)].append(row)
    for b in sorted(hourly):
        rows = hourly[b]
        def col(field):
            xs = [r[field] for r in rows if r[field] is not None]
            return median(xs) if xs else float('nan')
        rss_end = next((r["rss"] for r in reversed(rows) if r["rss"] is not None), float('nan'))
        sp_rate_p50 = col("spRate")
        print(f"{b:>5}  {len(rows):>7}  {rss_end:>10.2f}  {col('cons'):>10.1f}  {sp_rate_p50:>11.0f}  {col('step_ms'):>11.1f}  {col('pEnt'):>9.3f}  {col('pLoss'):>10.3f}")

    # --- Memory trajectory by step (smoothed) ------------------------
    print("\n=== Memory trajectory (RSS GB at ~1-hour intervals + first/last) ===")
    if stats_rows:
        # First, last, plus an hourly sample.
        seen = set()
        samples = []
        last_h = -1
        for row in stats_rows:
            h = int(row["t"] // BUCKET_SEC)
            if row["rss"] is None:
                continue
            if h != last_h:
                samples.append(row)
                last_h = h
        if stats_rows[-1] not in samples:
            samples.append(stats_rows[-1])
        for row in samples:
            print(f"  t={fmt_dur(row['t'])}  step={row['step']:>7}  rss={row['rss']:>6.2f} GB  champion={row['champion']}  bufUniq={row['bufUniq']}")

    # --- LEGAL-COST trajectory --------------------------------------
    if legalcost_rows:
        print("\n=== LEGAL-COST: phase + inter-step wall time over hourly buckets ===")
        print(f"{'hour':>5}  {'samples':>7}  {'p1 p50':>7}  {'p2 p50':>7}  {'p3 p50':>7}  {'inter p50':>10}  {'gap p50':>9}")
        lc_hourly = defaultdict(list)
        for row in legalcost_rows:
            lc_hourly[int(row["t"] // BUCKET_SEC)].append(row)
        for b in sorted(lc_hourly):
            rows = lc_hourly[b]
            def lc_col(k):
                xs = [r.get(k) for r in rows if r.get(k) is not None]
                return median(xs) if xs else float('nan')
            print(f"{b:>5}  {len(rows):>7}  {lc_col('p1ms_p50'):>7.1f}  {lc_col('p2ms_p50'):>7.1f}  {lc_col('p3ms_p50'):>7.1f}  {lc_col('interStepMs_p50'):>10.1f}  {lc_col('gapMs_p50'):>9.1f}")

    # --- ALARMs -----------------------------------------------------
    if alarm_counts:
        print(f"\n=== ALARMs ({sum(alarm_counts.values())} total) ===")
        for head, count in sorted(alarm_counts.items(), key=lambda kv: -kv[1])[:20]:
            print(f"  {count:>6}  {head}")


if __name__ == "__main__":
    main()
