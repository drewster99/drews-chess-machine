#!/usr/bin/env python3
"""session_failure_analysis.py — analyze a DCM session log for failure onset.

Parses one session log, extracts `[STATS]`, `[PARAM]`, `[ALARM]`, and arena
records, and prints a regime-change report focused on "when did this start,
what changed, and did playing strength actually collapse or just the trainer
numerics?".

Usage:
  python3 scripts/session_failure_analysis.py
  python3 scripts/session_failure_analysis.py --log ~/Library/Logs/DrewsChessMachine/dcm_log_20260506-194449.txt
  python3 scripts/session_failure_analysis.py --json
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LOG_GLOB = os.path.expanduser("~/Library/Logs/DrewsChessMachine/dcm_log_*.txt")
NUM_RE = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
TIME_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s+\[(\w[^\]]*)\]\s+(.*)$")
FILE_TS_RE = re.compile(r"dcm_log_(\d{8})-(\d{6})\.txt$")


@dataclass
class StatsRecord:
    line_no: int
    timestamp: str
    elapsed_sec: int
    step: int
    metrics: dict[str, Any]
    raw: str


@dataclass
class ParamChange:
    line_no: int
    timestamp: str
    name: str
    old: str
    new: str
    raw: str


@dataclass
class Alarm:
    line_no: int
    timestamp: str
    message: str


@dataclass
class ArenaRecord:
    line_no: int
    index: int
    timestamp: str
    step: int
    games: int
    wins: int
    draws: int
    losses: int
    score: float
    elo: int
    draw_rate: float
    promoted: bool
    candidate: str
    champion: str
    trainer: str
    duration_sec: float


def latest_log() -> Path | None:
    paths = sorted(glob.glob(LOG_GLOB))
    return Path(paths[-1]) if paths else None


def parse_float(text: str, key: str) -> float | None:
    m = re.search(rf"{re.escape(key)}={NUM_RE}", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_int(text: str, key: str) -> int | None:
    m = re.search(rf"{re.escape(key)}=(\d+)", text)
    return int(m.group(1)) if m else None


def parse_elapsed(text: str) -> int | None:
    m = re.search(r"elapsed=(\d+):(\d+):(\d+)", text)
    if not m:
        return None
    h, m_, s = map(int, m.groups())
    return h * 3600 + m_ * 60 + s


def parse_session_base_date(path: Path) -> dt.date:
    m = FILE_TS_RE.search(path.name)
    if not m:
        return dt.date.today()
    return dt.datetime.strptime(m.group(1), "%Y%m%d").date()


def parse_timestamp(base_date: dt.date, previous: dt.datetime | None, line: str) -> tuple[dt.datetime | None, str | None, str | None]:
    m = TIME_RE.match(line)
    if not m:
        return None, None, None
    hh, mm, ss, ms, tag, message = m.groups()
    current = dt.datetime.combine(
        base_date,
        dt.time(int(hh), int(mm), int(ss), int(ms) * 1000),
    )
    if previous and current < previous and (previous - current) > dt.timedelta(hours=12):
        current += dt.timedelta(days=1)
    return current, tag, message


def parse_stats_metrics(message: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in [
        "steps",
        "spGames",
        "spMoves",
        "avgLen",
        "rollingAvgLen",
        "p50",
        "p95",
        "pLoss",
        "pLossWin",
        "pLossLoss",
        "vLoss",
        "pEnt",
        "gNorm",
        "vNorm",
        "pwNorm",
        "pLogitAbsMax",
        "playedMoveProb",
        "playedMoveProbPosAdv",
        "playedMoveProbNegAdv",
        "legalMass",
        "top1Legal",
        "pEntLegal",
        "vMean",
        "vAbs",
        "vBaseDelta",
        "bufUniq",
        "lr",
    ]:
        value = parse_float(message, key)
        if value is not None:
            out[key] = value

    for key in ["steps", "spGames", "spMoves", "arenaGames"]:
        value = parse_int(message, key)
        if value is not None:
            out[key] = value

    elapsed = parse_elapsed(message)
    if elapsed is not None:
        out["elapsed_sec"] = elapsed

    m = re.search(r"buffer=(\d+)/(\d+)", message)
    if m:
        out["buffer_count"] = int(m.group(1))
        out["buffer_cap"] = int(m.group(2))

    m = re.search(r"ratio=\(target=" + NUM_RE + r" cur=" + NUM_RE, message)
    if m:
        out["ratio_target"] = float(m.group(1))
        out["ratio_cur"] = float(m.group(2))

    m = re.search(r"spRate=(\d+)/hr", message)
    if m:
        out["spRate_per_hr"] = int(m.group(1))
    m = re.search(r"trainRate=(\d+)/hr", message)
    if m:
        out["trainRate_per_hr"] = int(m.group(1))
    m = re.search(r"spDelay=(\d+)ms", message)
    if m:
        out["spDelay_ms"] = int(m.group(1))
    m = re.search(r"workers=(\d+)\)", message)
    if m:
        out["ratio_workers"] = int(m.group(1))

    reg = re.search(
        r"reg=\(clip=" + NUM_RE
        + r" decay=" + NUM_RE
        + r" ent=" + NUM_RE
        + r" drawPen=" + NUM_RE
        + r" pLossW=" + NUM_RE
        + r" vLossW=" + NUM_RE
        + r" μ=" + NUM_RE
        + r"\)",
        message,
    )
    if reg:
        (
            out["clip"],
            out["decay"],
            out["entropy_coeff"],
            out["draw_penalty"],
            out["policy_loss_weight"],
            out["value_loss_weight"],
            out["momentum"],
        ) = map(float, reg.groups())

    adv = re.search(
        r"adv=\(mean=" + NUM_RE
        + r" std=" + NUM_RE
        + r" min=" + NUM_RE
        + r" max=" + NUM_RE
        + r" frac\+=" + NUM_RE
        + r" fracSmall=" + NUM_RE
        + r" p05=" + NUM_RE
        + r" p50=" + NUM_RE
        + r" p95=" + NUM_RE
        + r"\)",
        message,
    )
    if adv:
        (
            out["adv_mean"],
            out["adv_std"],
            out["adv_min"],
            out["adv_max"],
            out["adv_frac_pos"],
            out["adv_frac_small"],
            out["adv_p05"],
            out["adv_p50"],
            out["adv_p95"],
        ) = map(float, adv.groups())

    for key in ["trainer", "champion"]:
        m = re.search(rf"{key}=(\S+)", message)
        if m:
            out[key] = m.group(1)

    return out


def parse_log(path: Path) -> dict[str, Any]:
    base_date = parse_session_base_date(path)
    current_date = base_date
    previous_dt: dt.datetime | None = None
    stats: list[StatsRecord] = []
    params: list[ParamChange] = []
    alarms: list[Alarm] = []
    arenas: list[ArenaRecord] = []
    app_launch: str | None = None
    session_log_path: str | None = None

    arena_kv_re = re.compile(
        r"#(?P<index>\d+) kv step=(?P<step>\d+) games=(?P<games>\d+) "
        r"w=(?P<wins>\d+) d=(?P<draws>\d+) l=(?P<losses>\d+) "
        r"score=(?P<score>[+-]?\d+(?:\.\d+)?) "
        r"elo=(?P<elo>[+-]?\d+) .* "
        r"draw_rate=(?P<draw_rate>[+-]?\d+(?:\.\d+)?) .* "
        r"promoted=(?P<promoted>\d) kind=\w+ dur_sec=(?P<dur>[+-]?\d+(?:\.\d+)?) .* "
        r"candidate=(?P<candidate>\S+) champion=(?P<champion>\S+) trainer=(?P<trainer>\S+)"
    )

    with path.open("r", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            ts, tag, message = parse_timestamp(current_date, previous_dt, line)
            if ts is None or tag is None or message is None:
                continue
            if previous_dt and ts.date() != previous_dt.date():
                current_date = ts.date()
            previous_dt = ts
            ts_s = ts.isoformat(sep=" ")

            if tag == "APP" and message.startswith("launched "):
                app_launch = message
            elif tag == "APP" and message.startswith("session log: "):
                session_log_path = message.split(": ", 1)[1]

            if tag == "STATS" and "elapsed=" in message:
                metrics = parse_stats_metrics(message)
                step = int(metrics.get("steps", -1))
                elapsed_sec = int(metrics.get("elapsed_sec", -1))
                stats.append(
                    StatsRecord(
                        line_no=line_no,
                        timestamp=ts_s,
                        elapsed_sec=elapsed_sec,
                        step=step,
                        metrics=metrics,
                        raw=message,
                    )
                )
            elif tag == "PARAM":
                m = re.match(r"([^:]+):\s+(.+?)\s+->\s+(.+)$", message)
                if m:
                    params.append(
                        ParamChange(
                            line_no=line_no,
                            timestamp=ts_s,
                            name=m.group(1),
                            old=m.group(2),
                            new=m.group(3),
                            raw=message,
                        )
                    )
            elif tag == "ALARM":
                alarms.append(Alarm(line_no=line_no, timestamp=ts_s, message=message))
            elif tag == "ARENA" and " kv " in message:
                m = arena_kv_re.search(message)
                if m:
                    arenas.append(
                        ArenaRecord(
                            line_no=line_no,
                            index=int(m.group("index")),
                            timestamp=ts_s,
                            step=int(m.group("step")),
                            games=int(m.group("games")),
                            wins=int(m.group("wins")),
                            draws=int(m.group("draws")),
                            losses=int(m.group("losses")),
                            score=float(m.group("score")),
                            elo=int(m.group("elo")),
                            draw_rate=float(m.group("draw_rate")),
                            promoted=(m.group("promoted") == "1"),
                            duration_sec=float(m.group("dur")),
                            candidate=m.group("candidate"),
                            champion=m.group("champion"),
                            trainer=m.group("trainer"),
                        )
                    )

    return {
        "path": str(path),
        "app_launch": app_launch,
        "session_log_path": session_log_path,
        "stats": stats,
        "params": params,
        "alarms": alarms,
        "arenas": arenas,
    }


def first_crossing(stats: list[StatsRecord], key: str, predicate) -> dict[str, Any] | None:
    for row in stats:
        value = row.metrics.get(key)
        if value is None:
            continue
        if predicate(value):
            return {
                "timestamp": row.timestamp,
                "step": row.step,
                "value": value,
                "line_no": row.line_no,
            }
    return None


def summarize_window(stats: list[StatsRecord], start_idx: int, end_idx: int) -> dict[str, Any]:
    rows = stats[start_idx:end_idx]
    if not rows:
        return {}
    keys = [
        "pEnt",
        "gNorm",
        "pLogitAbsMax",
        "legalMass",
        "pEntLegal",
        "pLoss",
        "pLossLoss",
        "vLoss",
        "vAbs",
        "adv_mean",
        "adv_frac_pos",
        "spDelay_ms",
    ]
    out: dict[str, Any] = {
        "from": rows[0].timestamp,
        "to": rows[-1].timestamp,
        "step_from": rows[0].step,
        "step_to": rows[-1].step,
        "samples": len(rows),
    }
    for key in keys:
        vals = [float(r.metrics[key]) for r in rows if key in r.metrics and isinstance(r.metrics[key], (int, float))]
        if vals:
            out[f"{key}_start"] = float(rows[0].metrics.get(key, math.nan))
            out[f"{key}_end"] = float(rows[-1].metrics.get(key, math.nan))
            out[f"{key}_min"] = min(vals)
            out[f"{key}_max"] = max(vals)
            out[f"{key}_mean"] = statistics.fmean(vals)
    return out


def render_summary(report: dict[str, Any]) -> str:
    stats: list[StatsRecord] = report["stats"]
    params: list[ParamChange] = report["params"]
    alarms: list[Alarm] = report["alarms"]
    arenas: list[ArenaRecord] = report["arenas"]
    if not stats:
        return "no [STATS] lines found"

    first = stats[0]
    last = stats[-1]
    first_entropy_alarm = next((a for a in alarms if a.message.startswith("policy entropy ")), None)
    alarm_idx = None
    if first_entropy_alarm:
        for idx, row in enumerate(stats):
            if row.timestamp >= first_entropy_alarm.timestamp:
                alarm_idx = idx
                break
    pre_alarm = summarize_window(stats, 0, alarm_idx or len(stats))
    post_alarm = summarize_window(stats, alarm_idx or len(stats), len(stats)) if alarm_idx is not None else {}

    crossings = {
        "pEnt<1.0": first_crossing(stats, "pEnt", lambda v: v < 1.0),
        "pEnt<0.8": first_crossing(stats, "pEnt", lambda v: v < 0.8),
        "gNorm>30": first_crossing(stats, "gNorm", lambda v: v > 30),
        "gNorm>100": first_crossing(stats, "gNorm", lambda v: v > 100),
        "gNorm>1000": first_crossing(stats, "gNorm", lambda v: v > 1000),
        "pLogitAbsMax>25": first_crossing(stats, "pLogitAbsMax", lambda v: v > 25),
        "pLogitAbsMax>50": first_crossing(stats, "pLogitAbsMax", lambda v: v > 50),
        "pLogitAbsMax>100": first_crossing(stats, "pLogitAbsMax", lambda v: v > 100),
        "pLogitAbsMax>1000": first_crossing(stats, "pLogitAbsMax", lambda v: v > 1000),
        "spDelay>500ms": first_crossing(stats, "spDelay_ms", lambda v: v > 500),
        "spDelay>1000ms": first_crossing(stats, "spDelay_ms", lambda v: v > 1000),
    }

    arena_scores = [a.score for a in arenas]
    draw_rates = [a.draw_rate for a in arenas]
    lines: list[str] = []
    lines.append(f"log: {report['path']}")
    if report.get("app_launch"):
        lines.append(f"launch: {report['app_launch']}")
    lines.append(
        f"stats: {len(stats)} samples  steps {first.step}->{last.step}  "
        f"time {first.timestamp}->{last.timestamp}"
    )
    lines.append(
        "start: "
        f"pEnt={first.metrics.get('pEnt')} gNorm={first.metrics.get('gNorm')} "
        f"pLogitAbsMax={first.metrics.get('pLogitAbsMax')} legalMass={first.metrics.get('legalMass')} "
        f"vLossW={first.metrics.get('value_loss_weight')} decay={first.metrics.get('decay')}"
    )
    lines.append(
        "end:   "
        f"pEnt={last.metrics.get('pEnt')} gNorm={last.metrics.get('gNorm')} "
        f"pLogitAbsMax={last.metrics.get('pLogitAbsMax')} legalMass={last.metrics.get('legalMass')} "
        f"vLossW={last.metrics.get('value_loss_weight')} decay={last.metrics.get('decay')}"
    )

    lines.append("")
    lines.append("threshold crossings:")
    for label, hit in crossings.items():
        if hit:
            lines.append(
                f"  {label:<17} first at {hit['timestamp']}  step={hit['step']}  value={hit['value']}"
            )
        else:
            lines.append(f"  {label:<17} never")

    lines.append("")
    lines.append("parameter changes:")
    if params:
        for p in params:
            lines.append(f"  {p.timestamp}  {p.name}: {p.old} -> {p.new}")
    else:
        lines.append("  none")

    lines.append("")
    lines.append("alarms:")
    interesting = [a for a in alarms if "legal-mass probe ok" not in a.message]
    if interesting:
        for a in interesting[:12]:
            lines.append(f"  {a.timestamp}  {a.message}")
        if len(interesting) > 12:
            lines.append(f"  ... {len(interesting) - 12} more")
    else:
        lines.append("  none")

    lines.append("")
    lines.append("phase summary:")
    lines.append(
        "  pre-alarm: "
        f"samples={pre_alarm.get('samples', 0)} "
        f"pEnt {pre_alarm.get('pEnt_start')} -> {pre_alarm.get('pEnt_end')} "
        f"gNorm {pre_alarm.get('gNorm_start')} -> {pre_alarm.get('gNorm_end')} "
        f"logit {pre_alarm.get('pLogitAbsMax_start')} -> {pre_alarm.get('pLogitAbsMax_end')} "
        f"spDelay {pre_alarm.get('spDelay_ms_start')} -> {pre_alarm.get('spDelay_ms_end')}"
    )
    if post_alarm:
        lines.append(
            "  post-alarm: "
            f"samples={post_alarm.get('samples', 0)} "
            f"pEnt {post_alarm.get('pEnt_start')} -> {post_alarm.get('pEnt_end')} "
            f"gNorm {post_alarm.get('gNorm_start')} -> {post_alarm.get('gNorm_end')} "
            f"logit {post_alarm.get('pLogitAbsMax_start')} -> {post_alarm.get('pLogitAbsMax_end')} "
            f"spDelay {post_alarm.get('spDelay_ms_start')} -> {post_alarm.get('spDelay_ms_end')}"
        )

    lines.append("")
    lines.append("arenas:")
    if arenas:
        lines.append(
            f"  count={len(arenas)} score_mean={statistics.fmean(arena_scores):.4f} "
            f"score_min={min(arena_scores):.4f} score_max={max(arena_scores):.4f} "
            f"draw_rate_mean={statistics.fmean(draw_rates):.4f} promotions={sum(1 for a in arenas if a.promoted)}"
        )
        for a in arenas[-8:]:
            lines.append(
                f"  #{a.index:<2} {a.timestamp} step={a.step:<5} score={a.score:.4f} "
                f"w/d/l={a.wins}/{a.draws}/{a.losses} draw={a.draw_rate:.4f} "
                f"elo={a.elo:+d} promoted={int(a.promoted)}"
            )
    else:
        lines.append("  none")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", help="session log path")
    parser.add_argument("--json", action="store_true", help="emit structured JSON")
    args = parser.parse_args()

    log_path = Path(args.log).expanduser() if args.log else latest_log()
    if log_path is None or not log_path.exists():
        print("no log found", file=sys.stderr)
        return 1

    report = parse_log(log_path)
    if args.json:
        payload = {
            "path": report["path"],
            "app_launch": report["app_launch"],
            "session_log_path": report["session_log_path"],
            "stats": [asdict(x) for x in report["stats"]],
            "params": [asdict(x) for x in report["params"]],
            "alarms": [asdict(x) for x in report["alarms"]],
            "arenas": [asdict(x) for x in report["arenas"]],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render_summary(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
