#!/usr/bin/env python3
"""tick_status.py — autotrain monitoring report for a single cron tick.

Pulls the latest [STATS] line from the most recent dcm session log, parses
the full set of metrics we track, compares them against the previous tick
(stored in scripts/.tick_history.jsonl), pulls arena trajectory, and prints
a single fixed-format block so successive ticks are visually comparable.

Designed to be safe to re-run: writes one line to .tick_history.jsonl per
invocation (only if the [STATS] line advanced), so the file is the rolling
trend source. Caps to 500 entries.

Exit code:
  0 — healthy / watch
  1 — hard-reject criterion has tripped (caller may want to SIGUSR1 after
      ≥2 consecutive ticks of the same trip)
  2 — no STATS line yet (app may have just started; check next tick)
  3 — app process not running

Usage:
  python3 scripts/tick_status.py [--log <path>] [--no-record]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
HIST_PATH = Path(__file__).resolve().parent / '.tick_history.jsonl'
HIST_CAP = 500
LOG_GLOB = os.path.expanduser('~/Library/Logs/DrewsChessMachine/dcm_log_*.txt')

# Health-band tables. Tuned for post-mask (legal-only) pEnt per
# commit 2f95f21; numeric metrics match in-repo thresholds.
BANDS = {
    'pEnt':         {'in': (1.5, 3.5), 'watch': (1.0, 1.5),  'fmt': '{:.2f}'},
    'gNorm':        {'in': (0, 60),    'watch': (60, 100),   'fmt': '{:.1f}'},
    'vAbs':         {'in': (0.05, 0.50), 'watch': (0.50, 0.85), 'fmt': '{:.3f}'},
    'pD':           {'in': (0.0, 0.60), 'watch': (0.60, 0.85), 'fmt': '{:.3f}'},
    'pIllM':        {'in': (0.0, 0.10), 'watch': (0.10, 0.30), 'fmt': '{:.4f}'},
    # avgLen is a CONFIG-dependent quantity, not a health signal on its own: it is
    # bounded above by self_play_max_plies_per_game (450 today) and self-play games
    # in this regime settle around 85 plies. The old (200,500) band was tuned when
    # games ran far longer and flagged every healthy tick as OUT.
    'avgLen':       {'in': (40, 450),  'watch': (20, 700),   'fmt': '{:.0f}'},
    # pLogitAbsMax deliberately has NO fixed band — see pLogit_label(). A net forked
    # from a corpus-distilled seed INHERITS a large logit scale (this lineage started
    # at 169 on step 10 and has drifted DOWN to ~159 over 938k steps), so any absolute
    # threshold either screams on every tick or is too loose to catch a real blow-up.
    # What matters is growth relative to where this run started.
}

# pLogitAbsMax growth multipliers, measured against the FIRST [STATS] line of the
# same log (exact, needs no tick history). Absolute backstop catches a true runaway
# even if the run somehow started high.
PLOGIT_WATCH_RATIO = 2.0
PLOGIT_TRIP_RATIO = 3.0
PLOGIT_ABS_BACKSTOP = 500.0


@dataclass
class Tick:
    ts: str
    elapsed_sec: int
    log_path: str
    metrics: dict[str, Any]
    arenas: list[dict[str, Any]]
    checkpoints: list[dict[str, Any]]
    alarms: list[str]


def latest_log(override: str | None) -> Path | None:
    if override:
        return Path(override)
    paths = sorted(glob.glob(LOG_GLOB))
    return Path(paths[-1]) if paths else None


def find_last_stats(path: Path) -> str | None:
    last = None
    with open(path, 'r', errors='replace') as f:
        for line in f:
            if '[STATS] elapsed=' in line:
                last = line.rstrip('\n')
    return last


_NUM = r'([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)'


def parse_stats(line: str) -> dict[str, Any]:
    m = re.search(r'elapsed=(\d+):(\d+):(\d+)', line)
    elapsed = (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))) if m else 0
    out: dict[str, Any] = {'elapsed_sec': elapsed}
    for key in [
        'steps', 'spGames', 'spMoves', 'avgLen', 'pEnt', 'pEntLegal', 'gNorm',
        'pLogitAbsMax', 'pwNorm', 'legalMass', 'top1Legal', 'pLoss',
        'pLossWin', 'pLossLoss', 'vLoss', 'vMean', 'vAbs', 'playedMoveProb',
        'buffer', 'pW', 'pD', 'pL', 'pIllM',
    ]:
        m = re.search(rf'{re.escape(key)}=' + _NUM, line)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    # buffer=N/M shape
    m = re.search(r'buffer=(\d+)/(\d+)', line)
    if m:
        out['buffer_count'] = int(m.group(1))
        out['buffer_cap'] = int(m.group(2))
    # ratio=(target=X cur=Y prod=Z cons=W ...)
    m = re.search(r'ratio=\(target=' + _NUM + r' cur=' + _NUM, line)
    if m:
        out['ratio_target'] = float(m.group(1))
        out['ratio_cur'] = float(m.group(2))
    m = re.search(r'spRate=(\d+)/hr', line)
    if m:
        out['spRate_per_hr'] = int(m.group(1))
    m = re.search(r'trainRate=(\d+)/hr', line)
    if m:
        out['trainRate_per_hr'] = int(m.group(1))
    # diversity=unique=A/B(C%) diverge=D
    m = re.search(r'diversity=unique=(\d+)/(\d+)\(\d+%\)\s+diverge=' + _NUM, line)
    if m:
        out['diversity_unique'] = int(m.group(1))
        out['diversity_window'] = int(m.group(2))
        out['diversity_diverge'] = float(m.group(3))
    # Trainer / champion ids
    m = re.search(r'trainer=(\S+)', line)
    if m:
        out['trainer'] = m.group(1)
    m = re.search(r'champion=(\S+)', line)
    if m:
        out['champion'] = m.group(1)
    return out


def collect_arenas(path: Path) -> list[dict[str, Any]]:
    arenas: list[dict[str, Any]] = []
    line_re = re.compile(
        r'\[ARENA\] #(\d+) kv .*step=(\d+) games=(\d+) w=(\d+) d=(\d+) l=(\d+) '
        r'score=(\d+\.\d+) elo=([+-]?\d+) elo_lo=([+-]?\d+) elo_hi=([+-]?\d+).*'
        r'promoted=(\d+) kind=(\w+) dur_sec=([\d.]+).*candidate=(\S+) champion=(\S+) trainer=(\S+)'
    )
    with open(path, 'r', errors='replace') as f:
        for line in f:
            if '[ARENA]' not in line:
                continue
            m = line_re.search(line)
            if not m:
                continue
            arenas.append({
                'index': int(m.group(1)),
                'step': int(m.group(2)),
                'games': int(m.group(3)),
                'w': int(m.group(4)),
                'd': int(m.group(5)),
                'l': int(m.group(6)),
                'score': float(m.group(7)),
                'elo': int(m.group(8)),
                'elo_ci': (int(m.group(9)), int(m.group(10))),
                'promoted': int(m.group(11)) == 1,
                'kind': m.group(12),
                'dur_sec': float(m.group(13)),
                'candidate': m.group(14),
                'champion': m.group(15),
                'trainer': m.group(16),
            })
    return arenas


def collect_checkpoints(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pat = re.compile(r'\[CHECKPOINT\] Saved (\S+)\s+\((\S+)\):\s+(.*)$')
    with open(path, 'r', errors='replace') as f:
        for line in f:
            if '[CHECKPOINT]' not in line:
                continue
            m = pat.search(line)
            if m:
                out.append({'kind': m.group(1), 'trigger': m.group(2), 'detail': m.group(3).strip()})
    return out


def collect_alarms(path: Path) -> list[str]:
    out: list[str] = []
    with open(path, 'r', errors='replace') as f:
        for line in f:
            if '[ALARM]' in line and 'legal-mass probe ok' not in line:
                out.append(line.rstrip('\n'))
    return out


def first_stats_pLogit(path: Path) -> float | None:
    """pLogitAbsMax from the FIRST [STATS] line of this log — the scale the run
    started at. A fork of a distilled net inherits its teacher's logit scale, so
    "is the policy blowing up?" is only answerable relative to this, never against
    a fixed constant. Returns None when the log has no parseable [STATS] line."""
    with open(path, 'r', errors='replace') as f:
        for line in f:
            if '[STATS]' not in line:
                continue
            m = re.search(r'pLogitAbsMax=([\d.]+)', line)
            if m:
                return float(m.group(1))
    return None


def pLogit_label(value: float | None, baseline: float | None) -> str:
    """Band pLogitAbsMax by growth over this run's own starting scale."""
    if value is None:
        return ''
    if baseline is None or baseline <= 0:
        return 'no-base'
    r = value / baseline
    if value > PLOGIT_ABS_BACKSTOP or r > PLOGIT_TRIP_RATIO:
        return 'OUT'
    if r > PLOGIT_WATCH_RATIO:
        return 'watch'
    return 'in-band'


def _plogit_note(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None or baseline <= 0:
        return 'no starting scale readable from this log — criterion not evaluated'
    return (f'{value / baseline:.2f}x this run\'s start ({baseline:.1f}); '
            f'>{PLOGIT_TRIP_RATIO:.0f}x = hard kill, >{PLOGIT_WATCH_RATIO:.0f}x = watch')


def label_band(value: float | None, key: str) -> str:
    if value is None or key not in BANDS:
        return ''
    band = BANDS[key]
    in_lo, in_hi = band['in']
    w_lo, w_hi = band['watch']
    if in_lo <= value <= in_hi:
        return 'in-band'
    if w_lo <= value <= w_hi:
        return 'watch'
    return 'OUT'


def hard_rejects(m: dict[str, Any], pLogit_base: float | None = None) -> list[str]:
    issues: list[str] = []
    e = m.get('elapsed_sec', 0)
    cur = m.get('pLogitAbsMax')
    if cur is not None:
        if pLogit_base is None or pLogit_base <= 0:
            # Do not silently pass: say that the criterion could not be evaluated.
            issues.append(
                f'H3 NOT EVALUATED — pLogitAbsMax={cur:.1f} but this log has no '
                f'readable starting scale to compare against')
        elif cur > PLOGIT_ABS_BACKSTOP:
            issues.append(f'H3 pLogitAbsMax={cur:.1f} > {PLOGIT_ABS_BACKSTOP:.0f} (absolute backstop)')
        elif cur / pLogit_base > PLOGIT_TRIP_RATIO:
            issues.append(
                f'H3 pLogitAbsMax={cur:.1f} = {cur / pLogit_base:.1f}x this run\'s '
                f'starting scale {pLogit_base:.1f} (> {PLOGIT_TRIP_RATIO:.0f}x)')
    if e > 3600 and m.get('pEnt', 99) < 1.0:
        issues.append(f'H2 pEnt={m["pEnt"]:.2f} < 1.0 at {e}s')
    if (e > 3600 and m.get('legalMass', 1) < 0.005
            and m.get('top1Legal', 1) == 0):
        issues.append(f'H4 legalMass={m["legalMass"]:.4f} top1Legal=0 at {e}s')
    if m.get('gNorm', 0) > 300:
        issues.append(f'H6 gNorm={m["gNorm"]:.1f} > 300 (single tick)')
    return issues


def fmt_elapsed(sec: int) -> str:
    return f'{sec // 3600}h{(sec % 3600) // 60:02d}m{sec % 60:02d}s'


def fmt_delta(cur: float | None, prev: float | None, fmt: str = '{:+.3f}') -> str:
    if cur is None or prev is None:
        return '   ----'
    d = cur - prev
    return fmt.format(d)


def load_history() -> list[dict[str, Any]]:
    if not HIST_PATH.exists():
        return []
    out = []
    with open(HIST_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def append_history(entry: dict[str, Any]) -> None:
    HIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    hist = load_history()
    # Skip if elapsed_sec hasn't advanced (same STATS line, idempotent re-run)
    if hist and hist[-1].get('metrics', {}).get('elapsed_sec') == entry['metrics'].get('elapsed_sec'):
        return
    hist.append(entry)
    if len(hist) > HIST_CAP:
        hist = hist[-HIST_CAP:]
    with open(HIST_PATH, 'w') as f:
        for h in hist:
            f.write(json.dumps(h) + '\n')


def app_running() -> tuple[bool, int | None]:
    try:
        out = subprocess.run(['pgrep', '-x', 'DrewsChessMachine'], capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return True, int(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return False, None


def render(t: Tick, prev: dict[str, Any] | None, pid: int | None,
           pLogit_base: float | None = None) -> tuple[str, int]:
    m = t.metrics
    pm = (prev or {}).get('metrics', {}) if prev else {}
    issues = hard_rejects(m, pLogit_base)

    # Header
    lines: list[str] = []
    lines.append('=' * 88)
    lines.append(
        f'autotrain tick  elapsed={fmt_elapsed(m.get("elapsed_sec", 0))}  '
        f'steps={int(m.get("steps", 0)):>5}  '
        f'pid={pid if pid else "?"}  '
        f'log={Path(t.log_path).name}'
    )
    if prev:
        prev_e = pm.get('elapsed_sec', 0)
        cur_e = m.get('elapsed_sec', 0)
        dt = max(1, cur_e - prev_e)
        d_steps = int(m.get('steps', 0)) - int(pm.get('steps', 0))
        lines.append(
            f'since prev tick: Δt={dt}s  Δsteps={d_steps:+d}  '
            f'Δsteps/min={(d_steps * 60 / dt):+.1f}'
        )
    lines.append('-' * 88)

    # Health table — six core + ancillary, with trend deltas
    lines.append(f'{"metric":<14}{"value":>12}  {"label":<8}  {"Δ since prev":>14}  notes')
    lines.append('-' * 88)
    rows = [
        ('pEnt',         m.get('pEnt'),         '{:+.3f}', 'post-mask legal-only entropy; ceiling≈ln(legal)≈3.4'),
        ('pEntLegal',    m.get('pEntLegal'),    '{:+.3f}', 'redundant w/ pEnt post-mask'),
        ('gNorm',        m.get('gNorm'),        '{:+.2f}', 'pre-clip gradient L2 norm'),
        ('pLogitAbsMax', m.get('pLogitAbsMax'), '{:+.2f}', _plogit_note(m.get('pLogitAbsMax'), pLogit_base)),
        ('vAbs',         m.get('vAbs'),         '{:+.3f}', 'mean |p_win - p_loss| (no tanh); 0 = indecisive'),
        ('vMean',        m.get('vMean'),        '{:+.3f}', 'should sit near 0 (chess avg ≈ draw)'),
        ('pW',           m.get('pW'),           '{:+.3f}', 'W/D/L softmax batch-mean: win'),
        ('pD',           m.get('pD'),           '{:+.3f}', 'draw; collapse signature is pD -> 1, not merely high'),
        ('pL',           m.get('pL'),           '{:+.3f}', 'loss'),
        ('pIllM',        m.get('pIllM'),        '{:+.4f}', 'illegal-move probability mass; should fall well below 0.1'),
        ('legalMass',    m.get('legalMass'),    '{:+.4f}', 'softmax mass on legal moves (replay sample, n=128)'),
        ('top1Legal',    m.get('top1Legal'),    '{:+.3f}', 'fraction of positions where argmax is legal'),
        ('pLoss',        m.get('pLoss'),        '{:+.4f}', 'outcome-weighted policy CE'),
        ('vLoss',        m.get('vLoss'),        '{:+.4f}', 'W/D/L categorical CE, scale ~[0, ln 3]'),
        ('avgLen',       m.get('avgLen'),       '{:+.1f}', 'avg ply per game'),
    ]
    for key, val, dfmt, note in rows:
        if val is None:
            continue
        band_key = key
        if band_key == 'pLogitAbsMax':
            label = pLogit_label(val, pLogit_base)
        else:
            label = label_band(val, band_key) if band_key in BANDS else ''
        fmt = BANDS.get(band_key, {}).get('fmt', '{:.4f}')
        delta = fmt_delta(val, pm.get(key), dfmt)
        lines.append(f'{key:<14}{fmt.format(val):>12}  {label:<8}  {delta:>14}  {note}')

    # Replay/throughput
    lines.append('-' * 88)
    rt = m.get('ratio_target')
    rc = m.get('ratio_cur')
    sp = m.get('spRate_per_hr')
    tr = m.get('trainRate_per_hr')
    if rt is not None and rc is not None:
        lines.append(f'replay ratio  cur={rc:.2f} target={rt:.2f}  '
                     f'sp_rate={sp:>10,}/hr  train_rate={tr:>10,}/hr')
    if 'buffer_count' in m and 'buffer_cap' in m:
        bc, bp = m['buffer_count'], m['buffer_cap']
        pct = 100.0 * bc / max(1, bp)
        lines.append(f'replay buffer {bc:>10,}/{bp:<10,} ({pct:5.1f}% full)')
    if 'diversity_unique' in m:
        lines.append(f'diversity     unique={m["diversity_unique"]}/{m["diversity_window"]} '
                     f'diverge={m["diversity_diverge"]:.1f}')
    if 'trainer' in m:
        lines.append(f'IDs           trainer={m["trainer"]}  champion={m.get("champion","?")}')

    # Arena trajectory
    lines.append('-' * 88)
    if t.arenas:
        lines.append(f'arenas ({len(t.arenas)}):')
        for a in t.arenas[-12:]:
            tag = 'PROMO' if a['promoted'] else 'kept '
            lines.append(
                f'  #{a["index"]:>2}  step={a["step"]:>5}  score={a["score"]:.3f} '
                f'(W{a["w"]}/D{a["d"]}/L{a["l"]}) '
                f'elo={a["elo"]:+d}[{a["elo_ci"][0]:+d},{a["elo_ci"][1]:+d}]  {tag}'
            )
    else:
        lines.append('arenas: none yet')

    # Checkpoints
    if t.checkpoints:
        lines.append('-' * 88)
        lines.append(f'checkpoints ({len(t.checkpoints)}):')
        for c in t.checkpoints[-5:]:
            lines.append(f'  {c["kind"]} ({c["trigger"]}): {c["detail"]}')

    # Alarms
    if t.alarms:
        lines.append('-' * 88)
        lines.append(f'alarms ({len(t.alarms)}):')
        for a in t.alarms[-5:]:
            lines.append(f'  {a}')

    # Hard-reject status
    lines.append('-' * 88)
    if issues:
        lines.append('HARD-REJECT TRIPPED: ' + '; '.join(issues))
        lines.append('  Action: confirm on next tick — if persistent ≥2 ticks, kill -SIGUSR1 <pid>')
        rc_exit = 1
    else:
        lines.append('hard-reject: clear')
        rc_exit = 0
    lines.append('=' * 88)
    return '\n'.join(lines), rc_exit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--log', default=None, help='override session log path')
    parser.add_argument('--no-record', action='store_true', help='do not append to .tick_history.jsonl')
    args = parser.parse_args()

    running, pid = app_running()
    log_path = latest_log(args.log)
    if not log_path or not log_path.exists():
        print(f'NO LOG FOUND under {LOG_GLOB}', file=sys.stderr)
        return 2
    line = find_last_stats(log_path)
    if not line:
        print(f'NO STATS YET in {log_path.name} (app may have just started)', file=sys.stderr)
        return 2
    metrics = parse_stats(line)
    arenas = collect_arenas(log_path)
    checkpoints = collect_checkpoints(log_path)
    alarms = collect_alarms(log_path)

    import datetime as _dt
    now = _dt.datetime.now().isoformat(timespec='seconds')
    tick = Tick(ts=now, elapsed_sec=metrics.get('elapsed_sec', 0), log_path=str(log_path),
                metrics=metrics, arenas=arenas, checkpoints=checkpoints, alarms=alarms)

    hist = load_history()
    prev = hist[-1] if hist else None
    pLogit_base = first_stats_pLogit(log_path)
    out_text, rc = render(tick, prev, pid, pLogit_base)
    print(out_text)

    if not running:
        print('NOTE: DrewsChessMachine process is NOT currently running', file=sys.stderr)
        rc = max(rc, 3)

    if not args.no_record:
        append_history({
            'ts': tick.ts,
            'log_path': tick.log_path,
            'metrics': tick.metrics,
            'arenas_count': len(tick.arenas),
            'arenas_last': tick.arenas[-1] if tick.arenas else None,
            'checkpoints_count': len(tick.checkpoints),
            'alarms_count': len(tick.alarms),
        })

    return rc


if __name__ == '__main__':
    sys.exit(main())
