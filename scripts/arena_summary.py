#!/usr/bin/env python3
"""arena_summary.py — summarize arena trajectory from the active session log.

Walks every [ARENA] line in the most recent session log and prints:
  - per-arena W/D/L, score, elo, promoted-or-kept tag
  - rolling moving average of score across the last 5 arenas
  - count of promotions

Usage:
  python3 scripts/arena_summary.py              # most recent session
  python3 scripts/arena_summary.py --log <path>
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

LOG_GLOB = os.path.expanduser('~/Library/Logs/DrewsChessMachine/dcm_log_*.txt')

LINE_RE = re.compile(
    r'\[ARENA\] #(\d+) kv .*step=(\d+) games=(\d+) w=(\d+) d=(\d+) l=(\d+) '
    r'score=(\d+\.\d+) elo=([+-]?\d+) elo_lo=([+-]?\d+) elo_hi=([+-]?\d+).*'
    r'promoted=(\d+) kind=(\w+) dur_sec=([\d.]+).*candidate=(\S+) champion=(\S+)'
)


def latest_log() -> Path | None:
    paths = sorted(glob.glob(LOG_GLOB))
    return Path(paths[-1]) if paths else None


def parse(log: Path) -> list[dict]:
    out = []
    with open(log, 'r', errors='replace') as f:
        for line in f:
            if '[ARENA]' not in line:
                continue
            m = LINE_RE.search(line)
            if not m:
                continue
            out.append({
                'index': int(m.group(1)),
                'step': int(m.group(2)),
                'w': int(m.group(4)), 'd': int(m.group(5)), 'l': int(m.group(6)),
                'score': float(m.group(7)),
                'elo': int(m.group(8)),
                'elo_ci': (int(m.group(9)), int(m.group(10))),
                'promoted': int(m.group(11)) == 1,
                'kind': m.group(12),
                'dur_sec': float(m.group(13)),
                'candidate': m.group(14), 'champion': m.group(15),
            })
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--log', default=None)
    args = p.parse_args()
    log = Path(args.log) if args.log else latest_log()
    if not log or not log.exists():
        print('no log found', file=sys.stderr)
        return 1
    arenas = parse(log)
    if not arenas:
        print(f'no [ARENA] lines yet in {log.name}', file=sys.stderr)
        return 0

    print(f'log: {log.name}   arenas={len(arenas)}   promoted={sum(1 for a in arenas if a["promoted"])}')
    print(f'{"#":>3}  {"step":>6}  {"score":>6}  {"W/D/L":>10}  {"elo":>5}  {"CI":>14}  {"dur":>5}  status')
    print('-' * 78)
    last5 = []
    for a in arenas:
        last5.append(a['score'])
        if len(last5) > 5:
            last5.pop(0)
        avg5 = sum(last5) / len(last5)
        status = 'PROMOTED' if a['promoted'] else 'kept'
        print(
            f'{a["index"]:>3}  {a["step"]:>6}  {a["score"]:.3f}  '
            f'{a["w"]:>2}/{a["d"]:>2}/{a["l"]:<2}  '
            f'{a["elo"]:>+5d}  [{a["elo_ci"][0]:+4d},{a["elo_ci"][1]:+4d}]  '
            f'{a["dur_sec"]:>5.1f}  {status}  (5-avg={avg5:.3f})'
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
