#!/usr/bin/env python3
"""trend.py — print the rolling tick history as a compact metric trajectory.

Reads scripts/.tick_history.jsonl (written by tick_status.py) and emits one
line per tick with the columns most useful for trend reading. Useful when
you want the last N ticks at a glance without reading every tick block.

Usage:
  python3 scripts/trend.py                 # last 20 ticks
  python3 scripts/trend.py --tail 60       # last 60 ticks
  python3 scripts/trend.py --metric pEnt   # only show pEnt trajectory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HIST_PATH = Path(__file__).resolve().parent / '.tick_history.jsonl'


def load() -> list[dict]:
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


def fmt_elapsed(s: int) -> str:
    return f'{s // 3600}h{(s % 3600) // 60:02d}m'


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--tail', type=int, default=20)
    p.add_argument('--metric', default=None,
                   help='only show one metric column (e.g. pEnt, gNorm, legalMass)')
    args = p.parse_args()

    hist = load()
    if not hist:
        print('no tick history yet — run scripts/tick_status.py first', file=sys.stderr)
        return 1
    rows = hist[-args.tail:]

    if args.metric:
        print(f'{"elapsed":>9}  {"steps":>6}  {args.metric:>14}  {"Δ":>10}')
        prev = None
        for r in rows:
            m = r.get('metrics', {})
            v = m.get(args.metric)
            d = (v - prev) if (prev is not None and v is not None) else None
            ds = f'{d:+.4g}' if d is not None else '-'
            vs = f'{v:.4g}' if v is not None else '-'
            print(f'{fmt_elapsed(m.get("elapsed_sec", 0)):>9}  '
                  f'{int(m.get("steps", 0)):>6}  {vs:>14}  {ds:>10}')
            if v is not None:
                prev = v
        return 0

    print(f'{"elapsed":>9}  {"steps":>6}  {"pEnt":>6}  {"gNorm":>6}  '
          f'{"pLogit":>7}  {"vAbs":>6}  {"legalMass":>10}  {"top1Lg":>7}  '
          f'{"avgLen":>6}  {"ratio":>6}  {"arenas":>7}')
    for r in rows:
        m = r.get('metrics', {})
        a = r.get('arenas_count', 0)
        last_a = r.get('arenas_last')
        a_str = f'{a}'
        if last_a:
            tag = 'P' if last_a.get('promoted') else 'k'
            a_str = f'{a}({last_a.get("score", 0):.2f}{tag})'
        print(
            f'{fmt_elapsed(m.get("elapsed_sec", 0)):>9}  '
            f'{int(m.get("steps", 0)):>6}  '
            f'{m.get("pEnt", float("nan")):>6.2f}  '
            f'{m.get("gNorm", float("nan")):>6.1f}  '
            f'{m.get("pLogitAbsMax", float("nan")):>7.2f}  '
            f'{m.get("vAbs", float("nan")):>6.3f}  '
            f'{m.get("legalMass", float("nan")):>10.4f}  '
            f'{m.get("top1Legal", float("nan")):>7.3f}  '
            f'{m.get("avgLen", float("nan")):>6.0f}  '
            f'{m.get("ratio_cur", float("nan")):>6.2f}  '
            f'{a_str:>7}'
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
