#!/usr/bin/env python3
"""sessions_summary.py — list .dcmsession autosaves with metadata.

Walks ~/Library/Application Support/DrewsChessMachine/Sessions/, parses each
.dcmsession's metadata.json, and emits a compact one-line-per-save summary
with trigger, model IDs, training time at save, arena count + promotions,
and any STATS-equivalent counters.

Usage:
  python3 scripts/sessions_summary.py              # last 20 sessions
  python3 scripts/sessions_summary.py --tail 50
  python3 scripts/sessions_summary.py --json       # raw JSON for scripting
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SESS_DIR = Path(os.path.expanduser('~/Library/Application Support/DrewsChessMachine/Sessions'))


def collect() -> list[dict]:
    if not SESS_DIR.exists():
        return []
    out = []
    for d in sorted(SESS_DIR.iterdir()):
        if not d.is_dir() or not d.name.endswith('.dcmsession'):
            continue
        meta = d / 'metadata.json'
        info: dict = {'name': d.name, 'path': str(d), 'size_mb': None,
                      'metadata_present': meta.exists()}
        try:
            total = sum(p.stat().st_size for p in d.rglob('*') if p.is_file())
            info['size_mb'] = round(total / (1024 * 1024), 1)
        except Exception:
            pass
        if meta.exists():
            try:
                info['metadata'] = json.loads(meta.read_text())
            except Exception as e:
                info['metadata_error'] = str(e)
        out.append(info)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--tail', type=int, default=20)
    p.add_argument('--json', action='store_true', help='raw JSON dump')
    args = p.parse_args()

    rows = collect()
    if not rows:
        print(f'no sessions found under {SESS_DIR}', file=sys.stderr)
        return 1
    rows = rows[-args.tail:]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print(f'{"name":<60}  {"size":>7}  {"trigger":<14}  {"trainer→champion":<40}  notes')
    print('-' * 140)
    for r in rows:
        meta = r.get('metadata', {}) or {}
        # Best-effort extraction; metadata schema may vary across builds.
        trigger = meta.get('trigger') or meta.get('saveTrigger') or '?'
        trainer = meta.get('trainerID') or meta.get('trainer_id') or '?'
        champ   = meta.get('championID') or meta.get('champion_id') or '?'
        elapsed = meta.get('trainingElapsedSeconds') or meta.get('training_elapsed_seconds')
        steps   = meta.get('completedTrainSteps') or meta.get('training_steps')
        arenas  = meta.get('arenaCount') or meta.get('arenas_count')
        prom    = meta.get('arenaPromoted') or meta.get('promotions')
        notes_parts = []
        if elapsed is not None:
            notes_parts.append(f'elapsed={int(elapsed)}s')
        if steps is not None:
            notes_parts.append(f'steps={steps}')
        if arenas is not None:
            notes_parts.append(f'arenas={arenas}')
        if prom is not None:
            notes_parts.append(f'promo={prom}')
        if not r.get('metadata_present'):
            notes_parts.append('NO_METADATA')
        notes = ' '.join(notes_parts)
        size = f'{r["size_mb"]}M' if r.get('size_mb') is not None else '?'
        print(f'{r["name"]:<60}  {size:>7}  {trigger:<14}  '
              f'{(trainer + "→" + champ):<40}  {notes}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
