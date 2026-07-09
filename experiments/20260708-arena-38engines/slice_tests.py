#!/usr/bin/env python3
"""Strip the stray leading test games from the raw tournament PGN.

The fresh cutechess PGN began with a short 2-engine warmup match
(`DCM - Qeu8e5` vs `Stockfish`) that we ran right before launching the real
38-engine round-robin. Those games are also tagged `Round 1`, so they can't be
separated by round number. This removes the leading CONTIGUOUS block of games
whose two players are both within the warmup pair, leaving only the real
tournament.

Usage: python3 slice_tests.py <in.pgn> <out.pgn>
"""
import re
import sys

TEST_PAIR = {"DCM - Qeu8e5", "Stockfish"}


def players(block):
    w = b = None
    for line in block:
        m = re.match(r'\[White "(.*)"\]', line)
        if m:
            w = m.group(1)
        m = re.match(r'\[Black "(.*)"\]', line)
        if m:
            b = m.group(1)
    return w, b


def main(src, dst):
    blocks, cur = [], []
    for line in open(src):
        if line.startswith("[Event ") and cur:
            blocks.append(cur)
            cur = []
        cur.append(line)
    if cur:
        blocks.append(cur)

    i = 0
    while i < len(blocks):
        w, b = players(blocks[i])
        if w in TEST_PAIR and b in TEST_PAIR:
            i += 1
        else:
            break

    with open(dst, "w") as f:
        for block in blocks[i:]:
            f.writelines(block)
    print(f"stripped {i} leading test games; wrote {len(blocks) - i} tournament games to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
