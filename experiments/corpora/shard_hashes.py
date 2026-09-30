#!/usr/bin/env python3
"""Print per-shard hashes of a DCM game corpus, for comparing a rebuilt corpus
against the table in its manifest (experiments/corpora/<corpusID>.md).

Usage: python3 shard_hashes.py <corpus dir | corpus ID under Corpora/>

Every sealed shard is `front header (256 B) + body + trailer (64 B)`. The front
header carries the corpus ID, source ID and creation time, and the trailer
carries the seal time and a SHA-256 over header + body, so a rebuilt shard never
matches the original's whole-file SHA-256: the corpus ID is freshly minted on
every import. The body (the framed game records) depends only on the games
imported and the shard size limit, so `body_sha256` is the column to compare.
"""
import hashlib, os, struct, sys

FRONT_HEADER_SIZE = 256
TRAILER_SIZE = 64

def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    arg = sys.argv[1]
    directory = arg if os.path.isdir(arg) else os.path.expanduser(
        f"~/Library/Application Support/DrewsChessMachine/Corpora/{arg}")
    if not os.path.isdir(directory):
        sys.exit(f"no corpus directory at {directory}")
    names = sorted(n for n in os.listdir(directory)
                   if n.startswith("shard-") and n.endswith(".dcmgames"))
    if not names:
        sys.exit(f"no sealed shards in {directory}")
    print("| shard | bytes | games | plies | sha256 (whole file) | body_sha256 |")
    print("|---|---:|---:|---:|---|---|")
    total_bytes = total_games = total_plies = 0
    for name in names:
        with open(os.path.join(directory, name), "rb") as f:
            data = f.read()
        if data[:8] != b"DCMGAME1":
            sys.exit(f"{name}: bad front magic")
        trailer = data[-TRAILER_SIZE:]
        if trailer[:8] != b"DCMGSEAL":
            sys.exit(f"{name}: bad trailer magic (not sealed)")
        games, plies = struct.unpack("<qq", trailer[8:24])
        if hashlib.sha256(data[:-TRAILER_SIZE]).digest() != trailer[32:64]:
            sys.exit(f"{name}: stored SHA-256 does not match contents")
        whole = hashlib.sha256(data).hexdigest()
        body = hashlib.sha256(data[FRONT_HEADER_SIZE:-TRAILER_SIZE]).hexdigest()
        print(f"| `{name}` | {len(data):,} | {games:,} | {plies:,} | `{whole}` | `{body}` |")
        total_bytes += len(data); total_games += games; total_plies += plies
    print(f"| **total ({len(names)} shards)** | {total_bytes:,} | {total_games:,} | {total_plies:,} | | |")

if __name__ == "__main__":
    main()
