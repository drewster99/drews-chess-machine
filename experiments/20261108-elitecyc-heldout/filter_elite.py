"""Stream a Lichess monthly PGN on stdin; write to stdout only the games that
meet the Lichess Elite Database rule (nikonoel, December 2021 on): one player
rated 2500+, the other 2300+, bullet and ultrabullet excluded, decisive or drawn
result. Counts go to stderr every 5M games and at the end."""
import sys

def main():
    inp, out, err = sys.stdin.buffer, sys.stdout.buffer, sys.stderr
    block, event, welo, belo, result = [], b"", -1, -1, b""
    seen = kept = 0

    def flush():
        nonlocal seen, kept
        if not block:
            return
        seen += 1
        hi, lo = max(welo, belo), min(welo, belo)
        if hi >= 2500 and lo >= 2300 and b"Bullet" not in event and result in (b"1-0", b"0-1", b"1/2-1/2"):
            kept += 1
            out.writelines(block)
        if seen % 5_000_000 == 0:
            print(f"seen={seen} kept={kept}", file=err, flush=True)

    def elo(line):
        v = line.split(b'"')[1]
        return int(v) if v.isdigit() else -1

    for line in inp:
        if line.startswith(b"[Event "):
            flush()
            block, event, welo, belo, result = [], line, -1, -1, b""
        elif line.startswith(b"[WhiteElo "):
            welo = elo(line)
        elif line.startswith(b"[BlackElo "):
            belo = elo(line)
        elif line.startswith(b"[Result "):
            result = line.split(b'"')[1]
        block.append(line)
    flush()
    print(f"done seen={seen} kept={kept}", file=err, flush=True)

main()
