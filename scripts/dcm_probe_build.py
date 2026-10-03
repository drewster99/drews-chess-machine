"""Which app build produced a probe measurement.

Two builds can score the same checkpoint differently (a numerics change such as the
policy-head tail moves pElo measurably), so every probe record names the binary that
made it. The identity is the app bundle's folder name plus a prefix of the
executable's SHA-256 — `<bundle>.app@sha256:<prefix>` — which tells two builds apart
even when both are called `DrewsChessMachine.app`, without recording a home-folder path.
"""
import hashlib
import os

# A measurement written before probe builds were recorded.
UNRECORDED = "unrecorded"


def probe_build_id(binary_path):
    real = os.path.realpath(binary_path)
    digest = hashlib.sha256()
    with open(real, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    bundle = next((part for part in reversed(real.split(os.sep)) if part.endswith(".app")), None)
    if bundle is None:
        raise ValueError(f"{binary_path}: not inside an .app bundle; cannot name its build")
    return f"{bundle}@sha256:{digest.hexdigest()[:12]}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit("usage: dcm_probe_build.py <app executable>")
    print(probe_build_id(sys.argv[1]))
