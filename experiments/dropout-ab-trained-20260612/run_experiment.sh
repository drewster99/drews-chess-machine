#!/bin/bash
# Dropout A/B v2 (2026-06-12): four arms from the TRAINED 5K7Z champion.
# Arms: 0.00A (control), 0.00B (replicate, calibrates run-to-run noise),
# 0.30, 0.70. 600 trainer steps each via --training-step-limit.
# No fork stage: --start-model points at the trained champion directly.
set -u
EXP="/Users/andrew/cursor/drews-chess-machine/experiments/dropout-ab-trained-20260612"
BIN="/Users/andrew/Library/Developer/Xcode/DerivedData/DrewsChessMachine-eyigcdvyyrcsakaqcybzcfgsurbr/Build/Products/Debug/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine"
LOGDIR="$HOME/Library/Logs/DrewsChessMachine"
START_MODEL="$HOME/Library/Application Support/DrewsChessMachine/Sessions/20260612-191329-20260601-12-5K7Z-sigusr2.dcmsession/champion.safetensors"

newest_log() { ls -t "$LOGDIR"/dcm_log_*.txt | head -1; }
ts() { date "+%H:%M:%S"; }

run_arm() {
    ARM="$1"; PARAMS="$2"
    echo "[$(ts)] === arm $ARM (params $PARAMS) ==="
    "$BIN" --train --start-model "$START_MODEL" \
        --parameters "$EXP/$PARAMS" \
        --training-step-limit 600 --training-time-limit 2700 \
        --output "$EXP/result_$ARM.json" > "$EXP/arm_$ARM.out" 2>&1 &
    PID=$!
    sleep 6
    AL=$(newest_log)
    echo "$AL" > "$EXP/armlog_$ARM.path"
    echo "[$(ts)] arm $ARM pid=$PID log=$AL"
    for i in $(seq 1 1500); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
    if kill -0 "$PID" 2>/dev/null; then
        echo "[$(ts)] ARM $ARM TIMEOUT - killing"
        kill "$PID"; sleep 10; kill -9 "$PID" 2>/dev/null
    fi
    REASON=$(python3 -c "import json;print(json.load(open('$EXP/result_$ARM.json')).get('termination_reason','?'))" 2>/dev/null || echo "no-result-json")
    echo "[$(ts)] arm $ARM finished: termination=$REASON stats_lines=$(grep -c '\[STATS\]' "$AL")"
}

run_arm "0.00A" "params_drop_0.00.json"
run_arm "0.00B" "params_drop_0.00.json"
run_arm "0.30"  "params_drop_0.30.json"
run_arm "0.70"  "params_drop_0.70.json"

echo "[$(ts)] === restore pointer + relaunch probe ==="
python3 - << 'PYEOF'
import plistlib, subprocess, binascii
exp = "/Users/andrew/cursor/drews-chess-machine/experiments/dropout-ab-trained-20260612"
with open(f"{exp}/defaults_backup.plist", "rb") as f:
    d = plistlib.load(f)
p = d["DrewsChessMachine.LastSessionPointer.v1"]
subprocess.run(["defaults", "write", "com.drewben.DrewsChessMachine",
                "DrewsChessMachine.LastSessionPointer.v1", "-data",
                binascii.hexlify(p).decode()], check=True)
print("pointer restored")
PYEOF
cd /Users/andrew/cursor/drews-chess-machine && ./run_latest.sh > "$EXP/relaunch.out" 2>&1 &
echo "[$(ts)] probe relaunched"
echo "[$(ts)] === EXPERIMENT v2 COMPLETE ==="
