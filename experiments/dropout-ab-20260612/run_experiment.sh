#!/bin/bash
# Dropout A/B experiment orchestrator (2026-06-12).
# Fork: one random-init champion saved via SIGUSR2 from a brief --train run.
# Arms: three headless --train runs from --start-model <fork champion>,
# dropout_rate 0.00 / 0.30 / 0.70, 600 trainer steps each.
# Afterwards: restore the resume pointer to the 5K7Z probe save and relaunch.
set -u
EXP="/Users/andrew/cursor/drews-chess-machine/experiments/dropout-ab-20260612"
BIN="/Users/andrew/Library/Developer/Xcode/DerivedData/DrewsChessMachine-eyigcdvyyrcsakaqcybzcfgsurbr/Build/Products/Debug/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine"
LOGDIR="$HOME/Library/Logs/DrewsChessMachine"
SESSDIR="$HOME/Library/Application Support/DrewsChessMachine/Sessions"

newest_log() { ls -t "$LOGDIR"/dcm_log_*.txt | head -1; }
ts() { date "+%H:%M:%S"; }

echo "[$(ts)] === STAGE 1: fork creation ==="
"$BIN" --train --parameters "$EXP/params_drop_0.00.json" > "$EXP/fork_launch.out" 2>&1 &
APP_PID=$!
sleep 6
FORKLOG=$(newest_log)
echo "[$(ts)] fork pid=$APP_PID log=$FORKLOG"
started=0
for i in $(seq 1 120); do
    if grep -q 'starting Play-and-Train' "$FORKLOG"; then started=1; break; fi
    if ! kill -0 "$APP_PID" 2>/dev/null; then echo "[$(ts)] FORK PROCESS DIED EARLY"; exit 1; fi
    sleep 2
done
if [ "$started" != "1" ]; then echo "[$(ts)] FORK NEVER STARTED TRAINING"; kill "$APP_PID"; exit 1; fi
sleep 20
echo "[$(ts)] sending SIGUSR2 to fork"
kill -USR2 "$APP_PID"
for i in $(seq 1 90); do kill -0 "$APP_PID" 2>/dev/null || break; sleep 2; done
if kill -0 "$APP_PID" 2>/dev/null; then echo "[$(ts)] FORK DID NOT EXIT AFTER USR2"; kill "$APP_PID"; exit 1; fi
FORK_NAME=$(grep -o 'Saved session (sigusr2): [^ ]*\.dcmsession' "$FORKLOG" | tail -1 | sed 's/Saved session (sigusr2): //')
FORK="$SESSDIR/$FORK_NAME"
echo "[$(ts)] fork session: $FORK"
if [ ! -f "$FORK/champion.safetensors" ]; then echo "[$(ts)] NO champion.safetensors IN FORK"; exit 1; fi

for R in 0.00 0.30 0.70; do
    echo "[$(ts)] === STAGE 2: arm dropout=$R ==="
    "$BIN" --train --start-model "$FORK/champion.safetensors" \
        --parameters "$EXP/params_drop_$R.json" \
        --training-step-limit 600 --training-time-limit 2700 \
        --output "$EXP/result_drop_$R.json" > "$EXP/arm_$R.out" 2>&1 &
    PID=$!
    sleep 6
    AL=$(newest_log)
    echo "$AL" > "$EXP/armlog_$R.path"
    echo "[$(ts)] arm $R pid=$PID log=$AL"
    for i in $(seq 1 1500); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
    if kill -0 "$PID" 2>/dev/null; then
        echo "[$(ts)] ARM $R TIMEOUT - killing"
        kill "$PID"; sleep 10; kill -9 "$PID" 2>/dev/null
    fi
    REASON=$(python3 -c "import json;print(json.load(open('$EXP/result_drop_$R.json')).get('termination_reason','?'))" 2>/dev/null || echo "no-result-json")
    echo "[$(ts)] arm $R finished: termination=$REASON stats_lines=$(grep -c '\[STATS\]' "$AL")"
done

echo "[$(ts)] === STAGE 3: restore probe pointer + relaunch ==="
python3 - << 'PYEOF'
import plistlib, subprocess, binascii
exp = "/Users/andrew/cursor/drews-chess-machine/experiments/dropout-ab-20260612"
with open(f"{exp}/defaults_backup.plist", "rb") as f:
    d = plistlib.load(f)
p = d["DrewsChessMachine.LastSessionPointer.v1"]
hexs = binascii.hexlify(p).decode()
subprocess.run(["defaults", "write", "com.drewben.DrewsChessMachine",
                "DrewsChessMachine.LastSessionPointer.v1", "-data", hexs], check=True)
print("pointer restored")
PYEOF
cd /Users/andrew/cursor/drews-chess-machine && ./run_latest.sh > "$EXP/relaunch.out" 2>&1 &
echo "[$(ts)] probe relaunched (auto-resume countdown will fire)"
echo "[$(ts)] === EXPERIMENT COMPLETE ==="
