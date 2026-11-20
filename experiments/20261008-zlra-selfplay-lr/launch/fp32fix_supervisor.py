#!/usr/bin/env python3
"""Keep R-elitecyc-fp32fix training unless the battery is low or something else needs the GPU/RAM.

Owner standing order 2026-11-19:
  - always running, except:
  - battery < 20 % -> gentle stop (one SIGINT = finish step, abort save, exit); resume when > 25 %.
  - another training run, or a local-LLM server (oMLX, ollama, LM Studio, ...) using the GPU/RAM heavily ->
    tell the owner; if no answer within GRACE_SEC, or at once when memory becomes dire, stop; resume when clear.

The supervisor runs detached (nohup) and appends one line per event to EVENTS; the Claude session waits on that
file and acts on each line (pushes the owner, probes checkpoints). Control files in this folder:
  fp32fix_supervisor.hold   present -> never start the trainer (owner said stop, or a refusal/health/GPU-fault exit)
  fp32fix_supervisor.allow  present -> owner said keep running through the current conflict (deleted once it clears)
"""
import json, os, re, signal, subprocess, sys, time
from pathlib import Path

OPS = Path.home() / "Library/Application Support/DrewsChessMachine/OpsScripts"
MODELS = Path.home() / "Library/Application Support/DrewsChessMachine/Models"
LAUNCHER = "/Users/andrew/cursor/drews-chess-machine/experiments/20261008-zlra-selfplay-lr/launch/elitecyc_fp32fix_resume_latest_launch.sh"
STEM = "20261114-zlra-elitecyc-fp32fix-latest"
TARGET_STEP = 1_000_000
EVENTS = OPS / "fp32fix_supervisor.events"
STATE = OPS / "fp32fix_supervisor.state.json"
EXITFILE = OPS / "fp32fix_trainer.exit"
HOLD = OPS / "fp32fix_supervisor.hold"
ALLOW = OPS / "fp32fix_supervisor.allow"
PIDFILE = OPS / "fp32fix_supervisor.pid"

POLL_SEC = 60
BATTERY_STOP_BELOW = 20
BATTERY_RESUME_ABOVE = 25
GRACE_SEC = 8 * 60                # owner: "5 to 10 minutes"
CLEAR_POLLS_BEFORE_RESUME = 3     # conflict must be gone this many polls in a row
STOP_WAIT_SEC = 15 * 60           # an abort save can take minutes; never send a second SIGINT
LLM_RSS_HEAVY_KB = 3 * 1024 * 1024
LLM_CPU_HEAVY = 80.0
OTHER_RSS_HEAVY_KB = 12 * 1024 * 1024
DIRE_FREE_LEVEL = 10              # kern.memorystatus_level (% free)
DIRE_PRESSURE_LEVEL = 4           # kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical
CRASH_RESTART_LIMIT = 3
CRASH_WINDOW_SEC = 2 * 3600

TRAINER_RE = re.compile(r"Contents/MacOS/DrewsChessMachine --replay-corpus \S+ --start-model .*" + re.escape(STEM) + r"-step")
DCM_TRAIN_RE = re.compile(r"Contents/MacOS/DrewsChessMachine\b.*(--replay-corpus|--train-vs-uci|--train\b)")
DCM_GUI_RE = re.compile(r"Contents/MacOS/DrewsChessMachine(\s|$)")
LLM_RE = re.compile(r"ollama|omlx|lm ?studio|/lms\b|llama-server|llama-cli|llama\.cpp|mlx_lm|mlx-lm|mlx\.launch|"
                    r"koboldcpp|localai|vllm|gpt4all|msty|jan\.app|draw things|\bexo\b", re.I)
IGNORE_RE = re.compile(r"Xcode|SWBBuildService|swift-frontend|SourceKit|kernel_task|WindowServer", re.I)


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def emit(kind, msg):
    line = f"{now()} {kind} {msg}"
    with open(EVENTS, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def battery():
    out = sh(["pmset", "-g", "batt"])
    m = re.search(r"(\d+)%", out)
    return (int(m.group(1)) if m else None), ("AC Power" in out)


def sysctl_int(name):
    try:
        return int(sh(["sysctl", "-n", name]).strip())
    except ValueError:
        return None


def processes():
    out = sh(["ps", "-axww", "-o", "pid=,rss=,%cpu=,command="])
    rows = []
    for line in out.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) == 4:
            try:
                rows.append((int(parts[0]), int(parts[1]), float(parts[2]), parts[3]))
            except ValueError:
                pass
    return rows


def gui_training_logs():
    """GUI Play-and-Train writes [STATS] lines; report any session log doing so in the last 10 minutes."""
    hits = []
    logdir = Path.home() / "Library/Logs/DrewsChessMachine"
    cutoff = time.time() - 600
    for p in logdir.glob("dcm_log_*.txt"):
        try:
            if p.stat().st_mtime < cutoff:
                continue
            with open(p, "rb") as f:
                f.seek(max(0, p.stat().st_size - 200_000))
                if b"[STATS]" in f.read():
                    hits.append(p.name)
        except OSError:
            pass
    return hits


def trainer_pids(rows):
    return [pid for pid, _, _, cmd in rows if TRAINER_RE.search(cmd)]


def conflicts(rows, me):
    found = []
    llm_rss = 0; llm_cpu = 0.0; llm_names = set()
    for pid, rss, cpu, cmd in rows:
        if pid in me or IGNORE_RE.search(cmd):
            continue
        if DCM_TRAIN_RE.search(cmd) and not TRAINER_RE.search(cmd):
            found.append(f"other DCM training pid={pid}: {cmd[:160]}")
            continue
        if DCM_GUI_RE.search(cmd) or "fp32fix_supervisor" in cmd:
            continue
        if LLM_RE.search(cmd):
            llm_rss += rss; llm_cpu += cpu
            llm_names.add(os.path.basename(cmd.split(" --")[0])[:40])
            continue
        if rss >= OTHER_RSS_HEAVY_KB:
            found.append(f"heavy RAM pid={pid} rss={rss // 1048576}GB: {cmd[:120]}")
    if llm_rss >= LLM_RSS_HEAVY_KB or llm_cpu >= LLM_CPU_HEAVY:
        found.append(f"local LLM {sorted(llm_names)} rss={llm_rss / 1048576:.1f}GB cpu={llm_cpu:.0f}%")
    for log in gui_training_logs():
        found.append(f"GUI Play-and-Train writing [STATS] to {log}")
    return found


def dire():
    free = sysctl_int("kern.memorystatus_level")
    pressure = sysctl_int("kern.memorystatus_vm_pressure_level")
    reasons = []
    if free is not None and free <= DIRE_FREE_LEVEL:
        reasons.append(f"memory free {free}%")
    if pressure is not None and pressure >= DIRE_PRESSURE_LEVEL:
        reasons.append("memory pressure critical")
    return reasons


def latest_step():
    best = None
    for p in MODELS.glob(f"{STEM}-step*.safetensors"):
        m = re.fullmatch(re.escape(STEM) + r"-step(\d+)\.safetensors", p.name)
        if m:
            best = max(best or 0, int(m.group(1)))
    return best


def start_trainer():
    if EXITFILE.exists():
        EXITFILE.unlink()
    subprocess.Popen(
        ["zsh", "-c", f'caffeinate -i zsh "{LAUNCHER}"; echo $? > "{EXITFILE}"'],
        stdin=subprocess.DEVNULL, stdout=open(OPS / "fp32fix_launcher.log", "a"), stderr=subprocess.STDOUT,
        start_new_session=True)
    for _ in range(60):
        time.sleep(2)
        pids = trainer_pids(processes())
        if pids:
            emit("STARTED", f"trainer pid={pids[0]} from step {latest_step()}")
            return True
        if EXITFILE.exists():
            break
    code = EXITFILE.read_text().strip() if EXITFILE.exists() else "?"
    emit("START-FAILED", f"trainer did not start (launcher exit {code}); holding. See fp32fix_launcher.log")
    HOLD.write_text(f"{now()} start failed, exit {code}\n")
    return False


SIGNALLED = set()   # pids already sent SIGINT: a second SIGINT force-quits without the abort save


def stop_trainer(reason):
    pids = trainer_pids(processes())
    if not pids or pids[0] in SIGNALLED:
        return
    SIGNALLED.add(pids[0])
    emit("STOPPING", f"pid={pids[0]} reason: {reason}")
    os.kill(pids[0], signal.SIGINT)
    deadline = time.time() + STOP_WAIT_SEC
    while time.time() < deadline:
        time.sleep(5)
        if not trainer_pids(processes()):
            emit("STOPPED", f"abort save at step {latest_step()} (reason: {reason})")
            return
    emit("STOP-SLOW", f"pid={pids[0]} still running {STOP_WAIT_SEC // 60} min after SIGINT; not signalling again")


def main():
    if PIDFILE.exists():
        old = PIDFILE.read_text().strip()
        if old.isdigit() and subprocess.run(["kill", "-0", old], capture_output=True).returncode == 0:
            print(f"supervisor already running pid={old}", file=sys.stderr)
            sys.exit(1)
    PIDFILE.write_text(str(os.getpid()))
    emit("SUPERVISOR", f"started pid={os.getpid()}")

    stop_reason = None          # why WE stopped it: "battery" | "conflict" | None
    conflict_since = None
    conflict_notified = None
    clear_polls = 0
    we_started = False
    crashes = []
    seen_ckpts = {s for s in [latest_step()] if s}
    for p in MODELS.glob(f"{STEM}-step*000.safetensors"):
        m = re.search(r"-step(\d+)\.", p.name)
        if m:
            seen_ckpts.add(int(m.group(1)))

    while True:
        try:
            rows = processes()
            pids = trainer_pids(rows)
            running = bool(pids)
            pct, on_ac = battery()
            me = set(pids) | {os.getpid()}
            found = conflicts(rows, me)
            dire_reasons = dire()

            # checkpoints to report (every 1000 trainer steps; owner 2026-11-19)
            for p in MODELS.glob(f"{STEM}-step*000.safetensors"):
                m = re.search(r"-step(\d+)\.safetensors$", p.name)
                if m and int(m.group(1)) not in seen_ckpts:
                    seen_ckpts.add(int(m.group(1)))
                    emit("CKPT", f"step {m.group(1)} {p}")

            # trainer exit we did not cause
            if we_started and not running and stop_reason is None:
                we_started = False
                code = EXITFILE.read_text().strip() if EXITFILE.exists() else "?"
                step = latest_step()
                if step is not None and step >= TARGET_STEP:
                    emit("DONE", f"reached step {step}; supervisor exiting")
                    break
                if code in ("2", "35", "36"):
                    HOLD.write_text(f"{now()} trainer exit {code}\n")
                    emit("TRAINER-EXIT", f"exit {code} (2=refused 35=health stop 36=GPU fault) at step {step}; HOLDING")
                else:
                    crashes = [t for t in crashes if time.time() - t < CRASH_WINDOW_SEC] + [time.time()]
                    if len(crashes) > CRASH_RESTART_LIMIT:
                        HOLD.write_text(f"{now()} {len(crashes)} unexpected exits in 2h\n")
                        emit("TRAINER-EXIT", f"exit {code} at step {step}; {len(crashes)} in 2h -> HOLDING")
                    else:
                        emit("TRAINER-EXIT", f"unexpected exit {code} at step {step}; will restart")

            # battery
            if pct is not None and pct < BATTERY_STOP_BELOW and running:
                stop_trainer(f"battery {pct}% < {BATTERY_STOP_BELOW}%")
                stop_reason = "battery"; we_started = False; running = False
            if stop_reason == "battery" and pct is not None and pct > BATTERY_RESUME_ABOVE:
                emit("BATTERY-OK", f"battery {pct}%")
                stop_reason = None

            # conflicts
            if found:
                clear_polls = 0
                if conflict_since is None:
                    conflict_since = time.time()
                key = "; ".join(sorted(found))
                if conflict_notified != key:
                    conflict_notified = key
                    emit("CONFLICT", f"{key} | trainer {'running' if running else 'stopped'}"
                         f"{' | DIRE: ' + ', '.join(dire_reasons) if dire_reasons else ''}")
                if running and dire_reasons:
                    stop_trainer(f"conflict + dire memory ({', '.join(dire_reasons)}): {key}")
                    stop_reason = "conflict"; we_started = False; running = False
                elif running and not ALLOW.exists() and time.time() - conflict_since >= GRACE_SEC:
                    stop_trainer(f"conflict unanswered {GRACE_SEC // 60} min: {key}")
                    stop_reason = "conflict"; we_started = False; running = False
            else:
                if conflict_since is not None:
                    clear_polls += 1
                    if clear_polls >= CLEAR_POLLS_BEFORE_RESUME:
                        emit("CONFLICT-CLEAR", "no conflicting GPU/RAM users")
                        conflict_since = None; conflict_notified = None
                        if ALLOW.exists():
                            ALLOW.unlink()
                        if stop_reason == "conflict":
                            stop_reason = None

            # (re)start
            if (not running and stop_reason is None and conflict_since is None and not HOLD.exists()
                    and (pct is None or pct > BATTERY_RESUME_ABOVE or (pct >= BATTERY_STOP_BELOW and on_ac))
                    and (latest_step() or 0) < TARGET_STEP):
                we_started = start_trainer()
            elif running:
                we_started = True

            STATE.write_text(json.dumps({
                "time": now(), "running": running, "pid": pids[0] if pids else None, "battery": pct, "ac": on_ac,
                "stop_reason": stop_reason, "conflicts": found, "dire": dire_reasons, "hold": HOLD.exists(),
                "allow": ALLOW.exists(), "latest_step": latest_step()}, indent=1))
        except Exception as e:  # keep supervising; report every failure
            emit("SUPERVISOR-ERROR", repr(e))
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
