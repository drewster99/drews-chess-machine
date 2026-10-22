#!/usr/bin/env python3
"""Benchmark Core ML exports against the app's MPSGraph network, per model.

For each model file:
  1. a float32 and a float16 copy (`--derive-model --set-compute-dtype ...`)
     where the model isn't that type already, for the same-math pairs;
  2. four Core ML packages via scripts/dcm_coreml_export.py: fp32 (from the
     float32 copy), mirror and fp16 (from the model itself), and mirror of the
     float16 copy (exactly the ops the float16 MPSGraph runs in fp16);
  3. a `--coreml-benchmark` config with eight arms: the model, float32 and
     float16 MPSGraph networks; Core ML fp32, mirror, fp16 and float16-mirror
     on cpu-gpu; Core ML fp16 on cpu-ne. A copy the model already is (its own
     type) is not repeated;
  4. the run: the app's table (`<stem>.txt`) and report (`<stem>.json`) in --out.

The packages are reused from --work by name, and a package's batch sizes are
fixed at export: use a fresh --work folder for a different --batch-sizes.

Before anything runs, it refuses while any other DrewsChessMachine process
except the Lichess bot is up (a trainer would share the GPU), and records each
bot's CPU seconds before and after every model's run in `machine.json`: a bot
game played mid-run shows up there.

The timing rules live in the app (`CoreMLBenchmark`); this script only
prepares files and runs it. Plan: documentation/plans-completed/COREML_BENCHMARK_PLAN.md.

  python3 scripts/dcm_coreml_benchmark.py --app <Release binary> --work <folder> --out <results folder> \
      MODEL.safetensors [MODEL ...] [--batch-sizes 1,8,32,64,128,256,512,1024] [--in-flight 1,4] \
      [--rounds 5] [--round-seconds 1.5]
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dcm_arch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORTER = os.path.join(REPO, "scripts", "dcm_coreml_export.py")
DEFAULT_PYTHON = os.path.expanduser("~/.venvs/dcm-coreml/bin/python")
BOT_FLAG = "--lichess-bot"
APP_EXECUTABLE_TAIL = ".app/Contents/MacOS/DrewsChessMachine"


class BenchmarkError(Exception):
    pass


def dcm_processes():
    """(pid, cpu seconds, command) of every running DrewsChessMachine process.
    Matched on the executable (`comm`), not the command line, so a process that
    only names the app in its arguments (this script's --app) isn't one."""
    executables = subprocess.run(["ps", "-Ao", "pid=,comm="], capture_output=True, text=True, check=True).stdout
    pids = set()
    for line in executables.splitlines():
        pid, _, executable = line.strip().partition(" ")
        if executable.endswith(APP_EXECUTABLE_TAIL):
            pids.add(int(pid))
    output = subprocess.run(["ps", "-Ao", "pid=,time=,args="], capture_output=True, text=True, check=True).stdout
    processes = []
    for line in output.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and int(parts[0]) in pids:
            processes.append((int(parts[0]), cpu_seconds(parts[1]), parts[2]))
    return processes


def cpu_seconds(text):
    """`ps` TIME ([[dd-]hh:]mm:ss.ss) in seconds."""
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        days = int(day_text)
    fields = [float(f) for f in text.split(":")]
    seconds = 0.0
    for field in fields:
        seconds = seconds * 60 + field
    return days * 86400 + seconds


def check_machine():
    """The Lichess bots running now; refuses any other DrewsChessMachine process."""
    bots, others = [], []
    for pid, cpu, command in dcm_processes():
        (bots if BOT_FLAG in command.split() else others).append({"pid": pid, "cpu_seconds": cpu, "command": command})
    if others:
        listed = "; ".join(f"{p['pid']} {p['command']}" for p in others)
        raise BenchmarkError(f"other DrewsChessMachine processes are running (they would share the GPU): {listed}")
    return bots


def run(command, log=True):
    if log:
        print("$ " + " ".join(command), file=sys.stderr, flush=True)
    completed = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if completed.returncode != 0:
        raise BenchmarkError(f"{os.path.basename(command[0])} exited {completed.returncode}: "
                             f"{completed.stderr.strip()[-2000:]}")
    return completed


def prepare(model, app, python, work, batch_sizes):
    """The model's arms, creating the float32 / float16 copies and the packages it needs."""
    # The app resolves a config's relative paths against the config's own folder.
    model = os.path.abspath(model)
    work = os.path.abspath(work)
    stem = os.path.splitext(os.path.basename(model))[0]
    folder = os.path.join(work, stem)
    os.makedirs(folder, exist_ok=True)
    architecture = dcm_arch.architecture(model)
    native = architecture["compute_data_type"]
    copies = {}
    for dtype in ("float32", "float16"):
        if native == dtype:
            copies[dtype] = model
        else:
            copies[dtype] = os.path.join(folder, f"{stem}-{dtype}.safetensors")
            if not os.path.exists(copies[dtype]):
                run([app, "--derive-model", "--from", model, "--set-compute-dtype", dtype, "--out", copies[dtype]])
    packages = {}
    for name, precision, source in (("fp32", "fp32", copies["float32"]), ("mirror", "mirror", model),
                                    ("fp16", "fp16", model), ("float16-mirror", "mirror", copies["float16"])):
        if name == "float16-mirror" and native == "float16":
            continue
        package = os.path.join(folder, f"{stem}-{name}.mlpackage")
        if not os.path.exists(package):
            run([python, EXPORTER, source, "--out", package, "--app", app, "--precision", precision,
                 "--batch-sizes", ",".join(map(str, batch_sizes))])
        packages[name] = package
    arms = [{"label": f"mps-{native}", "engine": "mpsgraph", "model": model}]
    arms += [{"label": f"mps-{dtype}", "engine": "mpsgraph", "model": copies[dtype]}
             for dtype in ("float32", "float16") if dtype != native]
    arms += [
        {"label": "coreml-fp32-gpu", "engine": "coreml", "model": copies["float32"], "coreml": packages["fp32"],
         "compute_units": "cpu-gpu"},
        {"label": "coreml-mirror-gpu", "engine": "coreml", "model": model, "coreml": packages["mirror"],
         "compute_units": "cpu-gpu"},
        {"label": "coreml-fp16-gpu", "engine": "coreml", "model": model, "coreml": packages["fp16"],
         "compute_units": "cpu-gpu"},
        {"label": "coreml-fp16-ne", "engine": "coreml", "model": model, "coreml": packages["fp16"],
         "compute_units": "cpu-ne"},
    ]
    if native != "float16":
        # A float16 model's own mirror arm is already this pair.
        arms.insert(-1, {"label": "coreml-float16-mirror-gpu", "engine": "coreml", "model": copies["float16"],
                         "coreml": packages["float16-mirror"], "compute_units": "cpu-gpu"})
    return stem, folder, arms


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--app", required=True, help="a Release DrewsChessMachine binary")
    parser.add_argument("--python", default=DEFAULT_PYTHON, help="a Python with coremltools")
    parser.add_argument("--work", required=True, help="folder for float32 copies and packages (reused)")
    parser.add_argument("--out", required=True, help="results folder (created; must not hold results already)")
    parser.add_argument("--batch-sizes", default="1,8,32,64,128,256,512,1024")
    parser.add_argument("--in-flight", default="1,4")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--round-seconds", type=float, default=1.5)
    parser.add_argument("--min-top-move-agreement", type=float, default=0.95,
                        help="parity floor every Core ML arm must meet at every batch size")
    args = parser.parse_args(argv)
    batch_sizes = [int(v) for v in args.batch_sizes.split(",")]
    in_flight = [int(v) for v in args.in_flight.split(",")]
    os.environ["DCM_BINARY"] = args.app
    try:
        bots = check_machine()
        os.makedirs(args.out, exist_ok=True)
        machine_path = os.path.join(args.out, "machine.json")
        for model in args.models:
            stem = os.path.splitext(os.path.basename(model))[0]
            if os.path.exists(os.path.join(args.out, f"{stem}.json")):
                raise BenchmarkError(f"{args.out} already holds results for {stem}")
        version = run([args.app, "--version"], log=False).stdout.strip()
        if "configuration=Release" not in version:
            raise BenchmarkError(f"{args.app} is not a Release build: {version}")
        # A later invocation adds its models to the folder's record.
        machine = {"runs": []}
        if os.path.exists(machine_path):
            with open(machine_path) as handle:
                machine = json.load(handle)
        machine["runs"].append({"app": args.app, "version": version, "bots_at_start": bots, "models": []})
        current = machine["runs"][-1]
        prepared = [prepare(m, args.app, args.python, args.work, batch_sizes) for m in args.models]
        for stem, folder, arms in prepared:
            config = {"batch_sizes": batch_sizes, "in_flight": in_flight, "rounds": args.rounds,
                      "round_seconds": args.round_seconds, "min_top_move_agreement": args.min_top_move_agreement,
                      "arms": arms}
            config_path = os.path.join(folder, "config.json")
            with open(config_path, "w") as handle:
                json.dump(config, handle, indent=2)
            json_out = os.path.join(args.out, f"{stem}.json")
            before = {b["pid"]: b["cpu_seconds"] for b in check_machine()}
            started = time.time()
            print(f"== {stem}", file=sys.stderr, flush=True)
            with open(os.path.join(args.out, f"{stem}.txt"), "w") as table:
                completed = subprocess.run([args.app, "--coreml-benchmark", config_path, "--json-out", json_out],
                                           stdin=subprocess.DEVNULL, stdout=table, stderr=sys.stderr)
            if completed.returncode != 0:
                raise BenchmarkError(f"--coreml-benchmark on {stem} exited {completed.returncode}")
            after = {b["pid"]: b["cpu_seconds"] for b in check_machine()}
            current["models"].append({
                "stem": stem, "seconds": round(time.time() - started, 1), "config": config,
                "bot_cpu_seconds_during": {str(pid): (round(after[pid] - cpu, 2) if pid in after else None)
                                           for pid, cpu in before.items()},
            })
            # Rewritten after each model, so a later failure keeps what finished.
            with open(machine_path, "w") as handle:
                json.dump(machine, handle, indent=2)
    except (BenchmarkError, dcm_arch.ArchitectureError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
