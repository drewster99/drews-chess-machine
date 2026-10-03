//
//  CommandLineHelp.swift
//
//  The top-level `--help` / `-h` request and the usage text it prints. The
//  same text is the banner the launch path's argument parser writes to stderr
//  after a usage error, so the help a user asks for and the help a mistake
//  shows can never drift apart.
//

import Foundation

enum CommandLineHelp {

    static let longFlag = "--help"
    static let shortFlag = "-h"

    /// Every spelling that asks for help. Matched as whole arguments only, so a
    /// stray `--helpme` or `-help` stays an unrecognized argument.
    static let flags: Set<String> = [longFlag, shortFlag]

    /// Which help a launch asked for.
    enum Request: Equatable {
        /// The full launch usage (`usageText`).
        case topLevelUsage
        /// `--derive-model`'s own help, generated from the operation catalog
        /// (`DeriveModelCLI.helpText`).
        case deriveModelUsage
    }

    /// The help `rawArgs` (the launch arguments without the executable path)
    /// ask for, or nil when they hold no help flag.
    ///
    /// Help wins over every other argument: a help flag beside a mode flag, or
    /// beside arguments that would otherwise be a usage error, still prints help
    /// and does nothing else, because someone who typed it wants to read what
    /// the command does, not have it run. The one mode with help of its own is
    /// `--derive-model`, whose operation flags come from a catalog the top-level
    /// usage only summarizes.
    static func request(in rawArgs: [String]) -> Request? {
        guard rawArgs.contains(where: { flags.contains($0) }) else { return nil }
        return rawArgs.contains(DeriveModelCLI.flag) ? .deriveModelUsage : .topLevelUsage
    }

    /// When `rawArgs` ask for help, writes it to stdout and exits 0 (never
    /// returns); otherwise returns without doing anything. Called before every
    /// other launch step, so a help request builds nothing, opens no window and
    /// writes no log.
    static func handleIfRequested(rawArgs: [String]) {
        guard let request = request(in: rawArgs) else { return }
        let text: String
        switch request {
        case .topLevelUsage:
            text = usageText
        case .deriveModelUsage:
            text = DeriveModelCLI.helpText
        }
        FileHandle.standardOutput.write(Data("\(text)\n".utf8))
        Darwin.exit(0)
    }

    /// The full launch usage: every mode and option. Printed to stdout for
    /// `--help` / `-h`, and to stderr after the launch parser's usage errors.
    static let usageText: String = """
    Usage: DrewsChessMachine [mode] [options]          (flags may be given in any order)

    Launch modes (pick at most one; default = open the normal training console GUI):
      --train                         Headless: auto-build a fresh network, start Play-and-Train,
                                      switch to the Candidate Test view.
      --playchess                     Open the GUI and immediately start a human-vs-network game.
                                      Opponent weights resolved like --uci (see --model below).

    Training options (with --train):
      --parameters <file>             JSON file of hyperparameter overrides (partial files allowed;
                                      only keys matching a known field are applied).
      --output <file>                 Write the JSON snapshot to <file> on training_time_limit expiry.
                                      Without this flag, the snapshot goes to stdout. Checked at launch:
                                      its folder must exist and be writable, and an existing <file>
                                      is refused (also for --replay-corpus and --train-vs-uci).
      --overwrite-output              Replace an existing --output file (only that file: if it is
                                      replaced or another appears during the run, it is left alone
                                      and the results go to <name>-2.<ext>, …, with an alarm).
      --training-time-limit <seconds> Seconds of Play-and-Train before the JSON snapshot is written
                                      and the process exits. Overrides any value in --parameters.
                                      Only honored under --train.
      --training-step-limit <steps>   Stop after the trainer completes this many SGD steps (snapshot
                                      + exit, same as the time limit; first budget to fire wins).
                                      Overrides any training_step_limit in --parameters.
      --start-model <path>            Load this saved model (.safetensors / .dcmmodel) as the starting
                                      champion instead of a fresh random init; the trainer forks from
                                      it. For controlled A/B runs from one identical starting net.
      --seed <n>                      Master seed (a whole number, 0 to 2^64-1, in digits only -- no sign)
                                      for every run this process starts; overrides random_seed_mode and
                                      random_seed. Also accepted by --replay-corpus and --train-vs-uci.
                                      Every run logs its seed on its [RUN] line, so an unseeded run can be
                                      repeated by passing it here.

    Opponent selection (with --playchess):
      --model <path>                  .safetensors or .dcmmodel weights to play against. Without it,
                                      the most recently saved session's trainer is used.

    Headless engine / tools (each runs without opening a window, then exits):
      --help, -h                      Print this usage to stdout and exit (with --derive-model: its own help).
      --uci [--model <path>]          Run as a UCI engine on stdin/stdout (cutechess, etc.). --model
                                      selects weights (default: latest saved session's trainer).
      --sweep [--sweep-sizes <csv>] [--sweep-seconds <n>]
                                      Batch-size throughput sweep; print the table and exit.
      --analyze-replay-buffer <path>  Analyze a replay_buffer.bin (or a .dcmsession dir); print JSON,
                                      human summary to stderr, and exit.
      --probe-model <path> [--probe-set 200|wide|both] [--probe-out <file>]
                         [--probe-positions-out <file>] [--probe-out-overwrite]
                                      Run the Lichess probe batteries against saved checkpoints
                                      (a weight file, one .dcmsession, or a directory of sessions);
                                      one JSON line per checkpoint x set, then exit.
                                      --probe-positions-out also writes one JSON line per position
                                      (rank, probability and NLL of the bookmove, top-1, entropy, ...).
                                      Output files must not already exist unless --probe-out-overwrite is
                                      given (replaces a regular file only, never a folder or link), and may
                                      never be a probed checkpoint or each other. Exits non-zero if an
                                      output cannot be written, or after the sweep if any checkpoint failed.
      --analyze-numerics <path> [--numerics-corpus <shard>] [--numerics-out <dir>] [--numerics-static-only]
                         [--policy-tail-precision fp32_from_pre_bn|mixed_final_projection]
                                      Numerics audit of a weight file or every weight file under a folder:
                                      fp32/bf16/fp16 fitness of weights and activations, head offsets,
                                      ties and cross-entropy. JSON per checkpoint (default: the analyses
                                      folder), one summary line each to stdout, then exit.
      --derive-model --from <model.safetensors> <operation> <value> [--group <index>]... --out <new.safetensors>
                                      Write a new model copied bit-exact from --from except the tensors the
                                      operation re-initializes, with a fresh ModelID, parent_model_id and a
                                      derivation_history record; shape-changing requests are refused.
                                      Operations: --set-se-beta-init glorot|zero (scale_and_bias groups),
                                      --set-rezero-alpha-init <float >= 0> (ReZero groups; also sets every
                                      ReZero alpha tensor of those groups to the value),
                                      --set-rezero-alpha-cap <float > 0> (ReZero groups; no tensors);
                                      --group <0-based index>, repeatable, narrows them. Full list, with
                                      what each rewrites: --derive-model --help.
      --show-default-parameters       Print every default training parameter as JSON and exit.
      --create-parameters-file [<path>] [--force]
                                      Write parameters.json + parameters.md and exit. <path> is a folder
                                      (existing, or ending in /) to write both into, or the .json file
                                      (the .md goes beside it); default ./parameters.json. --force replaces
                                      existing files, never a folder.

    Offline corpus replay & PGN import (each runs headless, then exits):
      --replay-corpus <dir|id>        Train on a fixed recorded game corpus — a path, or a bare corpus
                                      ID resolved under Corpora/ (repeatable, to mix corpora):
                                      no self-play, no arena, no promotion. Fills the replay buffer from
                                      the games and runs a step-locked SGD loop (K = batch / replay-ratio
                                      positions per step). Pair with --training-step-limit <n> OR
                                      --epochs <n> (default: 1 pass) and --parameters <file> to pin the
                                      hyperparameters. Pass --start-model <file> to continue training from a
                                      saved model (its embedded architecture is used). Ctrl-C stops cleanly
                                      and saves; press again to force-quit. The trainer model is saved every
                                      1000 steps and on exit/abort to a single rolling file (overwritten):
                                      --out-model <path>, else next to --start-model, else the app Models dir
                                      named after the corpus (<corpusID>-replay-latest.safetensors).
                                      Replay only reads the corpus: unsealed .open shards are skipped (and
                                      listed in a warning), never recovered or modified -- recover them
                                      with --validate-corpus <dir> --fix.
      --out-model <path>              Destination for the rolling trainer-model file (overwritten by this
                                      run's saves); a .safetensors extension is appended if you don't supply
                                      one (corpus replay only). Checked before training: never the
                                      --start-model itself, never anything but a regular file, never a name
                                      shaped like an enumerated checkpoint (<stem>-replay-step<N>,
                                      -vsuci-step<N>, -step<N>), and an existing file only when it holds the
                                      --start-model's model ID at the start model's own step (the rolling file
                                      of the state being continued). Any other existing file -- e.g. from an
                                      earlier run of the same command, which saved under its own model ID, or
                                      an earlier checkpoint of the same line -- refuses the run. Its name,
                                      and with --enumerate-checkpoints every step file's name, must leave
                                      room for the save's staging copy (a too-long name refuses the run).
                                      A save that fails for another reason is a warning once; the same
                                      save failing again at its next attempt stops the run.
      --overwrite-out-model           Use an --out-model the check above would refuse, replacing the file
                                      there (still never the --start-model, never a non-regular file).
      --enumerate-checkpoints         Also keep a copy of every save as <stem>-replay-step<N>.safetensors
                                      (<stem>-step<N> when the stem has no -replay-latest marker; see
                                      --train-vs-uci below for its step files). An exact resume of a recorded
                                      run writes <stem>-replay-seg<k>-step<N>, k its lineage segment index, so
                                      it can keep the stem. Never overwrites: step numbers restart in every
                                      segment, so the run refuses to start when its stem already has step
                                      files it could reach at its segment index -- give that run its own
                                      --out-model stem (e.g. <name>-resume2-replay-latest.safetensors).
      --epochs <n>                    Replay budget: number of full passes over the corpus (default 1 when
                                      no --training-step-limit is given), counted from the start of the
                                      run's lineage: a --resume-exact of a checkpoint saved in pass k
                                      (epoch k, 0-based) needs --epochs above k or a step limit, and is
                                      refused otherwise.
      --resume-exact                  (with --start-model) Continue the start model's run exactly instead of
                                      starting a new branch from its weights: fp32 master weights, optimizer
                                      velocity, the step clock and the warmup and LR/momentum cycle it
                                      trained under, its random streams, and its corpus position. The
                                      replay buffer is rebuilt by refeeding the games just before that
                                      position, and training continues from it. Needs a corpus-replay
                                      checkpoint written with exact trainer state, of this same corpus with
                                      unchanged shards (a changed corpus is always refused). Logs one
                                      [RESUME] EXACT or [RESUME] NOT EXACT: <items> line, and refuses to start
                                      when anything is missing unless --accept-inexact names every item.
      --accept-inexact <item,item,...>
                                      (with --resume-exact; also with --train-vs-uci) Resume although the
                                      named items cannot be continued: each starts fresh, and the segment
                                      is recorded as not exact. Items:
                                        \(ResumeGap.allCases.map(\.token).joined(separator: ", "))
      --gpu-capture-step <n> --gpu-capture-out <file.gputrace>
                                      (with --replay-corpus) Capture training step n as an Xcode GPU trace
                                      document for per-kernel profiling, then keep training. Needs the
                                      environment variable MTL_CAPTURE_ENABLED=1. The trace stores every
                                      GPU buffer the step touches: one batch-4096 step of a 512-channel-
                                      policy model passed 220 GB before filling the disk. Use a small
                                      model and batch size, and check free space first. Checked at launch:
                                      n within --training-step-limit, the trace's folder writable, nothing
                                      at the trace path. If the capture cannot start at step n, the run
                                      stops there, saves, and fails; a run that ends before step n warns.
      --policy-tail-precision fp32_from_pre_bn|mixed_final_projection
                                      (any mode: GUI, every CLI) Fixed for the process and recorded in its
                                      logs, trainer checkpoints, results.json and probe output. Where the policy head
                                      switches to fp32. Default mixed_final_projection: the pre-block and
                                      final projection run in the compute dtype and only the logits are
                                      widened. fp32_from_pre_bn widens from the pre-block's BatchNorm on
                                      (slower; slightly closer to an fp32 network on old pre-fix weights).
      --import-pgn <path>             Convert a .pgn / .pgn.zst (e.g. a Lichess monthly dump) into a
                                      corpus, then exit. .zst needs the `zstd` CLI on PATH; standard-start
                                      games only. Filters: --min-rating <elo> (both sides),
                                      --max-games <n>, --min-plies <n>,
                                      --time-control <bullet,blitz,rapid,classical>, --corpus-name <name>,
                                      --shard-soft-limit-mb <mb> (default 64),
                                      --max-storage <size> (e.g. 2GB; stops near that corpus body size),
                                      --import-threads <n> (default cores-2),
                                      --lenient (count parse failures instead of hard-failing on the first).
      --validate-corpus <dir|id>      Validate a corpus (repeatable): checks every sealed shard's integrity
                                      (front magic, corpus-ID stamp, trailer, whole-shard SHA-256, per-record
                                      CRC) and that corpus.json is consistent with the shards (per-source
                                      game/ply counts, sequence numbers, recording state), then prints a
                                      report and the true game/ply totals. Unsealed .open shards (a crashed
                                      recording/import, or one still running) are reported, untouched. Exit 0
                                      if valid, 1 if any problem remains. Add --fix to repair: recover each
                                      .open shard (truncate it to its last complete game and seal it, or
                                      remove it when it holds none; each is logged) and recompute stale
                                      per-source gamesAdded/pliesAdded from the shard trailers, rewriting
                                      corpus.json. Sealed shards are never modified. A shard a running
                                      recording or import still holds (its lock is held) is left untouched
                                      and reported, and corpus.json is then not rewritten; rerun --fix
                                      after that writer finishes. Add --quick to skip the
                                      SHA/CRC body pass (header/trailer counts only — fast, no integrity check).

    Training against UCI engines (headless, then exits):
      --train-vs-uci "cmd=<engine>;n=<instances>;go=<limit>;<UCI option>=<value>;…"   (repeatable)
                                      Play the live trainer against external UCI engines and train on the
                                      games (see documentation/UCI.md). Takes --parameters,
                                      --training-step-limit, --training-time-limit, --preset, --seed, --output.
      --start-model <model file | .dcmsession folder>
                                      Start from a model file or a session folder's trainer; with
                                      --resume-exact, continue its exact trainer state and random streams
                                      (and its replay buffer when the session saved one; otherwise
                                      --accept-inexact buffer is needed).
      --out-session-dir <folder>      Where session folders are saved (default: the app's Sessions folder).
                                      Every save is a new .dcmsession folder, written and verified like the
                                      GUI's: on the periodic_autosave_interval_sec cadence and at the end
                                      (…-vsuci-periodic / -vsuci-final / -vsuci-abort). The GUI does not
                                      load them; resume one with --start-model <folder> --resume-exact.
      --save-replay-buffer            Include replay_buffer.bin in every session save (several GB), so an
                                      exact resume restores the buffer instead of refilling it.
      --enumerate-checkpoints [--checkpoint-stem <path stem>]
                                      Also write the trainer file every 1000 steps and at the end as
                                      <stem>-vsuci-step<N>.safetensors (default stem: the --start-model
                                      file's own, next to it, else the run's model ID in Models/). Never
                                      overwrites; refuses a stem that already has step files it could reach.
      (--out-model and --overwrite-out-model do not apply to --train-vs-uci and are refused.)

    Self-play recording: set the `record_self_play_games` parameter (e.g. in a --parameters file) to
    record every kept self-play game into a corpus under Corpora/ during a --train run.

    Examples:
      # Cross-architecture A/B on identical games (build is fresh each run; average over N runs):
      DrewsChessMachine --replay-corpus <CorpusDir> --parameters frozen.json --training-step-limit 50000 --output archA.json

      # Same-architecture hyperparameter A/B from one identical starting net:
      DrewsChessMachine --train --start-model champ.safetensors --parameters lrA.json --training-step-limit 50000 --output lrA.json

      # Import a Lichess dump (rated blitz/rapid, 1800+, first 1M games):
      DrewsChessMachine --import-pgn lichess_2026-05.pgn.zst --min-rating 1800 --time-control blitz,rapid --max-games 1000000 --corpus-name lichess-2026-05

      # Replay a corpus for 3 full passes:
      DrewsChessMachine --replay-corpus <CorpusDir> --epochs 3 --parameters frozen.json

      # Paired copy of a fresh net with a zero-initialized SE beta path (every scale_and_bias group):
      DrewsChessMachine --derive-model --from fresh.safetensors --set-se-beta-init zero --out fresh-beta0.safetensors

      # Paired copy of a fresh net with zero-initialized ReZero (alpha tensors exactly 0) and cap 1.0:
      DrewsChessMachine --derive-model --from fresh.safetensors --set-rezero-alpha-init 0 --set-rezero-alpha-cap 1 --out fresh-rz0.safetensors
    """
}
