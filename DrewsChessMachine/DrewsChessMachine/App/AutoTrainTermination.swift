//
//  AutoTrainTermination.swift
//  DrewsChessMachine
//
//  How a GUI `--train` run ends its process.
//

import Darwin
import Foundation

/// The one way a GUI `--train` run writes its results and ends the process.
///
/// Four paths can end such a run: the `training_time_limit` deadline, the
/// `training_step_limit` watcher, the legal-mass collapse detector, and an
/// early stop (SIGUSR1, SIGHUP, or AppKit asking the app to terminate,
/// through `EarlyStopCoordinator`). They are polled or delivered
/// independently, so two can fire within moments of each other; `claim()`
/// lets exactly one of them write the results and exit, and the others
/// return without touching the output. Without the claim, a second writer
/// would find the first one's file at the destination and put its results
/// in a numbered sibling.
///
/// The process ends with `Darwin._exit`, not `exit`: `exit` runs the C++
/// atexit handlers (CoreAnalytics' exit barrier among them) while an MPS
/// self-play or training command may still be inside `graph.run` on its
/// dispatch queue, and those handlers tear down global state `MPSGraphOSLog`
/// still reads — an `EXC_BAD_ACCESS` inside the MPSGraph run path. `_exit`
/// runs no handlers and flushes nothing, which is safe here because:
/// - the results file is written through `FileSafety` (staged, fully synced,
///   then published) before the exit;
/// - stdout results go through `FileHandle`, an unbuffered write with no
///   libc buffer to lose;
/// - `SessionLogger.log` only queues each line, so the logger is shut down
///   (its queue drained, the file synced and closed) right before the exit;
///   without that, the last lines — the write's outcome and the exit line —
///   could still be queued when the process ends, and be lost.
final class AutoTrainTermination: Sendable {
    let recorder: CliTrainingRecorder
    /// The `--output` destination checked before the run; nil writes the
    /// results to stdout.
    let resultsOutput: CliResultsOutput?
    private let claimed = SyncBox(false)

    init(recorder: CliTrainingRecorder, resultsOutput: CliResultsOutput?) {
        self.recorder = recorder
        self.resultsOutput = resultsOutput
    }

    /// True for the first caller only: that path writes and exits; every
    /// later caller must return without writing.
    func claim() -> Bool {
        claimed.mutate { alreadyClaimed in
            if alreadyClaimed { return false }
            alreadyClaimed = true
            return true
        }
    }

    /// Where the results went, as `writeResults` reports it.
    enum Outcome: Equatable {
        case file(URL)
        case standardOutput
        case failed(String)
    }

    /// Record `reason`, write the results to the run's destination and log
    /// both the attempt and its outcome through `log`. A failed write is
    /// logged and reported, never thrown: the run is ending either way, and
    /// the training it did is already saved or lost by then.
    ///
    /// `trigger` names what ended the run, e.g. `training_step_limit=500
    /// reached at steps=500`; `elapsed` is the run's elapsed seconds, written
    /// as the results' total training time.
    @discardableResult
    func writeResults(reason: CliTrainingRecorder.TerminationReason, trigger: String, elapsed: Double,
                      log: (String) -> Void) -> Outcome {
        let destination = resultsOutput?.url.path ?? "<stdout>"
        log("[APP] --train: \(trigger) (elapsed=\(String(format: "%.1f", elapsed))s); writing snapshot to \(destination)")
        recorder.setTerminationReason(reason)
        let counts = recorder.countsSnapshot()
        let outcome: Outcome
        let written: String
        do {
            if let resultsOutput {
                // Not always `resultsOutput.url`: a destination that changed
                // during the run sends the results to a numbered sibling.
                let url = try recorder.write(to: resultsOutput, totalTrainingSeconds: elapsed)
                outcome = .file(url)
                written = url.path
            } else {
                try recorder.writeJSONToStdout(totalTrainingSeconds: elapsed)
                outcome = .standardOutput
                written = "<stdout>"
            }
        } catch {
            log("[APP] --train: snapshot write FAILED for \(destination): \(error.localizedDescription)")
            return .failed(error.localizedDescription)
        }
        log("[APP] --train: wrote snapshot to \(written) (arenas=\(counts.arenas), stats=\(counts.stats), probes=\(counts.probes))")
        return outcome
    }

    /// `writeResults` through the session log, then shut the logger down and
    /// end the process with status 0 (see the type's doc for why `_exit`).
    /// The caller must hold the claim.
    func writeResultsAndExit(reason: CliTrainingRecorder.TerminationReason, trigger: String,
                             elapsed: Double) -> Never {
        writeResults(reason: reason, trigger: trigger, elapsed: elapsed, log: SessionLogger.shared.log)
        SessionLogger.shared.log("[APP] --train: exiting process (termination_reason=\(reason.rawValue))")
        SessionLogger.shared.shutdown()
        Darwin._exit(0)
    }
}
