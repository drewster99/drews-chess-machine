//
//  BatchStatsLogLine.swift
//  DrewsChessMachine
//
//  The `[BATCH-STATS]` log line, written with the step lines rather than on
//  every batch-stats step.
//
//  Why: one `[BATCH-STATS]` line is 72–75 KB of JSON. Written on every
//  batch-stats step (every 10 steps by default) it was 99% of a corpus-replay
//  log (arm C's resumed segment: 41.1 MB of a 41.4 MB file). The trainer
//  still computes the summary on every batch-stats step — the GUI's `bufUniq`,
//  the GUI `results.json` `batch_stats`, `[SAMPLER]` and `[LEGAL-COST]` keep
//  that cadence — but the line itself rides the step line
//  (`TrainingStepLineSchedule`), whose fixed and time-based steps are all
//  batch-stats steps whenever `batch_stats_interval` is above 0.
//
//  One formatter for the three writers. The CLI runners observe every step
//  and log the summary of the very step they log (`ofTrainerStep:`). The GUI
//  ticker polls, so it logs the latest summary when its step differs from
//  the one it logged last (`lastLoggedStep:`) — "differs", not "is newer":
//  a promotion rewinds the trainer clock, after which every new summary's
//  step is below the last one logged until training passes it again, and a
//  "newer" rule would log nothing for that whole stretch.
//

import Foundation

enum BatchStatsLogLine {

    /// The line's tag, with its trailing space.
    static let tag = "[BATCH-STATS] "

    /// The CLI rule: log the summary only when it is this very step's.
    static func isDue(summaryStep: Int?, ofTrainerStep trainerStep: Int) -> Bool {
        summaryStep == trainerStep
    }

    /// The GUI rule: log the latest summary when its step differs from the
    /// last one logged (nil: none logged yet).
    static func isDue(summaryStep: Int?, lastLoggedStep: Int?) -> Bool {
        guard let summaryStep else { return false }
        return summaryStep != lastLoggedStep
    }

    /// The `[BATCH-STATS]` line for `summary` when it is `trainerStep`'s own
    /// summary, else nil.
    static func line(summary: ReplayBuffer.BatchStatsSummary?, ofTrainerStep trainerStep: Int) -> String? {
        guard let summary, isDue(summaryStep: summary.step, ofTrainerStep: trainerStep) else { return nil }
        return tag + summary.jsonLine()
    }

    /// The `[BATCH-STATS]` line for `summary` when its step differs from
    /// `lastLoggedStep`, else nil.
    static func line(summary: ReplayBuffer.BatchStatsSummary?, lastLoggedStep: Int?) -> String? {
        guard let summary, isDue(summaryStep: summary.step, lastLoggedStep: lastLoggedStep) else { return nil }
        return tag + summary.jsonLine()
    }
}
