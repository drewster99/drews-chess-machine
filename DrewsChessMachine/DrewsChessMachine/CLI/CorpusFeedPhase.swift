//
//  CorpusFeedPhase.swift
//  DrewsChessMachine
//
//  Corpus replay feeds whole games so that, before trainer step j, at least
//  `base + j × perStep` positions have been fed (`perStep` = batch size /
//  replay ratio). Whole games overshoot that target, so how far the feed runs
//  ahead of the next step's target — the phase — depends on every earlier
//  game's length: it is state, not something a resume can recompute. A save
//  records it (`fed.corpus.feed_ahead_positions`); an exact resume rebuilds
//  the buffer by refeeding the games before the save point and then takes its
//  base from the saved phase, so the games fed before every later step are
//  the uninterrupted run's (determinism plan C1 #8).
//

import Foundation

/// The feed cadence of one corpus-replay segment.
struct CorpusFeedPhase: Equatable, Sendable {
    /// Positions fed by the time the segment's step 0 target is set: a fresh
    /// run's prefill, or a resume's refeed minus the saved feed-ahead.
    let base: Int
    /// Positions fed per trainer step.
    let perStep: Int

    /// A segment that starts its cadence at `fedPositions` (a fresh run, or a
    /// resume without a saved phase).
    static func starting(fedPositions: Int, perStep: Int) -> CorpusFeedPhase {
        CorpusFeedPhase(base: fedPositions, perStep: perStep)
    }

    /// A resumed segment continuing a saved phase: `reconstructedFedPositions`
    /// positions were refed to rebuild the buffer up to the save point, which
    /// stood `savedFeedAheadPositions` ahead of its next step's target.
    static func continuing(reconstructedFedPositions: Int, savedFeedAheadPositions: Int,
                           perStep: Int) -> CorpusFeedPhase {
        CorpusFeedPhase(base: reconstructedFedPositions - savedFeedAheadPositions, perStep: perStep)
    }

    /// Positions that must have been fed before step `step` (0-based) trains.
    func target(step: Int) -> Int {
        base + step * perStep
    }

    /// The phase to record at a save taken before step `step` trains, with
    /// `fedPositions` fed so far: how far the feed stands ahead of that
    /// step's target (negative while its feed is still owed).
    func feedAhead(fedPositions: Int, step: Int) -> Int {
        fedPositions - target(step: step)
    }
}
