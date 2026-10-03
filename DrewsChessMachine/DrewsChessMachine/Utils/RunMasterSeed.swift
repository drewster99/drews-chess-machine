//
//  RunMasterSeed.swift
//  DrewsChessMachine
//
//  Where a run's master seed comes from until the seed parameters exist.
//

import Foundation

/// The master seed a run derives its random streams from
/// (`DCMRandomStreams`), for the places that build a trainer.
///
/// Integration point for the seed parameters (`random_seed_mode` /
/// `random_seed` / `--seed`, determinism plan A3.2, phase P3): until they are
/// wired, every run draws its master seed from the system, once, where it
/// builds its trainer, and logs it — so the run's streams are reproducible from
/// the log even before the seed can be chosen. Phase P3 replaces this with the
/// parameter-resolved seed; nothing else changes at the call sites, which
/// already take their streams from the returned `DCMRandomStreams`.
enum RunMasterSeed {
    /// A system-drawn master seed for the run `context` names, logged as
    /// `[RNG] <context> master seed=<n> …`.
    static func systemDrawn(context: String) -> DCMRandomStreams {
        let seed = UInt64.random(in: UInt64.min ... UInt64.max)
        SessionLogger.shared.log(
            "[RNG] \(context) master seed=\(seed) (system-drawn: the random_seed parameter is not wired yet)"
        )
        return DCMRandomStreams(masterSeed: seed)
    }
}
