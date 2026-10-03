//
//  DCMRandomStreams.swift
//  DrewsChessMachine
//
//  Named sub-streams of one master seed (determinism plan, Part A3.1).
//

import CryptoKit
import Foundation

/// The run-level random streams, one name each. A stream's generator is
/// seeded from the master seed and the stream's *name*, never from its
/// position in a list, so adding or removing a stream never shifts another.
///
/// Weight initialization is not here: its per-tensor seeds derive from a
/// model's own init seed (recorded with the model), not from a run's master
/// seed, and are built by the initializer with the same
/// `DCMRandomStreams.childSeed` function.
enum DCMStream: Sendable, Equatable {
    /// The trainer's replay-buffer minibatch draws.
    case sampler
    /// The seed of the training graph's dropout state.
    case dropout
    /// One self-play game, by its game serial.
    case selfPlayGame(serial: Int)
    /// One arena game: the arena's index in the run and the game's index in it.
    case arenaGame(arenaIndex: Int, gameIndex: Int)
    /// One train-vs-UCI game, by its game serial.
    case trainVsUciGame(serial: Int)
    /// One probe invocation, by probe name and the trainer step it ran at.
    case probe(name: String, trainerStep: Int)
    /// The order of one corpus epoch (reserved: replay reads in file order today).
    case corpusOrder(epoch: Int)

    /// The stream's name: the string its seed is derived from. These
    /// spellings are part of every recorded seed and never change.
    var name: String {
        switch self {
        case .sampler:
            return "sampler"
        case .dropout:
            return "dropout"
        case let .selfPlayGame(serial):
            return "selfplay.game.\(serial)"
        case let .arenaGame(arenaIndex, gameIndex):
            return "arena.\(arenaIndex).game.\(gameIndex)"
        case let .trainVsUciGame(serial):
            return "vsuci.game.\(serial)"
        case let .probe(name, trainerStep):
            return "probe.\(name).\(trainerStep)"
        case let .corpusOrder(epoch):
            return "corpus.order.\(epoch)"
        }
    }
}

/// A run's master seed and the one derivation every seeded stream uses.
struct DCMRandomStreams: Sendable, Equatable {
    let masterSeed: UInt64

    /// Identifies the derivation (`childSeed`, `stableHash64`, the stream
    /// names and `DCMRandom`'s draws) in logs and run records, so a recorded
    /// seed is only ever replayed under the derivation that produced it.
    /// Changing any part of the derivation means a new identifier, never an
    /// edit of this one.
    static let derivationVersion = "v1"

    /// Name of the stream that walks the random game whose positions calibrate
    /// a freshly initialized network's batch-norm statistics. Part of a
    /// model's initialization, so its parent is the model's init seed, not a
    /// run's master seed.
    static let batchNormCalibrationStreamName = "init.bn_calibration"

    /// Name under the master seed of the init seed of a model a run builds
    /// fresh (a corpus-replay or train-vs-UCI run without `--start-model`),
    /// so `--seed` reproduces that model's initialization too.
    static let freshModelInitStreamName = "init"

    /// The init seed of a model this run builds fresh.
    var freshModelInitSeed: UInt64 {
        Self.childSeed(parent: masterSeed, name: Self.freshModelInitStreamName)
    }

    /// The generator for a model's batch-norm calibration walk.
    static func batchNormCalibrationGenerator(initSeed: UInt64) -> DCMRandom {
        DCMRandom(seed: childSeed(parent: initSeed, name: batchNormCalibrationStreamName))
    }

    /// A generator for `stream`, seeded by `childSeed(parent: masterSeed, name: stream.name)`.
    func generator(_ stream: DCMStream) -> DCMRandom {
        DCMRandom(seed: Self.childSeed(parent: masterSeed, name: stream.name))
    }

    /// The first eight bytes of SHA-256 of the name's UTF-8, read big-endian.
    /// Never Swift's `Hasher` or `hashValue`: those are randomly keyed per
    /// process, so every "seeded" run would differ. Reproducible elsewhere as
    /// `int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")`.
    static func stableHash64(_ name: String) -> UInt64 {
        let digest = SHA256.hash(data: Data(name.utf8))
        return digest.prefix(8).reduce(UInt64(0)) { value, byte in (value << 8) | UInt64(byte) }
    }

    /// The seed of the stream called `name` under `parent`:
    /// `splitmix64(parent ^ stableHash64(name))`. The only seed derivation in
    /// the app — run streams, per-game streams and per-tensor init seeds all
    /// use it — so changing it would invalidate every recorded seed.
    static func childSeed(parent: UInt64, name: String) -> UInt64 {
        DCMSplitMix64.mix(parent ^ stableHash64(name))
    }
}
