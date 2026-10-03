import Foundation

/// Whether a run's master seed is the one in the settings (`seeded`) or drawn
/// fresh when the run starts (`unseeded`).
///
/// Both modes use the same seeded machinery — there is exactly one code path.
/// "Unseeded" means the seed was chosen for you: it is still drawn, logged on
/// the `[RUN]` line and recorded, so any unseeded run can be replayed by
/// passing its seed back with `--seed` or by switching to `seeded` with that
/// value. See `RunRandomSeed` and the determinism plan, Part A3.
///
/// **Why it is stored as an `Int`.** It follows `PolicyLabelSmoothingMode`
/// and `ArenaPromotionCriterion`: the raw integer is confined to the
/// persistence boundary (`random_seed_mode` in `parameters.json` and
/// `UserDefaults`); every use site reads this type. `parameterRawValueRange`
/// is pinned to the parameter's declared range by test.
public enum RandomSeedMode: Int, CaseIterable, Sendable, Identifiable {
    /// A seed is drawn from the system at run start.
    case unseeded = 0
    /// The configured `random_seed` is the master seed.
    case seeded = 1

    public var id: Int { rawValue }

    /// Label for the settings picker.
    public var displayName: String {
        switch self {
        case .unseeded: return "Draw a seed"
        case .seeded: return "Use this seed"
        }
    }

    /// Short, stable token for logs and `results.json`.
    public var logToken: String {
        switch self {
        case .unseeded: return "unseeded"
        case .seeded: return "seeded"
        }
    }

    /// Closed range of raw values this enum covers, for pinning against the
    /// `random_seed_mode` parameter definition.
    public static var parameterRawValueRange: ClosedRange<Int> {
        let raws = allCases.map(\.rawValue)
        guard let low = raws.min(), let high = raws.max() else {
            preconditionFailure("RandomSeedMode must have at least one case")
        }
        return low...high
    }

    /// Converts a persisted raw value. Every path that reaches this has
    /// checked the value against the parameter's declared range (pinned to
    /// `parameterRawValueRange` by test), so an unrepresentable value is a
    /// programmer error and traps rather than silently picking a mode.
    public init(persistedRawValue raw: Int) {
        guard let mode = RandomSeedMode(rawValue: raw) else {
            preconditionFailure(
                "random_seed_mode raw value \(raw) has no RandomSeedMode case; "
                + "the parameter's declared range and \(RandomSeedMode.self) have drifted apart"
            )
        }
        self = mode
    }
}

/// The master seed one run uses, and where it came from — resolved once at
/// run start by every training path (GUI Play-and-Train, corpus replay,
/// train-vs-UCI) through `resolve`, the one resolver.
struct RunRandomSeed: Sendable, Equatable {

    /// Where the master seed came from.
    enum Origin: Sendable, Equatable {
        /// `random_seed` from the settings, with `random_seed_mode` = seeded.
        case configured
        /// `--seed` on the command line, which overrides both settings for
        /// that process.
        case commandLine
        /// Drawn from the system because `random_seed_mode` = unseeded.
        case drawn
        /// Carried over from the checkpoint an exact resume continues: the
        /// resumed segment is the same run, so it keeps the run's seed
        /// (determinism plan C4). `firstSegment` is how the run's first
        /// segment got it.
        case inherited(firstSegment: LineageRecord.RunStreams.SeedOrigin)
    }

    let masterSeed: UInt64
    let origin: Origin
    /// The `random_seed` setting when the seed was resolved — the master
    /// seed when `origin` is `.configured`, ignored (and said so in the log)
    /// otherwise.
    let configuredSeed: UInt64

    /// The run's named streams.
    var streams: DCMRandomStreams { DCMRandomStreams(masterSeed: masterSeed) }

    /// The mode the run effectively ran in: a `--seed` run is seeded.
    var effectiveMode: RandomSeedMode {
        switch origin {
        case .configured, .commandLine, .inherited(firstSegment: .configured): return .seeded
        case .drawn, .inherited(firstSegment: .drawn): return .unseeded
        }
    }

    /// How the run's seed is recorded in its lineage record: configured
    /// (settings or `--seed`) or drawn, as the run's first segment got it.
    var recordedOrigin: LineageRecord.RunStreams.SeedOrigin {
        switch origin {
        case .configured, .commandLine: return .configured
        case .drawn: return .drawn
        case .inherited(let firstSegment): return firstSegment
        }
    }

    /// The seed's fields of the `[RUN]` provenance line
    /// (`RunProvenanceLine`): the master seed, how it was chosen, and the
    /// stream-derivation version.
    var provenanceFields: String {
        let modeText: String
        switch origin {
        case .configured: modeText = "seeded"
        case .commandLine: modeText = "seeded(--seed)"
        case .drawn: modeText = "unseeded(drawn)"
        case .inherited(let firstSegment): modeText = "resumed(\(firstSegment.rawValue))"
        }
        return "seed=\(masterSeed) mode=\(modeText) derivation=\(DCMRandomStreams.derivationVersion)"
    }

    /// The record of this run's streams at a save: the seed plus the stream
    /// positions the caller read in the save's consistent cut.
    func runStreams(samplerState: DCMRandom, dropoutStreamState: DCMRandom,
                    nextGameSerial: Int?, arenasStarted: Int?, opponentGameIndices: [Int]?) -> LineageRecord.RunStreams {
        LineageRecord.RunStreams(
            masterSeed: masterSeed,
            seedOrigin: recordedOrigin,
            streamDerivation: DCMRandomStreams.derivationVersion,
            samplerState: samplerState,
            dropoutStreamState: dropoutStreamState,
            nextGameSerial: nextGameSerial,
            arenasStarted: arenasStarted,
            opponentGameIndices: opponentGameIndices)
    }

    /// The seed of the run an exact resume continues. A `--seed` naming a
    /// different seed contradicts the resume and throws; the settings'
    /// `random_seed` is reported as not used.
    static func inherited(from streams: LineageRecord.RunStreams,
                          configuredSeed: UInt64,
                          commandLineSeed: UInt64?) throws -> RunRandomSeed {
        if let commandLineSeed, commandLineSeed != streams.masterSeed {
            throw RunRandomSeedError.resumeSeedConflict(commandLine: commandLineSeed, checkpoint: streams.masterSeed)
        }
        guard streams.streamDerivation == DCMRandomStreams.derivationVersion else {
            throw RunRandomSeedError.resumeDerivationMismatch(checkpoint: streams.streamDerivation,
                                                               running: DCMRandomStreams.derivationVersion)
        }
        return RunRandomSeed(masterSeed: streams.masterSeed,
                             origin: .inherited(firstSegment: streams.seedOrigin),
                             configuredSeed: configuredSeed)
    }

    /// The seed on its own as a `[RUN]` line. A training path does not log
    /// it: it logs `parameterNotes`, then the full provenance line, which
    /// carries the same fields.
    var logLine: String { "[RUN] " + provenanceFields }

    /// What happened to the configured seed when it was not used, logged by
    /// every path before its `[RUN]` line.
    var parameterNotes: [String] {
        switch origin {
        case .configured:
            return []
        case .commandLine:
            return ["[PARAM] random_seed from --seed: \(masterSeed) (overrides random_seed_mode and random_seed=\(configuredSeed) for this process)"]
        case .drawn:
            return ["[PARAM] random_seed=\(configuredSeed) ignored: random_seed_mode=unseeded draws the run seed"]
        case .inherited:
            return ["[PARAM] random_seed_mode and random_seed=\(configuredSeed) not used: an exact resume keeps the run's seed"]
        }
    }

    /// `parameterNotes` followed by the seed's own `logLine`.
    var logLines: [String] { parameterNotes + [logLine] }

    /// Decide the run's master seed. A command-line seed wins over the
    /// settings; otherwise `seeded` uses the configured seed and `unseeded`
    /// draws one with `drawSeed` (production passes `systemDrawnSeed`; tests
    /// pass a fixed function).
    static func resolve(
        mode: RandomSeedMode,
        configuredSeed: UInt64,
        commandLineSeed: UInt64?,
        drawSeed: () -> UInt64
    ) -> RunRandomSeed {
        if let commandLineSeed {
            return RunRandomSeed(masterSeed: commandLineSeed, origin: .commandLine, configuredSeed: configuredSeed)
        }
        switch mode {
        case .seeded:
            return RunRandomSeed(masterSeed: configuredSeed, origin: .configured, configuredSeed: configuredSeed)
        case .unseeded:
            return RunRandomSeed(masterSeed: drawSeed(), origin: .drawn, configuredSeed: configuredSeed)
        }
    }

    /// A master seed from the system's random source.
    static func systemDrawnSeed() -> UInt64 {
        UInt64.random(in: UInt64.min...UInt64.max)
    }

    /// Parse a `--seed` value: a decimal UInt64, digits only
    /// (`UInt64(strictDecimal:)` — no sign, so `+5` and `-0` are refused,
    /// as `--init-seed` refuses them). Throws with the text it could not
    /// read.
    static func parseCommandLineSeed(_ text: String) throws -> UInt64 {
        guard let seed = UInt64(strictDecimal: text) else {
            throw RunRandomSeedError.invalidCommandLineSeed(text)
        }
        return seed
    }
}

enum RunRandomSeedError: Error, Equatable, LocalizedError {
    case invalidCommandLineSeed(String)
    case resumeSeedConflict(commandLine: UInt64, checkpoint: UInt64)
    case resumeDerivationMismatch(checkpoint: String, running: String)

    var errorDescription: String? {
        switch self {
        case .invalidCommandLineSeed(let text):
            return "--seed needs a whole number from 0 to \(UInt64.max); got \"\(text)\""
        case .resumeSeedConflict(let commandLine, let checkpoint):
            return "--seed \(commandLine) contradicts the exact resume: the checkpoint's run seed is \(checkpoint). "
                + "Drop --seed to continue the run, or start a new branch without --resume-exact."
        case .resumeDerivationMismatch(let checkpoint, let running):
            return "the checkpoint's streams were named under derivation \(checkpoint), this build uses \(running); "
                + "its stream positions cannot be continued"
        }
    }
}
