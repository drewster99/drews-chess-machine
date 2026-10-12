import Foundation

/// The words the model-in-use views show (issue #20,
/// documentation/plans-active/LICHESS_BOT_MODEL_IN_USE_PLAN.md), kept out of
/// the views so Settings ▸ Model and the Online panel say the same thing and
/// tests can read it without rendering.
@MainActor
enum LichessBotModelInUseText {
    /// What the bot says before it is online, in place of a model.
    static let notOnline = "Not online — the model loads when the bot goes online."

    /// The file the generation was read from, or its source's name when it
    /// has no file (champion, trainer snapshot, live trainer).
    static func name(_ generation: LichessBotGenerationInfo) -> String {
        guard let filePath = generation.filePath else {
            return generation.sourceKind.displayName
        }
        return URL(fileURLWithPath: filePath).lastPathComponent
    }

    /// "<model_id> · cum <n>": the run's cumulative trainer step where the
    /// file records its lineage, else the step the weights state, else the
    /// model ID alone.
    static func identity(_ generation: LichessBotGenerationInfo) -> String {
        if let cum = generation.lineage?.cumTrainerStep {
            return "\(generation.modelID) · cum \(cum)"
        }
        if let step = generation.trainingStep {
            return "\(generation.modelID) · step \(step)"
        }
        return generation.modelID
    }

    /// "file modified <time>" when the generation came from a file.
    static func fileModified(_ generation: LichessBotGenerationInfo, now: Date) -> String? {
        generation.fileModifiedAt.map { "file modified \(LichessBotModelTimeText.withAge($0, now: now))" }
    }

    /// How the generation came to be playing: "since going online <time>"
    /// for the session's first, else "switched <time> from <previous>".
    static func switchText(_ modelSwitch: LichessBotModelSwitch, now: Date) -> String {
        let when = LichessBotModelTimeText.withAge(modelSwitch.at, now: now)
        guard let from = modelSwitch.from else {
            return "since going online \(when)"
        }
        return "switched \(when) from \(name(from)) (\(from.modelID))"
    }

    /// The Online panel's line: "Playing <name> (<identity>) · file
    /// modified <time> · <switch>".
    static func onlineLine(_ modelSwitch: LichessBotModelSwitch, now: Date) -> String {
        let playing = modelSwitch.to
        var parts = ["Playing \(name(playing)) (\(identity(playing)))"]
        if let modified = fileModified(playing, now: now) {
            parts.append(modified)
        }
        parts.append(switchText(modelSwitch, now: now))
        return parts.joined(separator: " · ")
    }

    /// One line per older generation games in progress are still playing
    /// ("2 games in progress still on generation 3 (<model_id>)"), oldest
    /// first. A game's generation is its latest move's; a game with no move
    /// of ours yet isn't counted. Equality is the whole generation info:
    /// generation IDs restart each time the bot goes online, so a game
    /// resumed after a relaunch can carry an ID the current session reuses.
    static func olderGenerationLines(games: [LichessBotLiveGame], current: LichessBotGenerationInfo) -> [String] {
        var older: [(generation: LichessBotGenerationInfo, count: Int)] = []
        for game in games {
            guard let latestPly = game.generations.keys.max(), let generation = game.generations[latestPly] else {
                continue
            }
            if generation == current {
                continue
            }
            if let index = older.firstIndex(where: { $0.generation == generation }) {
                older[index].count += 1
            } else {
                older.append((generation, 1))
            }
        }
        return older
            .sorted { $0.generation.snapshotAt < $1.generation.snapshotAt }
            .map { entry in
                let games = entry.count == 1 ? "1 game" : "\(entry.count) games"
                return "\(games) in progress still on generation \(entry.generation.generationID) (\(entry.generation.modelID))"
            }
    }

    /// The followed lineage's newest-file mark while the bot plays that
    /// lineage: "(in use)" when the generation playing was built from that
    /// file, else "(next game once loaded)". Nil while offline or when the
    /// lineage shown isn't the one applied — nothing plays it then.
    /// `appliedLineage` is the running bot's followed lineage only while its
    /// applied source is Follow lineage (`appliedFollowedLineage(of:)`): a
    /// lineage left in the settings under another source plays nothing.
    static func newestFileMark(newest: LichessBotLineageFile, followed: LichessBotFollowedLineage,
                               modelInUse: LichessBotModelSwitch?, appliedLineage: LichessBotFollowedLineage?) -> String? {
        guard let modelInUse, appliedLineage == followed else {
            return nil
        }
        return modelInUse.to.plays(newest, of: followed) ? "(in use)" : "(next game once loaded)"
    }

    /// The lineage the running bot's applied settings follow, nil when their
    /// source is another one.
    static func appliedFollowedLineage(of model: LichessBotModelSettings) -> LichessBotFollowedLineage? {
        model.source == .followLineage ? model.followedLineage : nil
    }
}
