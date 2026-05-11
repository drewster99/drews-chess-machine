import SwiftUI

/// Transactional scratch state for `ArenaSettingsPopover`, lifted out of
/// `UpperContentView`.
///
/// Each editable field is a `String` (the raw `TextField` contents) plus a
/// matching `*Error` flag that drives the red invalid-input overlay. Editing
/// any field clears its own error via `didSet` — the next `save()` re-validates
/// the whole form transactionally. `save()` parses every field, writes the
/// valid values back to `TrainingParameters.shared` (logging each `[PARAM]`
/// transition), pushes the freshly-edited arena τ-schedule into the live
/// `samplingScheduleBox` via the injected `onAfterSave` closure, and dismisses
/// the popover only if every field parsed.
///
/// `formatDurationSpec` / `parseDurationSpec` / `maxConcurrency` are injected
/// (they live as `UpperContentView` statics) so this model carries no
/// dependency on the view.
@MainActor
@Observable
final class ArenaSettingsPopoverModel {
    /// Drives the chip's popover presentation. Replaces the old
    /// `showArenaPopover` `@State` on `UpperContentView`.
    var isPresented = false

    var gamesText = "" { didSet { gamesError = false } }
    var concurrencyText = "" { didSet { concurrencyError = false } }
    var intervalText = "" { didSet { intervalError = false } }
    var promoteThresholdText = "" { didSet { promoteThresholdError = false } }
    var tauStartText = "" { didSet { tauStartError = false } }
    var tauDecayText = "" { didSet { tauDecayError = false } }
    var tauFloorText = "" { didSet { tauFloorError = false } }

    private(set) var gamesError = false
    private(set) var concurrencyError = false
    private(set) var intervalError = false
    private(set) var promoteThresholdError = false
    private(set) var tauStartError = false
    private(set) var tauDecayError = false
    private(set) var tauFloorError = false

    private let maxConcurrency: Int
    private let formatDurationSpec: (Double) -> String
    private let parseDurationSpec: (String) -> Double?

    /// Called after a successful `save()` (and only then). Wired by
    /// `UpperContentView` to push the new arena τ-schedule into the live
    /// `samplingScheduleBox` so the next tournament picks up the new curve
    /// without waiting for a Play-and-Train restart.
    var onAfterSave: () -> Void = {}

    init(
        maxConcurrency: Int,
        formatDurationSpec: @escaping (Double) -> String,
        parseDurationSpec: @escaping (String) -> Double?
    ) {
        self.maxConcurrency = maxConcurrency
        self.formatDurationSpec = formatDurationSpec
        self.parseDurationSpec = parseDurationSpec
        seedFromParams()
    }

    /// Seed the edit fields from the live `trainingParams` snapshot. Called
    /// when the popover opens so the user always sees current values, even if
    /// a CLI / parameters-file override changed them since the last open.
    func seedFromParams() {
        let p = TrainingParameters.shared
        gamesText = String(p.arenaGamesPerTournament)
        concurrencyText = String(p.arenaConcurrency)
        intervalText = formatDurationSpec(p.arenaAutoIntervalSec)
        promoteThresholdText = String(format: "%.3f", p.arenaPromoteThreshold)
        tauStartText = String(format: "%.2f", p.arenaStartTau)
        tauDecayText = String(format: "%.3f", p.arenaTauDecayPerPly)
        tauFloorText = String(format: "%.2f", p.arenaTargetTau)
        gamesError = false
        concurrencyError = false
        intervalError = false
        promoteThresholdError = false
        tauStartError = false
        tauDecayError = false
        tauFloorError = false
    }

    func cancel() {
        isPresented = false
    }

    /// Validate every popover field against its parameter range and write
    /// valid values back to `trainingParams`. On any parse failure the field's
    /// red-overlay flag is set and the popover stays open. On full success the
    /// popover dismisses.
    func save() {
        let p = TrainingParameters.shared
        var anyError = false

        let parsedGames = Int(gamesText.trimmingCharacters(in: .whitespaces))
        if let g = parsedGames, g >= 4, g <= 10000 {
            gamesError = false
            if g != p.arenaGamesPerTournament {
                p.arenaGamesPerTournament = g
            }
        } else {
            gamesError = true
            anyError = true
        }

        let parsedConcurrency = Int(concurrencyText.trimmingCharacters(in: .whitespaces))
        if let c = parsedConcurrency, c >= 1, c <= maxConcurrency {
            concurrencyError = false
            if c != p.arenaConcurrency {
                p.arenaConcurrency = c
            }
        } else {
            concurrencyError = true
            anyError = true
        }

        if let secs = parseDurationSpec(intervalText), secs >= 60, secs <= 86400 {
            intervalError = false
            if secs != p.arenaAutoIntervalSec {
                p.arenaAutoIntervalSec = secs
            }
        } else {
            intervalError = true
            anyError = true
        }

        // Promote threshold — `[0.5, 1.0]` matches the parameter's
        // declared range. Lower bound 0.5 means "at-least-even" —
        // anything below would let the candidate displace the
        // champion on a coin-flip arena, so the parameter type
        // refuses to go there.
        if let v = Double(promoteThresholdText.trimmingCharacters(in: .whitespaces)),
           v >= 0.5, v.isFinite, v <= 1.0 {
            promoteThresholdError = false
            if abs(v - p.arenaPromoteThreshold) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] arenaPromoteThreshold: %.3f -> %.3f", p.arenaPromoteThreshold, v)
                )
                p.arenaPromoteThreshold = v
            }
        } else {
            promoteThresholdError = true
            anyError = true
        }

        // τ Start — same range as the inline stats-panel editor it
        // replaced: (0, 10].
        if let v = Double(tauStartText.trimmingCharacters(in: .whitespaces)),
           v > 0, v.isFinite, v <= 10 {
            tauStartError = false
            if abs(v - p.arenaStartTau) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] ar.startTau: %.3f -> %.3f", p.arenaStartTau, v)
                )
                p.arenaStartTau = v
            }
        } else {
            tauStartError = true
            anyError = true
        }

        // τ Decay — [0, 1].
        if let v = Double(tauDecayText.trimmingCharacters(in: .whitespaces)),
           v >= 0, v.isFinite, v <= 1 {
            tauDecayError = false
            if abs(v - p.arenaTauDecayPerPly) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] ar.decayPerPly: %.4f -> %.4f", p.arenaTauDecayPerPly, v)
                )
                p.arenaTauDecayPerPly = v
            }
        } else {
            tauDecayError = true
            anyError = true
        }

        // τ Floor — same range as Start: (0, 10].
        if let v = Double(tauFloorText.trimmingCharacters(in: .whitespaces)),
           v > 0, v.isFinite, v <= 10 {
            tauFloorError = false
            if abs(v - p.arenaTargetTau) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] ar.floorTau: %.3f -> %.3f", p.arenaTargetTau, v)
                )
                p.arenaTargetTau = v
            }
        } else {
            tauFloorError = true
            anyError = true
        }

        // Push the freshly-edited arena schedule into the live
        // `samplingScheduleBox` so the next arena tournament picks
        // up the new τ curve. Without this push the box keeps its
        // session-start snapshot and updated `trainingParams` values
        // don't take effect until the next Play-and-Train restart.
        onAfterSave()

        if !anyError { isPresented = false }
    }

    /// Live "reached at N plies" hint for the floor field, computed from the
    /// *current* parsed values of all three τ fields (so typing into any field
    /// updates it immediately). Returns a placeholder when any field is invalid
    /// or the math is degenerate.
    var tauReachedAtHint: String {
        guard let start = Double(tauStartText), start > 0,
              let decay = Double(tauDecayText), decay >= 0,
              let floor = Double(tauFloorText), floor > 0 else {
            return "(reached at —)"
        }
        guard decay > 0 else { return "(no decay; floor unreached)" }
        guard floor < start else { return "(reached at ply 0)" }
        let plies = Int(((start - floor) / decay).rounded(.up))
        return "(reached at ply \(plies))"
    }
}
