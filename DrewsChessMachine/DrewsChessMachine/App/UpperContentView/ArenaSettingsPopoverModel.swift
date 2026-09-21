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

    /// Which rule decides promotion. Bound directly to a `Picker` rather than
    /// parsed from text, so unlike every other field here it cannot be
    /// invalid and has no matching `*Error` flag.
    ///
    /// It is still transactional: the picker writes to this scratch value and
    /// `save()` is what reaches `TrainingParameters`, so Cancel discards a
    /// criterion change exactly as it discards a typed one.
    var promotionCriterion: ArenaPromotionCriterion = .scoreThreshold

    var gamesText = "" { didSet { gamesError = false } }
    var concurrencyText = "" { didSet { concurrencyError = false } }
    var intervalText = "" { didSet { intervalError = false } }
    var promoteThresholdText = "" { didSet { promoteThresholdError = false } }
    var tauStartText = "" { didSet { tauStartError = false } }
    var tauDecayText = "" { didSet { tauDecayError = false } }
    var tauFloorText = "" { didSet { tauFloorError = false } }
    var sprtElo0Text = "" { didSet { sprtElo0Error = false } }
    var sprtElo1Text = "" { didSet { sprtElo1Error = false } }
    var sprtAlphaText = "" { didSet { sprtAlphaError = false } }
    var sprtBetaText = "" { didSet { sprtBetaError = false } }
    var sprtMinGamesText = "" { didSet { sprtMinGamesError = false } }
    var sprtMaxGamesText = "" { didSet { sprtMaxGamesError = false } }

    private(set) var gamesError = false
    private(set) var concurrencyError = false
    private(set) var intervalError = false
    private(set) var promoteThresholdError = false
    private(set) var tauStartError = false
    private(set) var tauDecayError = false
    private(set) var tauFloorError = false
    private(set) var sprtElo0Error = false
    private(set) var sprtElo1Error = false
    private(set) var sprtAlphaError = false
    private(set) var sprtBetaError = false
    private(set) var sprtMinGamesError = false
    private(set) var sprtMaxGamesError = false
    /// Cross-field complaint that belongs to no single box — `elo1 > elo0`,
    /// `alpha + beta < 1`, `minGames <= maxGames`. Shown as a line under the
    /// SPRT group, because marking one of the two fields red would be picking
    /// a culprit arbitrarily.
    private(set) var sprtRelationError: String?

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
        promotionCriterion = p.arenaPromotionCriterion
        sprtElo0Text = Self.formatElo(p.arenaSPRTElo0)
        sprtElo1Text = Self.formatElo(p.arenaSPRTElo1)
        sprtAlphaText = String(format: "%.3f", p.arenaSPRTAlpha)
        sprtBetaText = String(format: "%.3f", p.arenaSPRTBeta)
        sprtMinGamesText = String(p.arenaSPRTMinGames)
        sprtMaxGamesText = String(p.arenaSPRTMaxGames)
        gamesError = false
        concurrencyError = false
        intervalError = false
        promoteThresholdError = false
        tauStartError = false
        tauDecayError = false
        tauFloorError = false
        sprtElo0Error = false
        sprtElo1Error = false
        sprtAlphaError = false
        sprtBetaError = false
        sprtMinGamesError = false
        sprtMaxGamesError = false
        sprtRelationError = nil
    }

    /// Elo hypotheses are whole numbers at their defaults (0 and 10) and read
    /// better that way; anything else keeps one decimal.
    private static func formatElo(_ v: Double) -> String {
        v == v.rounded() ? String(Int(v.rounded())) : String(format: "%.1f", v)
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

        // --- SPRT block ---
        //
        // Validated unconditionally, not just when SPRT is selected. The
        // fields stay editable under the score threshold (greyed but not
        // erased), and letting an invalid value save while it happens to be
        // inactive means the failure surfaces later, at the start of the
        // first SPRT arena, which is exactly where a statistics
        // configuration error is most expensive to discover.
        if !applySPRTFields(to: p) {
            anyError = true
        }

        // Push the freshly-edited arena schedule into the live
        // `samplingScheduleBox` so the next arena tournament picks
        // up the new τ curve. Without this push the box keeps its
        // session-start snapshot and updated `trainingParams` values
        // don't take effect until the next Play-and-Train restart.
        onAfterSave()

        if !anyError {
            // The criterion is written last, after every field it depends on
            // has been accepted. Flipping it to `.sprt` alongside a rejected
            // hypothesis would leave the next arena running the new rule with
            // the old numbers.
            if promotionCriterion != p.arenaPromotionCriterion {
                SessionLogger.shared.log(
                    "[PARAM] arenaPromotionCriterion: \(p.arenaPromotionCriterion.logToken) -> \(promotionCriterion.logToken)"
                )
                p.arenaPromotionCriterion = promotionCriterion
            }
            isPresented = false
        }
    }

    /// Parse, range-check and cross-check the six SPRT fields, writing them
    /// back on success. Returns false if anything failed.
    ///
    /// Per-field ranges mirror the parameter declarations. The cross-field
    /// constraints are checked afterwards by asking `ArenaSPRT.SPRTConfig` to
    /// construct itself — the same validator the arena uses — rather than
    /// restating the rules here, so the popover cannot drift out of agreement
    /// with what the arena will accept. A cross-field failure marks no single
    /// box red; it reports on its own line, because blaming one of the two
    /// values in a relation is arbitrary.
    private func applySPRTFields(to p: TrainingParameters) -> Bool {
        sprtRelationError = nil
        var ok = true

        func parseDouble(
            _ text: String,
            range: ClosedRange<Double>,
            error: (Bool) -> Void
        ) -> Double? {
            guard let v = Double(text.trimmingCharacters(in: .whitespaces)),
                  v.isFinite, range.contains(v) else {
                error(true)
                ok = false
                return nil
            }
            error(false)
            return v
        }

        func parseInt(
            _ text: String,
            range: ClosedRange<Int>,
            error: (Bool) -> Void
        ) -> Int? {
            guard let v = Int(text.trimmingCharacters(in: .whitespaces)),
                  range.contains(v) else {
                error(true)
                ok = false
                return nil
            }
            error(false)
            return v
        }

        let elo0 = parseDouble(sprtElo0Text, range: -50...50) { self.sprtElo0Error = $0 }
        let elo1 = parseDouble(sprtElo1Text, range: -50...50) { self.sprtElo1Error = $0 }
        let alpha = parseDouble(sprtAlphaText, range: 0.001...0.5) { self.sprtAlphaError = $0 }
        let beta = parseDouble(sprtBetaText, range: 0.001...0.5) { self.sprtBetaError = $0 }
        let minGames = parseInt(sprtMinGamesText, range: 2...10000) { self.sprtMinGamesError = $0 }
        let maxGames = parseInt(sprtMaxGamesText, range: 0...1_000_000) { self.sprtMaxGamesError = $0 }

        guard ok,
              let elo0, let elo1, let alpha, let beta, let minGames, let maxGames else {
            return false
        }

        do {
            _ = try ArenaSPRT.SPRTConfig(
                elo0: elo0, elo1: elo1, alpha: alpha, beta: beta,
                minGames: minGames, maxGames: maxGames
            )
        } catch let error as ArenaSPRT.ConfigError {
            sprtRelationError = error.description
            return false
        } catch {
            sprtRelationError = "\(error)"
            return false
        }

        func assign(_ new: Double, to current: Double, name: String, write: (Double) -> Void) {
            guard abs(new - current) > Double.ulpOfOne else { return }
            SessionLogger.shared.log(String(format: "[PARAM] %@: %.4f -> %.4f", name, current, new))
            write(new)
        }

        assign(elo0, to: p.arenaSPRTElo0, name: "arenaSPRTElo0") { p.arenaSPRTElo0 = $0 }
        assign(elo1, to: p.arenaSPRTElo1, name: "arenaSPRTElo1") { p.arenaSPRTElo1 = $0 }
        assign(alpha, to: p.arenaSPRTAlpha, name: "arenaSPRTAlpha") { p.arenaSPRTAlpha = $0 }
        assign(beta, to: p.arenaSPRTBeta, name: "arenaSPRTBeta") { p.arenaSPRTBeta = $0 }
        if minGames != p.arenaSPRTMinGames {
            SessionLogger.shared.log("[PARAM] arenaSPRTMinGames: \(p.arenaSPRTMinGames) -> \(minGames)")
            p.arenaSPRTMinGames = minGames
        }
        if maxGames != p.arenaSPRTMaxGames {
            SessionLogger.shared.log("[PARAM] arenaSPRTMaxGames: \(p.arenaSPRTMaxGames) -> \(maxGames)")
            p.arenaSPRTMaxGames = maxGames
        }
        return true
    }

    /// One-line summary of what the selected criterion will do, for the hint
    /// under the picker. Live — it reads the scratch fields, so it updates as
    /// the user types rather than after Save.
    var criterionHint: String {
        switch promotionCriterion {
        case .scoreThreshold:
            return "Fixed \(gamesText.isEmpty ? "N" : gamesText) games; promote if score ≥ \(promoteThresholdText)."
        case .sprt:
            return "Plays until the evidence decides. # of games and promote threshold are unused."
        }
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
