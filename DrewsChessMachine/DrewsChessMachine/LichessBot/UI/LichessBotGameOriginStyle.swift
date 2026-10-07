import SwiftUI

/// The one place a game's origin becomes a glyph, words and a color
/// (challenge-log plan §3.9), so a glyph means the same thing in every view:
/// the All Games window, the Recent games list, the live picker, the tiles,
/// the game detail and the Challenge Log window.
///
/// A shown origin's **basis** marks how it is known. One inferred from the
/// absence of every other sender's line (`inferredFromAbsence`) gets a
/// trailing "≈" and the secondary color; an unknown one the secondary color.
/// The help text always says the long label, the detail and the basis, in
/// words, with the evidence for a rebuilt one.
enum LichessBotGameOriginStyle {

    /// Everything a view draws for one shown origin, or for a live game's
    /// origin that is not known yet (`presentation(of: nil)`).
    struct Presentation: Equatable {
        let systemImage: String
        let shortLabel: String
        let longLabel: String
        /// "≈" for an inferred origin, else empty.
        let marker: String
        /// Drawn in the secondary color: inferred, unknown, or not yet known.
        let isMuted: Bool
        let help: String

        var color: Color { isMuted ? .secondary : .primary }
        /// The short label with its marker.
        var markedShortLabel: String { shortLabel + marker }
        /// The long label with its marker.
        var markedLongLabel: String { longLabel + marker }
    }

    /// Every origin glyph takes this width, so the text beside glyphs of
    /// different widths lines up down a column.
    static let glyphWidth: CGFloat = 16

    static func systemImage(for category: LichessBotGameOriginCategory) -> String {
        switch category {
        case .incoming: return "arrow.down.left"
        case .challengeSheet: return "arrow.up.right"
        case .casualResendOffer: return "arrow.uturn.right"
        case .challengeQueue: return "list.bullet"
        case .matchmaking: return "wand.and.stars"
        case .matchmakingCasualResend: return "arrow.uturn.right.circle"
        case .outgoingSenderNotRecorded: return "arrow.up.right"
        case .tournament: return "trophy"
        case .unknown: return "questionmark"
        }
    }

    static func shortLabel(for category: LichessBotGameOriginCategory) -> String {
        switch category {
        case .incoming: return "Incoming"
        case .challengeSheet: return "Sheet"
        case .casualResendOffer: return "Resend offer"
        case .challengeQueue: return "Queue"
        case .matchmaking: return "Matchmaking"
        case .matchmakingCasualResend: return "Casual resend"
        case .outgoingSenderNotRecorded: return "Outgoing (sender not recorded)"
        case .tournament: return "Tournament"
        case .unknown: return "Unknown"
        }
    }

    static func longLabel(for category: LichessBotGameOriginCategory) -> String {
        switch category {
        case .incoming: return "Accepted a challenge someone sent DCM"
        case .challengeSheet: return "DCM's challenge, sent from the Challenge sheet"
        case .casualResendOffer: return "DCM's challenge, resent as casual by the operator (Resend as Casual)"
        case .challengeQueue: return "DCM's challenge, sent by the challenge queue"
        case .matchmaking: return "DCM's challenge, sent by matchmaking"
        case .matchmakingCasualResend: return "DCM's challenge, resent as casual by matchmaking"
        case .outgoingSenderNotRecorded: return "DCM's challenge; who sent it was not recorded"
        case .tournament: return "Paired by Lichess in a tournament"
        case .unknown: return "Unknown"
        }
    }

    /// The glyph for a live game whose origin is not decided yet. Distinct
    /// from `unknown`'s, which means "decided, and it couldn't be told".
    static let notYetKnownSystemImage = "ellipsis"
    static let notYetKnownLabel = "Not yet known"
    /// The live picker's suffix for a game whose origin is not decided yet.
    static let notYetKnownPickerSuffix = "origin not yet known"
    static let notYetKnownHelp = "How this game began is not known yet. It is decided when the challenge it came from is recorded, or, failing that, when its session ends."

    /// "≈" when the origin was inferred from the absence of every other
    /// sender's line.
    static func marker(for basis: LichessBotOriginBasis) -> String {
        if case .reconstructed(.inferredFromAbsence) = basis { return "≈" }
        return ""
    }

    static func isMuted(_ basis: LichessBotOriginBasis) -> Bool {
        switch basis {
        case .reconstructed(.inferredFromAbsence), .unknown: return true
        case .recorded, .challengeLog, .reconstructed(.certain), .reconstructed(.paired): return false
        }
    }

    /// How the shown origin is known, in words.
    static func basisText(_ basis: LichessBotOriginBasis) -> String {
        switch basis {
        case .recorded:
            return "Recorded in the game's own record when it started."
        case .challengeLog:
            return "From the challenge log's entry for the game's challenge."
        case .reconstructed(let confidence):
            return "Reconstructed from the protocol log (\(confidenceText(confidence)))."
        case .unknown(let reason):
            return unknownReasonText(reason)
        }
    }

    /// A rebuilt origin's confidence, with its evidence, in words.
    static func confidenceText(_ confidence: LichessBotReconstructionConfidence) -> String {
        switch confidence {
        case .certain:
            return "certain: a line carrying the challenge's id says so"
        case .paired:
            return "paired: a sender's line right after the 'challenge sent to' line, matched by player name"
        case .inferredFromAbsence:
            return "inferred: 'challenge sent to' with no matchmaking or queue line beside it, so the operator sent it"
        }
    }

    /// The one-word confidence the Challenge Log's Source column shows.
    static func confidenceWord(_ confidence: LichessBotReconstructionConfidence) -> String {
        switch confidence {
        case .certain: return "certain"
        case .paired: return "paired"
        case .inferredFromAbsence: return "inferred"
        }
    }

    /// Why an origin is unknown (plan §3.6 step 5), in words.
    static func unknownReasonText(_ reason: LichessBotOriginUnknownReason) -> String {
        switch reason {
        case .playedBeforeOriginsWereRecorded:
            return "Unknown — played before origins were recorded; no challenge with this id in the protocol log."
        case .notRecorded:
            return "Unknown — no origin recorded for this game, and no challenge with this id in the challenge log."
        case .gap(.noChallengeRecord):
            return "Unknown — no challenge with this game's id was in the challenge log when its session ended."
        case .gap(.challengeLogIncomplete):
            return "Unknown — the challenge log did not load completely when the game's session ended, so its challenge may be in the part that is missing."
        }
    }

    /// What a view draws for `display`; nil is a live game whose origin is
    /// not decided yet.
    static func presentation(of display: LichessBotGameOriginDisplay?) -> Presentation {
        guard let display else {
            return Presentation(systemImage: notYetKnownSystemImage, shortLabel: notYetKnownLabel, longLabel: notYetKnownLabel,
                                marker: "", isMuted: true, help: notYetKnownHelp)
        }
        let category = display.category
        let marker = marker(for: display.basis)
        // An unknown origin's long label is its reason ("Unknown — …"), so
        // the detail row says why; the help then doesn't repeat it.
        if category == .unknown, case .unknown(let reason) = display.basis {
            let reasonText = unknownReasonText(reason)
            return Presentation(systemImage: systemImage(for: category), shortLabel: shortLabel(for: category), longLabel: reasonText,
                                marker: marker, isMuted: true, help: [reasonText, display.detail].joined(separator: "\n"))
        }
        let help = [longLabel(for: category) + marker, display.detail, basisText(display.basis)].joined(separator: "\n")
        return Presentation(systemImage: systemImage(for: category), shortLabel: shortLabel(for: category), longLabel: longLabel(for: category),
                            marker: marker, isMuted: isMuted(display.basis), help: help)
    }
}
