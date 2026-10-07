import Foundation

/// The kinds of beginning a game can show (challenge-log plan §3.6). Each
/// has one glyph and one label, in `LichessBotGameOriginStyle`.
enum LichessBotGameOriginCategory: String, Sendable, Hashable, CaseIterable {
    case incoming
    case challengeSheet
    case casualResendOffer
    case challengeQueue
    case matchmaking
    case matchmakingCasualResend
    case outgoingSenderNotRecorded
    case tournament
    case unknown
}

/// Why a game's origin is unknown — said, never left blank.
enum LichessBotOriginUnknownReason: Sendable, Hashable {
    /// The record says why (`undetermined`).
    case gap(LichessBotGameOriginGap)
    /// Created before the challenge log's first entry (or with no challenge
    /// log at all), and no challenge with its id in the protocol log.
    case playedBeforeOriginsWereRecorded
    /// Created while the challenge log existed, yet no origin was recorded
    /// and no challenge with its id is in the log: the origin write failed,
    /// or launch recovery filed the game without a session in this build.
    case notRecorded
}

/// Where a shown origin comes from.
enum LichessBotOriginBasis: Sendable, Hashable {
    /// The game's own record (its journal).
    case recorded
    /// The live challenge log's row for the game's id.
    case challengeLog
    /// The challenge rebuilt from the protocol log (§3.7), and how sure.
    case reconstructed(LichessBotReconstructionConfidence)
    case unknown(LichessBotOriginUnknownReason)
}

/// What a game shows for how it began: the one resolver every view reads
/// through `LichessBotController.originsByGameID` (§3.6). Views never
/// resolve anything themselves.
struct LichessBotGameOriginDisplay: Sendable, Hashable {
    let category: LichessBotGameOriginCategory
    /// Specifics for the help text and the detail view (for example
    /// "automatic pass, every free slot · challenge aBc123").
    let detail: String
    let basis: LichessBotOriginBasis

    /// The shown origin, in order:
    /// 1. the record's determined origin;
    /// 2. the live challenge log's row for the game id (covers a lost
    ///    journal write, and a record that says `undetermined` when the log
    ///    learned the answer later);
    /// 3. the reconstructed challenge;
    /// 4. the record's `undetermined` gap;
    /// 5. unknown, with a reason chosen by when the game was created:
    ///    before the challenge log's first entry (or with no log), "played
    ///    before origins were recorded"; otherwise "not recorded".
    static func resolve(
        gameID: String,
        createdAt: Date,
        recorded: LichessBotGameOrigin?,
        ledger: LichessBotChallengeLedger?,
        reconstruction: LichessBotReconstructedChallengeLookup?,
        liveLogFirstEntryAt: Date?
    ) -> LichessBotGameOriginDisplay {
        if let recorded, recorded.isDetermined {
            return display(recorded, basis: .recorded)
        }
        if let row = ledger?.row(challengeID: gameID), let fromLog = LichessBotGameOrigin.fromChallengeLog(gameID: gameID, row: row) {
            return display(fromLog, basis: .challengeLog)
        }
        if let reconstruction, let reconstructed = reconstructedDisplay(gameID: gameID, lookup: reconstruction) {
            return reconstructed
        }
        if case .undetermined(_, let gap)? = recorded {
            return LichessBotGameOriginDisplay(category: .unknown, detail: "challenge \(gameID)", basis: .unknown(.gap(gap)))
        }
        let reason: LichessBotOriginUnknownReason
        if let liveLogFirstEntryAt, createdAt >= liveLogFirstEntryAt {
            reason = .notRecorded
        } else {
            reason = .playedBeforeOriginsWereRecorded
        }
        return LichessBotGameOriginDisplay(category: .unknown, detail: "game \(gameID)", basis: .unknown(reason))
    }

    /// The display of a known origin.
    static func display(_ origin: LichessBotGameOrigin, basis: LichessBotOriginBasis) -> LichessBotGameOriginDisplay {
        switch origin {
        case .acceptedIncomingChallenge(let challengeID, let challengerID):
            return .init(category: .incoming, detail: "from \(challengerID) · challenge \(challengeID)", basis: basis)
        case .outgoingChallengeAccepted(let challengeID, let sender):
            let category: LichessBotGameOriginCategory
            let what: String
            switch sender {
            case .challengeSheet:
                category = .challengeSheet
                what = "the Challenge sheet"
            case .casualResendOffer:
                category = .casualResendOffer
                what = "Resend as Casual"
            case .challengeQueue:
                category = .challengeQueue
                what = "the challenge queue"
            case .matchmaking(let trigger, let fillMode):
                category = .matchmaking
                let triggerText = trigger == .automaticPass ? "automatic pass" : "Fill Open Slots"
                let fillText = fillMode == .everyFreeSlot ? "every free slot" : "only when idle"
                what = "\(triggerText), \(fillText)"
            case .matchmakingCasualResend:
                category = .matchmakingCasualResend
                what = "matchmaking's casual resend"
            }
            return .init(category: category, detail: "\(what) · challenge \(challengeID)", basis: basis)
        case .outgoingChallengeSenderNotRecorded(let challengeID):
            return .init(category: .outgoingSenderNotRecorded, detail: "DCM's challenge, sender not recorded · challenge \(challengeID)", basis: basis)
        case .tournament(let source, let tournamentID):
            return .init(category: .tournament, detail: "\(source.raw)\(tournamentID.map { " · tournament \($0)" } ?? "")", basis: basis)
        case .undetermined(let source, let gap):
            return .init(category: .unknown, detail: source.map { "source \($0.raw)" } ?? "no source", basis: .unknown(.gap(gap)))
        }
    }

    /// Step 3: the reconstructed challenge with the game's id, if any. The
    /// operator's sends are rebuilt without telling the Challenge sheet from
    /// Resend as Casual, so they show as the sheet, marked with their
    /// confidence (inferred when no other sender's line was beside them).
    private static func reconstructedDisplay(gameID: String, lookup: LichessBotReconstructedChallengeLookup) -> LichessBotGameOriginDisplay? {
        guard let row = lookup.row(challengeID: gameID) else { return nil }
        switch lookup.gameOrigin(gameID: gameID) {
        case .noReconstructedChallenge:
            return nil
        case .incoming:
            let challenger = row.challenge?.challenger.id ?? row.opponent?.id
            return .init(category: .incoming, detail: "\(challenger.map { "from \($0) · " } ?? "")challenge \(gameID)", basis: .reconstructed(.certain))
        case .outgoing(.attributed(let sender, let senderConfidence)):
            let category: LichessBotGameOriginCategory
            let what: String
            switch sender {
            case .byOperator:
                category = .challengeSheet
                what = "the operator (Challenge sheet or Resend as Casual)"
            case .challengeQueue:
                category = .challengeQueue
                what = "the challenge queue"
            case .matchmaking:
                category = .matchmaking
                what = "matchmaking"
            case .matchmakingCasualResend:
                category = .matchmakingCasualResend
                what = "matchmaking's casual resend"
            }
            return .init(category: category, detail: "\(what) · challenge \(gameID)", basis: .reconstructed(senderConfidence))
        case .outgoing(.ambiguousCompanion), .outgoing(.noSendLine):
            // The direction is certain (a raw challenge event); the
            // category says the sender is not.
            return .init(category: .outgoingSenderNotRecorded, detail: "DCM's challenge, sender not determined · challenge \(gameID)",
                         basis: .reconstructed(.certain))
        }
    }
}
