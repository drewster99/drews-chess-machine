import Foundation

/// A game event the bot can comment on in chat (chat-events plan §2). Each
/// trigger is detected from the board and the value head's win probability,
/// both already computed while playing, so commentary costs no extra work.
///
/// Tone is fixed by owner decision (2026-10-15): neutral or self-deprecating
/// only, never gloating on the opponent's mistakes. Every default template
/// follows that, and every rule ships disabled.
enum LichessBotChatEventTrigger: String, Sendable, Codable, CaseIterable, Identifiable, Hashable {
    /// Our queen left the board (we lost it, or traded it).
    case weLostQueen
    /// The opponent's queen left the board (we took it, or traded for it).
    case weWonQueen
    /// A check appears on the board within the opening (given or received).
    case earlyCheck
    /// Our win probability reaches a high-water threshold.
    case winningBig
    /// Our win probability falls to a low threshold.
    case losingBig
    /// Our win probability swings by a large amount between our moves.
    case bigSwing
    /// A pawn promoted.
    case promotion
    /// The game passed a given full-move number.
    case passedMove

    var id: String { rawValue }

    /// A short label for the Settings row. Each names whose event it is
    /// (ours, the opponent's, or either side's), because a bare "a check" or
    /// "a promotion" left the owner unsure which side fires it.
    var title: String {
        switch self {
        case .weLostQueen: return "Our queen is captured"
        case .weWonQueen: return "We capture their queen"
        case .earlyCheck: return "An early check by either side"
        case .winningBig: return "Our win chance rises to a level"
        case .losingBig: return "Our win chance falls to a level"
        case .bigSwing: return "Our win chance swings, either way"
        case .promotion: return "A promotion by either side"
        case .passedMove: return "The game reaches a move number"
        }
    }

    /// One line under the title saying exactly when the rule fires, matching
    /// `LichessBotChatEventEvaluator`'s condition for the trigger. Keep the
    /// two in step: this is the only place a user learns, for example, that
    /// a queen trade fires both queen rules.
    var firingDescription: String {
        switch self {
        case .weLostQueen:
            return "Fires when the opponent takes our queen, trades included."
        case .weWonQueen:
            return "Fires when we take the opponent's queen, trades included, whether or not we come out ahead."
        case .earlyCheck:
            return "Fires when either side gives check within the first plies set here."
        case .winningBig:
            return "Fires on our move whose win-chance reading crosses up to this level (the network's own estimate)."
        case .losingBig:
            return "Fires on our move whose win-chance reading crosses down to this level (the network's own estimate)."
        case .bigSwing:
            return "Fires when our win-chance reading moves by at least this many points from our previous move, up or down."
        case .promotion:
            return "Fires when either side promotes a pawn, to any piece."
        case .passedMove:
            return "Fires when the game reaches this full-move number."
        }
    }

    /// The numeric parameter this trigger uses, with the Settings label, the
    /// allowed range and the default. `nil` for triggers with no parameter
    /// (queen captures, promotion).
    struct Parameter: Sendable, Equatable {
        let label: String
        let range: ClosedRange<Int>
        let defaultValue: Int
        /// A unit shown after the field, e.g. "%", "plies", "move".
        let unit: String
    }

    var parameter: Parameter? {
        switch self {
        case .weLostQueen, .weWonQueen, .promotion:
            return nil
        case .earlyCheck:
            return Parameter(label: "within the first", range: 2...80, defaultValue: 12, unit: "plies")
        case .winningBig:
            return Parameter(label: "win chance at least", range: 50...99, defaultValue: 90, unit: "%")
        case .losingBig:
            return Parameter(label: "win chance at most", range: 1...50, defaultValue: 10, unit: "%")
        case .bigSwing:
            return Parameter(label: "swing of at least", range: 5...90, defaultValue: 30, unit: "points")
        case .passedMove:
            return Parameter(label: "past move", range: 20...300, defaultValue: 80, unit: "")
        }
    }

    var defaultThreshold: Int { parameter?.defaultValue ?? 0 }

    /// Higher wins when several triggers fire on the same ply: the most
    /// salient event is the one that gets the single message that ply's spam
    /// guards allow.
    var priority: Int {
        switch self {
        case .losingBig: return 80
        case .winningBig: return 75
        case .weLostQueen: return 70
        case .weWonQueen: return 65
        case .bigSwing: return 50
        case .promotion: return 40
        case .earlyCheck: return 30
        case .passedMove: return 10
        }
    }

    /// Default message variants. Neutral or self-deprecating; one is chosen
    /// at random when the rule fires. Placeholders: `{opponent}`, `{winPct}`,
    /// `{move}`, `{ply}`.
    var defaultTemplates: [String] {
        switch self {
        case .weLostQueen: return ["Ouch, there goes my queen.", "Lost my queen — this just got harder."]
        case .weWonQueen: return ["Their queen is off the board.", "No more queen on that side."]
        case .earlyCheck: return ["A check already, on move {move}.", "Checks this early — lively game."]
        case .winningBig: return ["Feeling good about this one ({winPct}% by my read).", "Things are looking up for me here."]
        case .losingBig: return ["This is slipping away from me.", "I'm in real trouble here, {opponent}."]
        case .bigSwing: return ["Big momentum shift that move.", "Well, that changed things."]
        case .promotion: return ["A pawn's made it to the end.", "Promotion on the board."]
        case .passedMove: return ["We're well past move {move} — long game, {opponent}!", "Still going on move {move}."]
        }
    }
}

/// One configurable rule: a trigger, which room it posts to, the odds it
/// fires when its condition is met, the numeric threshold the trigger reads,
/// and one or more message variants. Each rule fires at most once per game.
///
/// Decoding tolerates saves written before any field existed: every field
/// but `trigger` falls back to its default (owner rule: saved settings
/// survive changes). Rules live inside an array, which the settings store
/// takes whole, so per-field carry-forward happens here rather than in the
/// store's object overlay.
struct LichessBotChatEventRule: Sendable, Equatable, Codable, Identifiable {
    var trigger: LichessBotChatEventTrigger
    var enabled: Bool
    var room: LichessBotChatRoom
    /// 0…100; the chance the message is sent once the condition is met and
    /// the spam guards allow it.
    var chancePercent: Int
    /// Meaning depends on `trigger.parameter`; ignored for triggers without
    /// one.
    var threshold: Int
    /// At least one; a random variant is chosen each time.
    var templates: [String]

    var id: LichessBotChatEventTrigger { trigger }

    init(
        trigger: LichessBotChatEventTrigger,
        enabled: Bool = false,
        room: LichessBotChatRoom = .spectator,
        chancePercent: Int = 100,
        threshold: Int? = nil,
        templates: [String]? = nil
    ) {
        self.trigger = trigger
        self.enabled = enabled
        self.room = room
        self.chancePercent = chancePercent
        self.threshold = threshold ?? trigger.defaultThreshold
        let supplied = templates ?? trigger.defaultTemplates
        self.templates = supplied.isEmpty ? trigger.defaultTemplates : supplied
    }

    private enum CodingKeys: String, CodingKey {
        case trigger, enabled, room, chancePercent, threshold, templates
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let trigger = try c.decode(LichessBotChatEventTrigger.self, forKey: .trigger)
        self.init(
            trigger: trigger,
            enabled: (try? c.decodeIfPresent(Bool.self, forKey: .enabled)) ?? nil ?? false,
            room: (try? c.decodeIfPresent(LichessBotChatRoom.self, forKey: .room)) ?? nil ?? .spectator,
            chancePercent: (try? c.decodeIfPresent(Int.self, forKey: .chancePercent)) ?? nil ?? 100,
            threshold: (try? c.decodeIfPresent(Int.self, forKey: .threshold)) ?? nil,
            templates: (try? c.decodeIfPresent([String].self, forKey: .templates)) ?? nil
        )
    }
}

/// The event-message feature's settings: a master switch, the spam guards
/// shared across rules, and one rule per trigger.
struct LichessBotChatEventSettings: Sendable, Equatable, Codable {
    var enabled = false
    /// Nothing fires while either player's clock is below this (owner:
    /// 4 minutes, to stay well clear of time scrambles).
    var minClockSecondsEitherSide = 240
    var maxMessagesPerGame = 3
    var minSecondsBetweenMessages = 20
    var minPliesBetweenMessages = 8
    /// One entry per trigger, in `CaseIterable` order. Always normalized so
    /// every known trigger is present exactly once.
    var rules: [LichessBotChatEventRule] = LichessBotChatEventSettings.starterRules

    /// Every trigger, disabled, posting to the spectator room (the owner's
    /// chosen default room for event messages).
    static var starterRules: [LichessBotChatEventRule] {
        LichessBotChatEventTrigger.allCases.map { LichessBotChatEventRule(trigger: $0) }
    }

    /// Saved rules merged onto the full starter list by trigger: every known
    /// trigger appears once (new triggers added in a later build show up
    /// disabled), user edits are kept, and any unknown trigger is dropped.
    static func normalized(_ saved: [LichessBotChatEventRule]) -> [LichessBotChatEventRule] {
        let byTrigger = Dictionary(saved.map { ($0.trigger, $0) }, uniquingKeysWith: { first, _ in first })
        return LichessBotChatEventTrigger.allCases.map { byTrigger[$0] ?? LichessBotChatEventRule(trigger: $0) }
    }

    init() {}

    private enum CodingKeys: String, CodingKey {
        case enabled, minClockSecondsEitherSide, maxMessagesPerGame
        case minSecondsBetweenMessages, minPliesBetweenMessages, rules
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        var defaults = LichessBotChatEventSettings()
        defaults.enabled = (try? c.decodeIfPresent(Bool.self, forKey: .enabled)) ?? nil ?? defaults.enabled
        defaults.minClockSecondsEitherSide = (try? c.decodeIfPresent(Int.self, forKey: .minClockSecondsEitherSide)) ?? nil ?? defaults.minClockSecondsEitherSide
        defaults.maxMessagesPerGame = (try? c.decodeIfPresent(Int.self, forKey: .maxMessagesPerGame)) ?? nil ?? defaults.maxMessagesPerGame
        defaults.minSecondsBetweenMessages = (try? c.decodeIfPresent(Int.self, forKey: .minSecondsBetweenMessages)) ?? nil ?? defaults.minSecondsBetweenMessages
        defaults.minPliesBetweenMessages = (try? c.decodeIfPresent(Int.self, forKey: .minPliesBetweenMessages)) ?? nil ?? defaults.minPliesBetweenMessages
        let savedRules = (try? c.decodeIfPresent([LichessBotChatEventRule].self, forKey: .rules)) ?? nil
        defaults.rules = LichessBotChatEventSettings.normalized(savedRules ?? LichessBotChatEventSettings.starterRules)
        self = defaults
    }
}
