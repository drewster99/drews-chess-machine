import Foundation

/// A chat command DCM answers (plan §12.5a). Every one is read-only and
/// safe in both rooms: none reveals the value head's evaluation or the
/// policy for the current position.
enum LichessBotChatCommand: String, CaseIterable, Sendable {
    case help, name, about, motor, cpu, gpu, ram
}

/// What a reply may say about the game's model and this Mac.
struct LichessBotChatCommandContext: Sendable {
    /// Our Lichess username as displayed (not the lowercased id).
    let ourUsername: String
    let modelID: String
    let trainingStep: Int?
    let build: Int
    let hardware: HardwareInfo
}

enum LichessBotChatCommandError: LocalizedError, Equatable {
    case replyTooLong(command: LichessBotChatCommand, length: Int)

    var errorDescription: String? {
        switch self {
        case .replyTooLong(let command, let length):
            return "the !\(command.rawValue) reply would be \(length) UTF-16 units; Lichess allows \(LichessBotChat.maximumLength)"
        }
    }
}

/// Parsing and reply text for chat commands (plan §12.5a). Pure, so every
/// rule is unit-tested; the game session only decides *whether* to send.
///
/// Replies stay within Lichess's cleanup rules (see
/// `documentation/research/lichess-bot/chat-commands.md`): no emoji or chess
/// symbols (stripped), nothing link-like (the whole message is dropped
/// silently for bots), and no mostly-uppercase text (lowercased).
enum LichessBotChatCommands {

    /// The command in `line`, or nil when it isn't one DCM answers. A
    /// command is `!` plus a known word as a *prefix*, case-insensitive,
    /// so "!name2" or "!help please" still count: Lichess's duplicate filter
    /// blocks a human repeating the exact same text. Our own lines (our
    /// greeting mentions !help) and the system user's are never commands.
    static func parse(_ line: LichessBotChatLine, ourAccountID: String) -> LichessBotChatCommand? {
        let speaker = line.username.lowercased()
        guard speaker != ourAccountID.lowercased(), speaker != "lichess" else { return nil }
        let text = line.text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard text.hasPrefix("!") else { return nil }
        let word = text.dropFirst()
        return LichessBotChatCommand.allCases.first { word.hasPrefix($0.rawValue) }
    }

    /// The messages to send for `command`, in order. Each is within
    /// `LichessBotChat.maximumLength`; an optional tail is kept only when it
    /// fits, and a reply whose required part doesn't fit throws.
    static func replies(to command: LichessBotChatCommand, context: LichessBotChatCommandContext) throws -> [String] {
        switch command {
        case .help:
            return [try fitted(command, "Commands: !name, !about, !motor, !cpu, !gpu, !ram. I answer in the chat you ask in.")]
        case .name:
            let core = "\(context.ourUsername) running \(engineName(context)) (build \(context.build))"
            return [try fitted(command, core, optionalTail: " · no search, one forward pass per move")]
        case .motor:
            return [try fitted(command, engineName(context))]
        case .about:
            return [
                try fitted(command, "Drew's Chess Machine is a from-scratch chess engine by drewster99 (GitHub). A neural net picks each move in one forward pass: no search."),
                try fitted(command, "It learns from self-play and from replayed games. It's written in Swift and runs on Apple silicon, on the GPU through Metal's MPSGraph."),
                try fitted(command, "This machine: \(cpuBrand(context.hardware)), \(cpuCoresSummary(context.hardware)), \(gpuCoresSummary(context.hardware)), \(memorySummary(context.hardware)).")
            ]
        case .cpu:
            return [try fitted(command, "\(cpuBrand(context.hardware)) · \(cpuCoresSummary(context.hardware))")]
        case .gpu:
            return [try fitted(command, "\(context.hardware.gpuModel ?? "unknown GPU") · \(gpuCoresSummary(context.hardware))")]
        case .ram:
            return [try fitted(command, memorySummary(context.hardware))]
        }
    }

    // MARK: - Pieces

    /// "DCM <modelID> step <n>", or without the step when the model has
    /// none (an untrained or champion export).
    static func engineName(_ context: LichessBotChatCommandContext) -> String {
        guard let step = context.trainingStep else { return "DCM \(context.modelID)" }
        return "DCM \(context.modelID) step \(step)"
    }

    static func cpuBrand(_ hardware: HardwareInfo) -> String {
        hardware.cpuBrand ?? "unknown CPU"
    }

    /// The core count, with each performance level's cores.
    static func cpuCoresSummary(_ hardware: HardwareInfo) -> String {
        guard let cores = hardware.cpuPhysicalCores else { return "unknown core count" }
        let levels = hardware.cpuPerformanceLevels.map { "\($0.physicalCores) \($0.name.lowercased())" }
        return levels.isEmpty ? "\(cores)-core CPU" : "\(cores)-core CPU (\(levels.joined(separator: " + ")))"
    }

    /// The GPU core count.
    static func gpuCoresSummary(_ hardware: HardwareInfo) -> String {
        guard let cores = hardware.gpuCoreCount else { return "unknown GPU core count" }
        return "\(cores)-core GPU"
    }

    /// Unified memory, in base-2 GB.
    static func memorySummary(_ hardware: HardwareInfo) -> String {
        guard let bytes = hardware.memoryBytes else { return "unknown memory" }
        let gigabytes = Double(bytes) / Double(1 << 30)
        let text = gigabytes == gigabytes.rounded() ? String(format: "%.0f", gigabytes) : String(format: "%.1f", gigabytes)
        return "\(text) GB unified memory"
    }

    /// Lichess counts UTF-16 code units (Java `String.length`).
    private static func fitted(_ command: LichessBotChatCommand, _ core: String, optionalTail: String = "") throws -> String {
        let limit = LichessBotChat.maximumLength
        guard core.utf16.count <= limit else {
            throw LichessBotChatCommandError.replyTooLong(command: command, length: core.utf16.count)
        }
        let full = core + optionalTail
        return full.utf16.count <= limit ? full : core
    }
}

/// Limits on command replies for one game (plan §12.5a): replies are
/// requests through the shared request gate, so an opponent must not be
/// able to spend DCM's request budget or delay its moves.
struct LichessBotChatCommandBudget: Sendable, Equatable {
    static let cooldown: Duration = .seconds(3)
    static let maximumRepliesPerGame = 20
    /// Below this on our clock, commands go unanswered.
    static let minimumClockMilliseconds = 30_000

    enum Decision: Equatable {
        case reply
        case skip(reason: String)
    }

    private(set) var repliesSent = 0
    private(set) var lastReplyAt: Duration?

    /// A game's budget at its start.
    init() {}

    /// A resumed game's budget: `repliesSent` already spent by earlier
    /// sessions of the game (counted from its journal). The cooldown starts
    /// over: the last reply was at least a relaunch ago.
    init(repliesSent: Int) {
        self.repliesSent = repliesSent
    }

    /// Whether to answer a command now; records the reply when it says yes.
    mutating func decide(now: Duration, ourClockMilliseconds: Int?) -> Decision {
        if repliesSent >= Self.maximumRepliesPerGame {
            return .skip(reason: "the per-game limit of \(Self.maximumRepliesPerGame) command replies is reached")
        }
        if let lastReplyAt, now - lastReplyAt < Self.cooldown {
            return .skip(reason: "the previous command reply was under \(Self.cooldown) ago")
        }
        if let ourClockMilliseconds, ourClockMilliseconds < Self.minimumClockMilliseconds {
            return .skip(reason: "our clock is under \(Self.minimumClockMilliseconds / 1000) s")
        }
        repliesSent += 1
        lastReplyAt = now
        return .reply
    }
}
