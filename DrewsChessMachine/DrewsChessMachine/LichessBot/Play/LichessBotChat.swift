import Foundation

/// Chat message templates (plan §12.5, E52, E53).
///
/// Templates may use `{modelID}`, `{source}`, `{build}` and `{opponent}`.
/// A message is checked against Lichess's length limit after expansion, and
/// an over-limit template is rejected in Settings rather than truncated when
/// sent. Opponent chat is interpreted only as the read-only commands in
/// `LichessBotChatCommands` (plan §12.5a), within a per-game budget.
enum LichessBotChat {

    /// Lichess's chat message length limit (`Line.textMaxSize` in lila),
    /// counted in UTF-16 code units (Java `String.length`); a longer message
    /// is rejected with HTTP 400. Count with `.utf16.count`, never `.count`.
    static let maximumLength = 140

    static let placeholders = ["modelID", "source", "build", "opponent"]

    /// Longest plausible expansion of each placeholder, used to validate a
    /// template before any real values exist. Lichess usernames are at most
    /// this long; ModelIDs and build labels are generous upper bounds.
    private static let longestPlausibleValues: [String: String] = [
        "modelID": String(repeating: "M", count: 32),
        "source": String(repeating: "S", count: 20),
        "build": String(repeating: "B", count: 12),
        "opponent": String(repeating: "O", count: 20),
    ]

    /// `template` with every `{name}` replaced by `values[name]`. Unknown
    /// placeholders are left as written (and flagged by `templateProblem`).
    static func expand(_ template: String, values: [String: String]) -> String {
        var result = template
        for (name, value) in values {
            result = result.replacingOccurrences(of: "{\(name)}", with: value)
        }
        return result
    }

    /// Why `template` can't be used, or nil if it can.
    static func templateProblem(_ template: String) -> String? {
        let trimmed = template.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return "the message is empty"
        }
        let expanded = expand(template, values: longestPlausibleValues)
        if let unknown = unknownPlaceholder(in: expanded) {
            return "unknown placeholder {\(unknown)}; use \(placeholders.map { "{\($0)}" }.joined(separator: ", "))"
        }
        if expanded.utf16.count > maximumLength {
            return "can reach \(expanded.utf16.count) characters with long values; Lichess allows \(maximumLength)"
        }
        return nil
    }

    /// The final message to send, or nil if it would exceed the limit with
    /// these actual values (then it is not sent, and the skip is logged).
    static func message(from template: String, values: [String: String]) -> String? {
        let expanded = expand(template, values: values)
        return expanded.utf16.count <= maximumLength ? expanded : nil
    }

    private static func unknownPlaceholder(in text: String) -> String? {
        guard let open = text.firstIndex(of: "{"),
              let close = text[open...].firstIndex(of: "}") else {
            return nil
        }
        return String(text[text.index(after: open)..<close])
    }
}

/// Rules for the operator's own chat messages (plan §14.3c), which Lichess
/// applies to every bot message: at most `LichessBotChat.maximumLength`
/// UTF-16 units, and anything link-like is dropped silently (with HTTP 200).
enum LichessBotOperatorChat {
    /// Lichess's count: UTF-16 code units.
    static func length(of text: String) -> Int {
        text.utf16.count
    }

    /// A URL, or a word followed by a dot and a top-level-domain-like word
    /// ("lichess.org", "github.com"): Lichess drops link-bearing bot
    /// messages without an error.
    static func looksLikeLink(_ text: String) -> Bool {
        let lowered = text.lowercased()
        if lowered.contains("http://") || lowered.contains("https://") || lowered.contains("www.") {
            return true
        }
        return lowered.contains(#/[a-z0-9-]+\.[a-z]{2,}(?:[/?#]|\b)/#)
    }

    /// Why `text` can't be sent, or nil if it can. A link-like message can
    /// still be sent (the heuristic can be wrong); the UI warns about it.
    static func problem(with text: String) -> String? {
        if text.isEmpty {
            return "the message is empty"
        }
        let length = length(of: text)
        if length > LichessBotChat.maximumLength {
            return "\(length) characters; Lichess allows \(LichessBotChat.maximumLength)"
        }
        return nil
    }
}
