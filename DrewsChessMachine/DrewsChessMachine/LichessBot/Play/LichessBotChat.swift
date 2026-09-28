import Foundation

/// Chat message templates (plan §12.5, E52, E53).
///
/// Templates may use `{modelID}`, `{source}`, `{build}` and `{opponent}`.
/// A message is checked against Lichess's length limit after expansion, and
/// an over-limit template is rejected in Settings rather than truncated when
/// sent. Opponent chat is never interpreted: there are no chat commands, so
/// nothing an opponent types can trigger a request or an action.
enum LichessBotChat {

    /// Lichess's chat message length limit, as currently understood. Plan
    /// §20.11 lists confirming it on a live game.
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
        if expanded.count > maximumLength {
            return "can reach \(expanded.count) characters with long values; Lichess allows \(maximumLength)"
        }
        return nil
    }

    /// The final message to send, or nil if it would exceed the limit with
    /// these actual values (then it is not sent, and the skip is logged).
    static func message(from template: String, values: [String: String]) -> String? {
        let expanded = expand(template, values: values)
        return expanded.count <= maximumLength ? expanded : nil
    }

    private static func unknownPlaceholder(in text: String) -> String? {
        guard let open = text.firstIndex(of: "{"),
              let close = text[open...].firstIndex(of: "}") else {
            return nil
        }
        return String(text[text.index(after: open)..<close])
    }
}
