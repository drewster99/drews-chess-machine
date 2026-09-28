import Foundation

/// A duration in whole seconds, as Lichess uses in challenge time controls
/// (`timeControl.limit`, `timeControl.increment`).
///
/// Lichess mixes units across its payloads: challenges carry seconds, while
/// game streams (`gameFull.clock`, `gameState.wtime`/`btime`/`winc`/`binc`)
/// carry milliseconds. Distinct wrapper types make passing one where the
/// other is expected a compile error instead of a 1000× bug (plan E17).
struct LichessBotSeconds: Sendable, Hashable, Comparable, Codable {
    let value: Int

    init(_ value: Int) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        value = try decoder.singleValueContainer().decode(Int.self)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }

    static func < (lhs: LichessBotSeconds, rhs: LichessBotSeconds) -> Bool {
        lhs.value < rhs.value
    }
}

/// A duration in whole milliseconds, as Lichess uses in game streams. See
/// `LichessBotSeconds`.
struct LichessBotMilliseconds: Sendable, Hashable, Comparable, Codable {
    let value: Int

    init(_ value: Int) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        value = try decoder.singleValueContainer().decode(Int.self)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }

    static func < (lhs: LichessBotMilliseconds, rhs: LichessBotMilliseconds) -> Bool {
        lhs.value < rhs.value
    }

    var duration: Duration {
        .milliseconds(value)
    }
}
