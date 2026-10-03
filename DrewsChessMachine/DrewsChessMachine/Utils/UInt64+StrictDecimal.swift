import Foundation

extension UInt64 {
    /// The value of `text` when it is a `UInt64` written the way every writer
    /// in this app writes one — `String(value)`: one or more ASCII digits and
    /// nothing else. Nil for anything else, including text Swift's own
    /// `UInt64(_:)` accepts: a leading `+` (`"+5"`), a negative zero
    /// (`"-0"`), which no writer produces. Also nil for whitespace, other
    /// signs, non-ASCII digits, and a value above `UInt64.max`.
    ///
    /// The single parser for seed text — `--seed`, `--init-seed`, the run
    /// seed in parameters files and stored settings, the settings field, the
    /// Build New Model init-seed field, and the seeds and stream states in
    /// lineage records — so one spelling is accepted or refused everywhere.
    /// A caller that tolerates surrounding whitespace (an edit field) trims
    /// it before calling.
    init?<Text: StringProtocol>(strictDecimal text: Text) {
        guard !text.isEmpty, text.utf8.allSatisfy({ $0 >= UInt8(ascii: "0") && $0 <= UInt8(ascii: "9") }) else {
            return nil
        }
        guard let value = UInt64(text, radix: 10) else { return nil }
        self = value
    }
}
