import SwiftUI

/// A small `TextField` wrapper that calls `onCommit` both when the user
/// presses Return (the default `.onSubmit` behavior) **and** when the
/// field loses focus (Tab away, click elsewhere, window deactivation).
///
/// The plain `.onSubmit` modifier on `TextField` only fires on
/// Return / Enter on macOS — not on focus-loss. That made every
/// parameter-editor field in this app silently throw away edits when
/// the user typed a value and tabbed out instead of pressing Enter.
/// The text binding still showed the typed value (because two-way
/// binding updates on every keystroke), but the underlying parameter
/// store kept its prior value, which then overwrote the displayed text
/// on the next session start. From the outside that looked like
/// "I edited the LR to 1e-3, the field showed 1e-3, training ran with
/// 1e-2."
///
/// This view fixes that systemically. Use it in place of a bare
/// `TextField` for any parameter editor:
///
///     ParameterTextField(
///         placeholder: "LR",
///         text: $learningRateEditText,
///         width: 80
///     ) { typed in
///         // parse + apply + reseed text from authoritative source
///     }
///
/// The `onCommit` closure receives the current text. It is
/// responsible for parsing, validating, applying to the parameter
/// store, and (typically) re-formatting the text binding from the
/// authoritative source so an unparseable value reverts visually to
/// the last good one. The same closure runs on Return and on
/// focus-loss, so the two paths can never disagree.
///
/// Help text: pass `.help(...)` on the result like any other view —
/// `ParameterTextField` does not eat modifiers.
struct ParameterTextField: View {
    /// Placeholder rendered inside the empty field.
    let placeholder: String
    /// Text content of the field. The caller owns this state so it can
    /// re-seed it from the authoritative source after parameter edits
    /// elsewhere (e.g., session start re-renders all the editors).
    @Binding var text: String
    /// Fixed pixel width for the field. Parameter editors lay out
    /// horizontally in tight rows and rely on per-field widths to
    /// keep the row from reflowing.
    var width: CGFloat
    /// Invoked on Return AND on focus-loss with the current text.
    /// Caller parses + applies + reseeds.
    let onCommit: (String) -> Void

    @FocusState private var isFocused: Bool

    var body: some View {
        TextField(placeholder, text: $text)
            .monospacedDigit()
            .frame(width: width)
            .textFieldStyle(.roundedBorder)
            .focused($isFocused)
            .onSubmit {
                onCommit(text)
            }
            .onChange(of: isFocused) { _, focused in
                // Focus-loss commits whatever's currently in the field.
                // Symmetric with .onSubmit — the same closure runs in
                // both paths so they cannot disagree on which value
                // gets persisted.
                if !focused {
                    onCommit(text)
                }
            }
    }
}
