import SwiftUI

/// One cell in the top-of-window status bar — a small caption label
/// above a monospaced numeric value. Optionally clickable: when
/// `action` is non-nil, the cell renders a hover highlight and the
/// pointer cursor over the whole label+value stack, and the entire
/// cell is a hit target that invokes the action on click.
///
/// Kept as a standalone view (rather than a `@ViewBuilder` helper)
/// because hover + click state needs to live in the cell itself —
/// inlining these into `ContentView.body` would smear `@State` for
/// each cell across the parent and blow up the type-checker budget.
struct StatusBarCell: View {
    let label: String
    let value: String
    /// Optional action. `nil` = static display, no hover affordance.
    var action: (() -> Void)?
    /// Optional override for the value color. Defaults to `.primary`.
    /// Used by the "Score" cell to dim `"—"` when no arenas have
    /// completed yet, so the user can tell the cell is present but
    /// currently empty rather than mistaking an em-dash for a value.
    var valueColor: Color = .primary

    @State private var hovering: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(.callout, design: .monospaced).weight(.semibold))
                .monospacedDigit()
                .foregroundStyle(valueColor)
        }
        .padding(.horizontal, action == nil ? 0 : 4)
        .padding(.vertical, action == nil ? 0 : 2)
        .background(
            RoundedRectangle(cornerRadius: 4)
                .fill(hoverTint)
        )
        .contentShape(Rectangle())
        .onHover { inside in
            // Only change the cursor and paint a hover background
            // when the cell is interactive. A non-clickable cell
            // still wants `onHover` to be a no-op so the hover
            // state ghost doesn't leak into downstream renders.
            guard action != nil else { return }
            hovering = inside
            if inside {
                NSCursor.pointingHand.push()
            } else {
                NSCursor.pop()
            }
        }
        .onDisappear {
            // If the cell is removed from the view tree while
            // hovered (e.g. `hasHistory` flips false on Stop while
            // the mouse is over the cell), `.onHover` never fires
            // the exit phase and the pointing-hand cursor stays
            // pushed on the NSCursor stack. Balance it here.
            if hovering {
                NSCursor.pop()
                hovering = false
            }
        }
        .onTapGesture {
            action?()
        }
    }

    private var hoverTint: Color {
        (action != nil && hovering)
            ? Color.secondary.opacity(0.18)
            : .clear
    }
}
