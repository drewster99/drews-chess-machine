import SwiftUI

/// Capsule badge identifying which save trigger produced the
/// session pointer. Post-promotion saves get the accent color
/// ("the just-trained model just got better"); manual / periodic
/// still get a badge for visual consistency but in muted colors.
struct AutoResumeTriggerBadgeView: View {
    let trigger: String

    var body: some View {
        let label: String
        let bgColor: Color
        let fgColor: Color
        switch trigger {
        case "post-promotion":
            label = "POST-PROMOTION"
            bgColor = Color.accentColor
            fgColor = Color.white
        case "manual":
            label = "MANUAL SAVE"
            bgColor = Color.secondary.opacity(0.18)
            fgColor = Color.primary
        case "periodic":
            label = "PERIODIC AUTOSAVE"
            bgColor = Color.secondary.opacity(0.18)
            fgColor = Color.primary
        default:
            label = trigger.uppercased()
            bgColor = Color.secondary.opacity(0.18)
            fgColor = Color.primary
        }
        return Text(label)
            .font(.caption2)
            .fontWeight(.semibold)
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(
                Capsule().fill(bgColor)
            )
            .foregroundStyle(fgColor)
    }
}
