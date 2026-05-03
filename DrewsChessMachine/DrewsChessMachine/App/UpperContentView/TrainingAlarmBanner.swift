import SwiftUI

/// Yellow banner shown when an alarm condition (policy collapse,
/// value-head saturation, etc.) is currently active. "Silence"
/// quiets the alert sound but keeps the banner visible; "Dismiss"
/// clears the banner and resets the streak counters so the alarm
/// only re-raises if the condition deteriorates fresh from a
/// healthy baseline.
struct TrainingAlarmBanner: View {
    let alarm: TrainingAlarm
    let isSilenced: Bool
    let onSilence: () -> Void
    let onDismiss: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                // Title forced black against the yellow background
                // for legibility — default `.headline` color in
                // dark-mode SwiftUI is white, which washes out
                // against `.yellow.opacity(0.8)`.
                Text(alarm.title)
                    .font(.headline)
                    .foregroundStyle(Color.black)
                // Detail text uses a darker red + medium weight so
                // numeric values in the alarm (entropy, gNorm) read
                // clearly against the yellow background instead of
                // washing out as default `.red`.
                Text(alarm.detail)
                    .font(.callout.weight(.medium))
                    .foregroundStyle(Color(red: 0.55, green: 0.0, blue: 0.0))
            }
            Spacer()
            HStack(spacing: 8) {
                if isSilenced {
                    Text("Silenced")
                        .font(.caption)
                        .foregroundStyle(Color.black)
                } else {
                    Button("Silence", action: onSilence)
                }
                Button("Dismiss", action: onDismiss)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color.yellow.opacity(0.8))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
