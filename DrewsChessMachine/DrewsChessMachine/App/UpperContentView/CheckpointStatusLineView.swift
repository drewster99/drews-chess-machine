import SwiftUI

/// One line in the checkpoint status row — icon + colored message.
/// `kind` drives both the leading SF Symbol and the foreground color
/// so the line communicates state at a glance: secondary gray while
/// a save is in progress, amber clock when it has gone slow, green
/// check on success, red triangle on error.
struct CheckpointStatusLineView: View {
    let kind: CheckpointStatusKind
    let message: String

    var body: some View {
        HStack(spacing: 4) {
            if let iconName {
                Image(systemName: iconName)
                    .foregroundStyle(color)
            }
            Text(message)
                .font(.callout)
                .foregroundStyle(color)
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }

    private var color: Color {
        switch kind {
        case .progress: .secondary
        case .slowProgress: .orange
        case .success: .green
        case .error: .red
        }
    }

    private var iconName: String? {
        switch kind {
        case .progress: nil
        case .slowProgress: "clock.badge.exclamationmark.fill"
        case .success: "checkmark.circle.fill"
        case .error: "exclamationmark.triangle.fill"
        }
    }
}
