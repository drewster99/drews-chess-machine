import SwiftUI

/// Pill-shaped status chip rendered in the top status bar. Color
/// + label come from the resolved `Kind` plus the warmup progress
/// (when in the warmup state). A small spinner is rendered to the
/// left of the label in any active state so motion is visible
/// even when the text doesn't change for stretches at a time. Idle
/// stays static (no spinner).
struct SessionStatusChipView: View {
    /// What the live training session is currently doing. Outside
    /// of a Play-and-Train session the chip reads Idle, regardless
    /// of transient single-shot operations like Build or Forward
    /// Pass — those have their own busy indicators and don't merit
    /// a "what is this session doing?" chip.
    enum Kind {
        case idle
        case selfPlayPrefill
        case trainingWarmup
        case training

        var background: Color {
            switch self {
            case .idle: return Color.gray.opacity(0.25)
            case .selfPlayPrefill: return Color.orange.opacity(0.85)
            case .trainingWarmup: return Color.blue.opacity(0.85)
            case .training: return Color.green.opacity(0.85)
            }
        }
        var foreground: Color {
            switch self {
            case .idle: return .secondary
            default: return .white
            }
        }
    }

    let kind: Kind
    /// Steps the trainer has completed inside the current LR-warmup
    /// window. `nil` outside warmup; consumers pass it from
    /// `trainerWarmupSnap?.completedSteps`. Drives the `NN%` suffix
    /// on the warmup chip's label.
    let warmupCompletedSteps: Int?
    /// Total steps in the LR-warmup window. `nil` outside warmup.
    let warmupTotalSteps: Int?

    var body: some View {
        HStack(spacing: 6) {
            if kind != .idle {
                ProgressView()
                    .controlSize(.small)
                    .scaleEffect(0.6)
                    .frame(width: 12, height: 12)
                    .tint(kind.foreground)
            }
            Text(label)
                .font(.callout.weight(.semibold))
                .foregroundStyle(kind.foreground)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(kind.background)
        )
    }

    /// Label string. Idle / prefill / Training are static; the
    /// warm-up entry tacks `NN%` onto the end so the user can see
    /// warmup progress at a glance.
    private var label: String {
        switch kind {
        case .idle:            return "Idle"
        case .selfPlayPrefill: return "Self-play prefill"
        case .training:        return "Training"
        case .trainingWarmup:
            guard let total = warmupTotalSteps, total > 0,
                  let completed = warmupCompletedSteps else {
                return "Training: LR warm-up"
            }
            // Clamp to 0…99: 100% means warmup is complete and the
            // chip should already have transitioned to .training,
            // so capping here avoids a 100% flash on the boundary
            // tick.
            let raw = Double(completed) / Double(total) * 100.0
            let pct = min(99, max(0, Int(raw.rounded())))
            return "Training: LR warm-up \(pct)%"
        }
    }
}
