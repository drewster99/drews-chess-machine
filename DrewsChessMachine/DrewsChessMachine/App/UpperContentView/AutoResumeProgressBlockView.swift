import SwiftUI

/// Training-progress block — steps, positions, self-play game
/// count, arena/promotion counts, and active wall-clock time.
/// All numbers grouped (12,345) and the rare-but-useful column
/// alignment is achieved via `monospacedDigit()` on the values
/// so the user can scan vertically. Two columns: label / value.
struct AutoResumeProgressBlockView: View {
    let summary: SessionResumeSummary

    var body: some View {
        let stepsStr = AutoResumeFormat.count(summary.trainingSteps)
        let positionsStr = AutoResumeFormat.count(summary.trainingPositionsSeen)
        let gamesStr = AutoResumeFormat.count(summary.selfPlayGames)
        let movesStr = AutoResumeFormat.count(summary.selfPlayMoves)
        let promoPct: String = {
            guard summary.arenaCount > 0 else { return "—" }
            let pct = Double(summary.promotionCount) / Double(summary.arenaCount) * 100.0
            return String(format: "%.0f%%", pct)
        }()
        let arenasStr = AutoResumeFormat.count(summary.arenaCount)
        let promotionsStr = "\(AutoResumeFormat.count(summary.promotionCount)) (\(promoPct))"
        let activeStr = AutoResumeFormat.activeDuration(summary.elapsedTrainingSec)

        return VStack(alignment: .leading, spacing: 2) {
            AutoResumeStatRowView(label: "Training", value: "\(stepsStr) steps · \(positionsStr) positions")
            AutoResumeStatRowView(label: "Self-play", value: "\(gamesStr) games · \(movesStr) moves")
            AutoResumeStatRowView(label: "Arenas", value: arenasStr)
            AutoResumeStatRowView(label: "Promotions", value: promotionsStr)
            AutoResumeStatRowView(label: "Active", value: activeStr)
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 10)
        .background(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(Color.secondary.opacity(0.08))
        )
    }
}
