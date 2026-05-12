import SwiftUI

/// Small `ⓘ` button shown beside the "Value label smoothing:" field on
/// the Optimizer tab of `TrainingSettingsPopover`. Tapping it opens a
/// popover that explains what the ε knob does to the per-class W/D/L
/// cross-entropy target and lists, for the low-ε range we'd actually
/// consider, the smoothed target a hard `[1, 0, 0]` (clean-win) outcome
/// maps to.
///
/// The formula matches `ChessTrainer.buildTrainingOps` (the value-CE
/// path): `target = (1 − ε)·oneHot + ε·(1/3)`, so the true class gets
/// `1 − 2ε/3` and each of the other two classes gets `ε/3`. The
/// reference table walks ε from 0 to 0.05 in 0.005 steps — large enough
/// to bound the value-head logits away from ±∞ on noisy outcome labels,
/// small enough to barely dent the head's confidence ceiling (≥ 0.967
/// win on a clean win even at ε = 0.05). Bigger ε is a logit-blow-up
/// regularizer, not a `pD → 1` draw-collapse fix — Draw penalty is the
/// lever for that.
struct ValueLabelSmoothingInfoButton: View {
    @State private var showingInfo = false

    /// ε rows shown in the reference table: 0, 0.005, 0.010, … 0.050.
    /// Built by integer scaling (not `stride(through:by:)`) so the last
    /// row lands on exactly 0.050 rather than a floating-point near-miss
    /// that the `through:` bound would drop.
    private static let epsilons: [Double] = (0...10).map { Double($0) * 0.005 }

    var body: some View {
        Button(action: { showingInfo.toggle() }, label: {
            Image(systemName: "info.circle")
        })
        .buttonStyle(.borderless)
        .help("What does value-label-smoothing ε do?")
        .popover(isPresented: $showingInfo) {
            VStack(alignment: .leading, spacing: 10) {
                Text("Value-head label smoothing (ε)")
                    .font(.headline)

                Text("""
                The W/D/L value head is trained with categorical cross-entropy \
                against the one-hot game outcome. Label smoothing replaces that \
                one-hot target with  target = (1 − ε)·oneHot + ε·(1⁄3)  — the \
                true class gets 1 − 2ε⁄3 and each other class gets ε⁄3. Larger \
                ε caps how confident the head can become and raises the \
                irreducible loss floor; ε = 0 disables it entirely. It is a \
                logit-blow-up regularizer, not a draw-collapse fix — for \
                pD → 1 use the Draw penalty knob instead.
                """)
                .font(.callout)
                .fixedSize(horizontal: false, vertical: true)
                .frame(width: 380, alignment: .leading)

                Divider()

                Text("A clean-win outcome [1, 0, 0] becomes:")
                    .font(.callout)

                Grid(alignment: .trailing, horizontalSpacing: 18, verticalSpacing: 3) {
                    GridRow {
                        Text("ε")
                        Text("win")
                        Text("draw")
                        Text("loss")
                    }
                    .fontWeight(.bold)

                    Divider().gridCellColumns(4)

                    ForEach(Self.epsilons, id: \.self) { eps in
                        GridRow {
                            Text(String(format: "%.3f", eps))
                            Text(String(format: "%.5f", 1.0 - 2.0 * eps / 3.0))
                            Text(String(format: "%.5f", eps / 3.0))
                            Text(String(format: "%.5f", eps / 3.0))
                        }
                    }
                }
                .font(.system(.callout, design: .monospaced))
            }
            .padding(16)
        }
    }
}
