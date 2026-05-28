import SwiftUI

/// Two-row horizontal bar comparing the replay buffer's resident
/// per-material-bucket distribution against the most recent training
/// batch's per-bucket distribution.
///
/// Buckets follow `ReplayBufferAnalyzer.materialBuckets` (`0–4`,
/// `5–8`, `9–14`, `15–22` non-pawn pieces; the 5th `23–30` is
/// structurally unreachable in standard chess and is folded into the
/// rightmost slot — almost always 0).
///
/// **Behavior:**
/// - When `stratifyOn == false`, the two rows should look nearly
///   identical (the batch is a uniform draw from the buffer). This is
///   itself a useful diagnostic: if they look different something is
///   wrong with the bucket index.
/// - When `stratifyOn == true`, the batch row should converge to
///   `1 / activeBuckets` per active bucket (balanced target). A dashed
///   reference line at that target is drawn on the batch row.
///
/// **View-stability:** the chart is always present in its parent's view
/// tree (per project SwiftUI conventions: no `if`-gated visible content
/// — opacity + zero-height frame is used at the parent to toggle
/// visibility while keeping the type identity stable).
struct ReplaySamplingBucketChartView: View {
    /// Resident-set per-bucket counts, indexed by
    /// `ReplayBufferAnalyzer.materialBuckets`. May be nil (no
    /// composition snapshot yet).
    let bufferCounts: [Int]?
    /// Last-batch achieved per-bucket counts, indexed by
    /// `ReplayBufferAnalyzer.materialBuckets`. May be nil (no batch
    /// has landed yet).
    let batchCounts: [Int]?
    /// Whether stratification is currently on. When true the target
    /// reference line is drawn on the batch row and the colors slightly
    /// brighten.
    let stratifyOn: Bool

    /// Per-bucket fill color. Kept as a small palette so the same
    /// bucket reads the same color across the buffer and batch rows.
    /// Standard SwiftUI semantic colors so dark mode renders correctly.
    private static let bucketColors: [Color] = [
        .blue,        // 0–4: deep endgame
        .green,       // 5–8: late endgame
        .orange,      // 9–14: middlegame
        .purple,      // 15–22: full piece set
        .gray         // 23–30: structurally unreachable
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .center, spacing: 8) {
                Text("Bucket mix")
                    .font(.caption.weight(.medium))
                Spacer()
                Text(stratifyOn ? "Stratified (target: balanced)" : "Natural (uniform sample)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            // Header row — bucket labels above the bars.
            HStack(spacing: 4) {
                Text("")
                    .frame(width: 60, alignment: .trailing)
                ForEach(activeBucketIndices, id: \.self) { idx in
                    Text(ReplayBufferAnalyzer.materialBuckets[idx].label)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity)
                }
            }
            // Buffer row.
            bucketRow(
                label: "Buffer",
                counts: bufferCounts,
                showTargetLine: false
            )
            // Batch row.
            bucketRow(
                label: "Last batch",
                counts: batchCounts,
                showTargetLine: stratifyOn
            )
            // Footer: per-bucket numbers under each bar, condensed.
            // Surfaces the exact achieved counts so the user can
            // verify "32/32/32/32" when stratification is on and
            // see the natural skew when it's off.
            HStack(spacing: 4) {
                Text("counts")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 60, alignment: .trailing)
                ForEach(activeBucketIndices, id: \.self) { idx in
                    Text(countLabel(forBatchBucket: idx))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity)
                }
            }
        }
    }

    /// Indices of buckets to render. We hide the 5th bucket (23–30
    /// non-pawn pieces) because it's structurally unreachable in
    /// standard chess and always reads 0 / 0 / 0 — including it would
    /// just waste horizontal pixels.
    private var activeBucketIndices: [Int] {
        let total = ReplayBufferAnalyzer.materialBuckets.count
        return Array(0..<max(total - 1, 0))
    }

    /// Renders one bar row (buffer or batch). Each bucket gets a
    /// fixed-width column with a bar whose height is proportional to
    /// that bucket's share of the row total. Empty rows render as
    /// zero-height bars (the underlying box still draws).
    @ViewBuilder
    private func bucketRow(
        label: String,
        counts: [Int]?,
        showTargetLine: Bool
    ) -> some View {
        HStack(alignment: .center, spacing: 4) {
            Text(label)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 60, alignment: .trailing)
            let active = activeBucketIndices
            let activeSum = active.reduce(0) { acc, idx in
                acc + ((counts.flatMap { $0.indices.contains(idx) ? $0[idx] : 0 }) ?? 0)
            }
            ForEach(active, id: \.self) { idx in
                let count = (counts.flatMap { $0.indices.contains(idx) ? $0[idx] : 0 }) ?? 0
                let frac = activeSum > 0 ? Double(count) / Double(activeSum) : 0
                ZStack(alignment: .leading) {
                    // Background slot — always present for view stability.
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color.secondary.opacity(0.10))
                        .frame(height: 16)
                    // Filled portion.
                    GeometryReader { geo in
                        RoundedRectangle(cornerRadius: 2)
                            .fill(Self.bucketColors[idx].opacity(0.7))
                            .frame(width: max(0, geo.size.width * CGFloat(frac)),
                                   height: 16)
                    }
                    .frame(height: 16)
                    // Target reference line — dashed vertical at the
                    // balanced target. Only rendered on the batch row
                    // when stratification is on. View-stability rule:
                    // we render the shape unconditionally and zero its
                    // opacity when not needed, rather than `if`-gating
                    // it out of the tree.
                    GeometryReader { geo in
                        let target = active.count > 0 ? CGFloat(1.0 / Double(active.count)) : 0
                        Path { p in
                            let x = geo.size.width * target
                            p.move(to: CGPoint(x: x, y: 2))
                            p.addLine(to: CGPoint(x: x, y: 14))
                        }
                        .stroke(
                            Color.primary.opacity(0.6),
                            style: StrokeStyle(lineWidth: 1, dash: [2, 2])
                        )
                        .opacity(showTargetLine ? 1 : 0)
                    }
                    .frame(height: 16)
                }
                .frame(maxWidth: .infinity)
            }
        }
    }

    /// Per-bucket label rendered under the batch row's bars. Shows
    /// the batch's achieved count for that bucket (the number the
    /// stratifier landed on), with a dash when no batch has come in
    /// yet.
    private func countLabel(forBatchBucket idx: Int) -> String {
        guard let counts = batchCounts, counts.indices.contains(idx) else {
            return "—"
        }
        return "\(counts[idx])"
    }
}
