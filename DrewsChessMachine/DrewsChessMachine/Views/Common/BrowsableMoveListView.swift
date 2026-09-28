import SwiftUI

/// A game's moves in numbered rows (`12.  Nf3  Nc6`), with the displayed
/// position's move highlighted (plan §14.3a). Clicking a move asks the host
/// to *view* the position after it; the list never changes the game. While
/// following live it scrolls to keep the latest move in view.
struct BrowsableMoveListView: View {
    /// SAN of every ply played so far.
    let sanMoves: [String]
    /// The ply count of the displayed position.
    let displayedPlyCount: Int
    let isLive: Bool
    /// Called with the ply count to view (the position after that move).
    let onSelectPlyCount: (Int) -> Void

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(0..<rowCount, id: \.self) { row in
                        BrowsableMoveRow(
                            moveNumber: row + 1,
                            whiteSAN: san(at: row * 2),
                            blackSAN: san(at: row * 2 + 1),
                            highlightedPly: displayedPlyCount - 1,
                            whitePly: row * 2,
                            onSelectPlyCount: onSelectPlyCount
                        )
                        .id(row)
                    }
                }
                .padding(.vertical, 4)
            }
            .onChange(of: sanMoves.count) {
                if isLive, rowCount > 0 {
                    proxy.scrollTo(rowCount - 1, anchor: .bottom)
                }
            }
            .onChange(of: displayedPlyCount) {
                if displayedPlyCount > 0 {
                    proxy.scrollTo((displayedPlyCount - 1) / 2)
                }
            }
        }
    }

    private var rowCount: Int {
        (sanMoves.count + 1) / 2
    }

    private func san(at ply: Int) -> String? {
        ply < sanMoves.count ? sanMoves[ply] : nil
    }
}

/// One numbered row of `BrowsableMoveListView`.
struct BrowsableMoveRow: View {
    let moveNumber: Int
    let whiteSAN: String?
    let blackSAN: String?
    /// The 0-based ply to highlight (-1 for none).
    let highlightedPly: Int
    let whitePly: Int
    let onSelectPlyCount: (Int) -> Void

    var body: some View {
        HStack(spacing: 4) {
            Text("\(moveNumber).")
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 36, alignment: .trailing)
            BrowsableMoveCell(san: whiteSAN, ply: whitePly, isHighlighted: highlightedPly == whitePly, onSelectPlyCount: onSelectPlyCount)
            BrowsableMoveCell(san: blackSAN, ply: whitePly + 1, isHighlighted: highlightedPly == whitePly + 1, onSelectPlyCount: onSelectPlyCount)
        }
        .padding(.horizontal, 4)
    }
}

/// One half-move in a `BrowsableMoveRow`. An empty cell (black has not moved
/// yet) keeps its width so columns stay aligned.
struct BrowsableMoveCell: View {
    let san: String?
    let ply: Int
    let isHighlighted: Bool
    let onSelectPlyCount: (Int) -> Void

    var body: some View {
        Button {
            onSelectPlyCount(ply + 1)
        } label: {
            Text(san ?? "")
                .font(.system(.callout, design: .monospaced))
                .frame(width: 64, alignment: .leading)
                .padding(.horizontal, 4)
                .padding(.vertical, 1)
                .background(
                    RoundedRectangle(cornerRadius: 3)
                        .fill(Color.accentColor.opacity(isHighlighted ? 0.30 : 0))
                )
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(san == nil)
        .accessibilityLabel(san.map { "Move \(ply + 1), \($0)" } ?? "")
    }
}
