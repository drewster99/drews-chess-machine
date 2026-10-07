import SwiftUI

/// Lays out its one subview at least `minimumHeight` tall, and taller when
/// the subview needs more: the Record card's dragged height is a minimum,
/// never a clip.
///
/// A `.frame(minHeight: h, idealHeight: h)` can't express that inside a
/// scroll view. The scroll view proposes no height, so the frame takes its
/// ideal `h` and reports exactly `h` even when its content is laid out
/// taller; the overflow is drawn past the card's bottom, under the next
/// card. That is how the stacked recent games disappeared below the Record
/// card (they were still there, hidden under "Outgoing challenges").
///
/// Here the subview is asked for its size at `minimumHeight` (or the
/// proposed height, when larger), and the layout reports whichever is
/// taller. Placement then gives the subview the whole height, so flexible
/// content (the statistics column) still fills a card dragged taller than
/// it needs.
struct LichessBotAtLeastHeightLayout: Layout {
    let minimumHeight: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        guard let content = subviews.first else {
            return CGSize(width: proposal.width ?? 0, height: minimumHeight)
        }
        let height = max(minimumHeight, proposal.height ?? minimumHeight)
        let needed = content.sizeThatFits(ProposedViewSize(width: proposal.width, height: height))
        return CGSize(width: needed.width, height: max(height, needed.height))
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        for content in subviews {
            content.place(at: bounds.origin, anchor: .topLeading, proposal: ProposedViewSize(width: bounds.width, height: bounds.height))
        }
    }
}
