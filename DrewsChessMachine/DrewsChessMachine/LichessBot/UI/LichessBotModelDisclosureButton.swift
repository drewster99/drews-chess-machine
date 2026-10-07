import SwiftUI

/// Opens or closes a model group's checkpoints in the Models table. A row
/// that is not a group (`groupModelID` nil) gets an empty, disabled button
/// of the same size, so the column stays aligned without an `if`.
struct LichessBotModelDisclosureButton: View {
    let groupModelID: String?
    @Binding var expanded: Set<String>

    var body: some View {
        let isExpanded = groupModelID.map { expanded.contains($0) } ?? false
        Button(action: toggle) {
            Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                .frame(width: 12)
        }
        .buttonStyle(.borderless)
        .opacity(groupModelID == nil ? 0 : 1)
        .disabled(groupModelID == nil)
        .accessibilityLabel(isExpanded ? "Hide checkpoints" : "Show checkpoints")
        .accessibilityHidden(groupModelID == nil)
    }

    private func toggle() {
        guard let groupModelID else { return }
        if expanded.contains(groupModelID) {
            expanded.remove(groupModelID)
        } else {
            expanded.insert(groupModelID)
        }
    }
}
