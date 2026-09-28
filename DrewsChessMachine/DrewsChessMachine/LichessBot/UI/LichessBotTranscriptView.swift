import SwiftUI

/// A game's protocol traffic as a chat-style transcript (plan §14.3a):
/// Lichess's stream lines on the left, DCM's requests on the right, local
/// notes in between. Click an entry to show its raw text. Follows the newest
/// entry while "Follow" is on; turn it off to read back without the view
/// jumping.
struct LichessBotTranscriptView: View {
    let entries: [LichessBotTranscriptEntry]
    /// Whether the transcript is on screen; coming back into view scrolls
    /// to the newest entry while following.
    let isVisible: Bool
    @State private var expanded: Set<Int> = []
    @State private var follows = true

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("\(entries.count) entries")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                Spacer()
                Toggle("Follow", isOn: $follows)
                    .toggleStyle(.checkbox)
                    .font(.caption)
            }
            .padding(.horizontal, 6)
            .padding(.top, 4)
            LichessBotTranscriptList(entries: entries, expanded: $expanded, follows: follows, isVisible: isVisible)
        }
    }
}

/// The scrolling list of transcript entries.
struct LichessBotTranscriptList: View {
    let entries: [LichessBotTranscriptEntry]
    @Binding var expanded: Set<Int>
    let follows: Bool
    let isVisible: Bool

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 4) {
                    ForEach(entries) { entry in
                        LichessBotTranscriptRow(
                            entry: entry,
                            isExpanded: expanded.contains(entry.id),
                            onToggle: {
                                if expanded.contains(entry.id) {
                                    expanded.remove(entry.id)
                                } else {
                                    expanded.insert(entry.id)
                                }
                            }
                        )
                        .id(entry.id)
                    }
                }
                .padding(6)
            }
            .onChange(of: entries.last?.id) {
                if follows, let last = entries.last?.id {
                    proxy.scrollTo(last, anchor: .bottom)
                }
            }
            .onChange(of: isVisible) {
                if isVisible, follows, let last = entries.last?.id {
                    proxy.scrollTo(last, anchor: .bottom)
                }
            }
            .onAppear {
                if follows, let last = entries.last?.id {
                    proxy.scrollTo(last, anchor: .bottom)
                }
            }
        }
    }
}

/// One transcript entry: a bubble aligned by direction.
struct LichessBotTranscriptRow: View {
    let entry: LichessBotTranscriptEntry
    let isExpanded: Bool
    let onToggle: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            Spacer(minLength: 0)
                .frame(maxWidth: entry.direction == .outgoing ? .infinity : 0)
            VStack(alignment: .leading, spacing: 2) {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(entry.at.formatted(.dateTime.hour(.twoDigits(amPM: .omitted)).minute(.twoDigits).second(.twoDigits)))
                        .font(.system(.caption2, design: .monospaced))
                        .foregroundStyle(.secondary)
                    Text(directionLabel)
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                    // Expanding shows the whole title too: notes and
                    // anomalies carry all their text there, with no detail.
                    Text(entry.title)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(entry.isProblem ? Color.red : Color.primary)
                        .lineLimit(isExpanded ? nil : 1)
                        .fixedSize(horizontal: false, vertical: isExpanded)
                    Text("×\(entry.repeatCount)")
                        .font(.system(.caption2, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .shown(entry.repeatCount > 1)
                }
                Text(entry.detail)
                    .font(.system(.caption2, design: .monospaced))
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .shown(isExpanded && !entry.detail.isEmpty)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(bubbleColor)
            )
            // Narrower than the panel, leaving room to tell the two sides
            // apart (plan §14.3c).
            .containerRelativeFrame(.horizontal, alignment: entry.direction == .outgoing ? .trailing : .leading) { length, _ in
                length * 0.75
            }
            .contentShape(Rectangle())
            .onTapGesture(perform: onToggle)
            .help(isExpanded ? "Click to collapse" : (entry.detail.isEmpty ? "Click to show the full text" : "Click to show the raw text"))
            .accessibilityElement(children: .combine)
            .accessibilityAddTraits(.isButton)
            .accessibilityAction(.default, onToggle)
            .accessibilityHint(isExpanded ? "Collapses the entry" : (entry.detail.isEmpty ? "Shows the full text" : "Shows the raw text"))
            Spacer(minLength: 0)
                .frame(maxWidth: entry.direction == .incoming ? .infinity : 0)
        }
    }

    private var directionLabel: String {
        switch entry.direction {
        case .incoming: return "LICHESS ▸"
        case .outgoing: return "◂ DCM"
        case .note: return "·"
        }
    }

    private var bubbleColor: Color {
        switch entry.direction {
        case .incoming: return Color.blue.opacity(0.10)
        case .outgoing: return Color.green.opacity(0.10)
        case .note: return Color.gray.opacity(0.10)
        }
    }
}
