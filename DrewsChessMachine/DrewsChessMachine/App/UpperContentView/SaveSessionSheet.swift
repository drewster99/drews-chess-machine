import SwiftUI

/// What File ▸ Save Session asks before it saves (determinism plan D-8):
/// whether this save includes the replay buffer. The checkbox starts from
/// `session_save_include_replay_buffer` and applies to this save only.
struct SaveSessionSheetRequest: Identifiable {
    let id = UUID()
    /// The checkbox's starting state: the automatic-save setting.
    let initialIncludeReplayBuffer: Bool
    /// The buffer's size on disk if this save includes it.
    let replayBufferSizeText: String
}

/// The Save Session confirmation: an "Include replay buffer" checkbox with
/// the buffer's size, and what a resume from the save does either way.
struct SaveSessionSheet: View {
    let request: SaveSessionSheetRequest
    let onSave: (_ includeReplayBuffer: Bool) -> Void
    let onCancel: () -> Void
    @State private var includeReplayBuffer: Bool

    init(request: SaveSessionSheetRequest,
         onSave: @escaping (_ includeReplayBuffer: Bool) -> Void,
         onCancel: @escaping () -> Void) {
        self.request = request
        self.onSave = onSave
        self.onCancel = onCancel
        _includeReplayBuffer = State(initialValue: request.initialIncludeReplayBuffer)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Save Session")
                .font(.title3.weight(.semibold))
            Text("Writes the champion, the trainer with its optimizer state, and the run's state to a new session folder.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 8) {
                Toggle("Include replay buffer", isOn: $includeReplayBuffer)
                    .toggleStyle(.checkbox)
                Text(request.replayBufferSizeText)
                    .font(.callout)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            Text(includeReplayBuffer
                 ? "A resume from this save restores the buffer as saved."
                 : "A resume from this save refills the buffer from new games before training continues.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                Button("Cancel", role: .cancel, action: onCancel)
                    .keyboardShortcut(.cancelAction)
                Spacer()
                Button("Save") { onSave(includeReplayBuffer) }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(minWidth: 460)
    }
}
