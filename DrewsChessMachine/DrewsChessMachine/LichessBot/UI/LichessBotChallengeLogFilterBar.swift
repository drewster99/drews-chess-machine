import SwiftUI

/// The Challenge Log window's filters (challenge-log plan §3.9): date range,
/// direction, state, sender, whether to include the history rebuilt from the
/// protocol log, and a search on the opponent.
struct LichessBotChallengeLogFilterBar: View {
    @Binding var filter: LichessBotChallengeLogFilter

    var body: some View {
        HStack(spacing: 12) {
            Picker("Period", selection: $filter.dateRange) {
                ForEach(LichessBotChallengeLogFilter.DateRange.allCases, id: \.self) { range in
                    Text(LichessBotChallengeLogStyle.label(range)).tag(range)
                }
            }
            .pickerStyle(.segmented)
            .fixedSize()
            Picker("Direction", selection: $filter.direction) {
                ForEach(LichessBotChallengeLogFilter.DirectionChoice.allCases, id: \.self) { direction in
                    Text(LichessBotChallengeLogStyle.label(direction)).tag(direction)
                }
            }
            .pickerStyle(.menu)
            .fixedSize()
            Picker("State", selection: $filter.state) {
                Text("All").tag(LichessBotChallengeLogFilter.StateChoice.all)
                Divider()
                ForEach(LichessBotChallengeLogStateKind.allCases, id: \.self) { kind in
                    Text(LichessBotChallengeLogStyle.label(kind)).tag(LichessBotChallengeLogFilter.StateChoice.only(kind))
                }
            }
            .pickerStyle(.menu)
            .fixedSize()
            Picker("Sender", selection: $filter.sender) {
                Text("All").tag(LichessBotChallengeLogFilter.SenderChoice.all)
                Divider()
                ForEach(LichessBotChallengeLogSenderKind.allCases, id: \.self) { kind in
                    Text(LichessBotChallengeLogStyle.label(kind)).tag(LichessBotChallengeLogFilter.SenderChoice.only(kind))
                }
            }
            .pickerStyle(.menu)
            .fixedSize()
            Toggle("Include reconstructed", isOn: $filter.includeReconstructed)
                .fixedSize()
                .help("Include the challenges rebuilt from the protocol log, from before the challenge log began")
            Spacer(minLength: 8)
            TextField("Opponent", text: $filter.opponentSearch)
                .textFieldStyle(.roundedBorder)
                .frame(minWidth: 120, maxWidth: 200)
        }
        .font(.callout)
    }
}
