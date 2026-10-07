import SwiftUI

/// The All Games window's Origin filter (challenge-log plan §3.9): every
/// game, or the games of one origin category, each with its glyph.
struct LichessBotAllGamesOriginFilterPicker: View {
    @Binding var filter: LichessBotAllGamesOriginFilter

    var body: some View {
        Picker("Origin", selection: $filter) {
            Text("All").tag(LichessBotAllGamesOriginFilter.all)
            Divider()
            ForEach(LichessBotGameOriginCategory.allCases, id: \.self) { category in
                Label(LichessBotGameOriginStyle.shortLabel(for: category), systemImage: LichessBotGameOriginStyle.systemImage(for: category))
                    .tag(LichessBotAllGamesOriginFilter.only(category))
            }
        }
        .pickerStyle(.menu)
        .fixedSize()
    }
}
