import SwiftUI

/// The Opponents filter (`LichessBotOpponentKindFilter`) as a menu: one view
/// for the Record card and the All Games window, so the two read the same.
struct LichessBotOpponentKindFilterPicker: View {
    @Binding var filter: LichessBotOpponentKindFilter

    var body: some View {
        Picker("Opponents", selection: $filter) {
            ForEach(LichessBotOpponentKindFilter.allCases, id: \.self) { kind in
                Text(kind.label).tag(kind)
            }
        }
        .pickerStyle(.menu)
        .fixedSize()
        .help(LichessBotOpponentKindFilter.helpText)
    }
}
