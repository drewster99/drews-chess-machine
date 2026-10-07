import SwiftUI

/// The pane menu: the first pass and the later panes in two sections.
///
/// A menu, not a segmented control: eleven panes (the first pass and §11's
/// later ones) do not fit one segmented row at the narrowest window, and
/// two segmented rows bound to one selection would leave one row with a
/// selection it has no tag for, which SwiftUI reports as an invalid
/// selection at run time.
struct LichessBotRecordPanePicker: View {
    @Binding var pane: LichessBotRecordPane

    var body: some View {
        Picker("Show", selection: $pane) {
            Section {
                ForEach(LichessBotRecordPane.firstPass) { pane in
                    Text(pane.label).tag(pane)
                }
            }
            Section {
                ForEach(LichessBotRecordPane.later) { pane in
                    Text(pane.label).tag(pane)
                }
            }
        }
        .fixedSize()
    }
}
