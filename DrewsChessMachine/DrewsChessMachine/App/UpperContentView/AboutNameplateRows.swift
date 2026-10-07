import SwiftUI

/// The About popover's Name, Preset and File format rows for the champion
/// (`ModelNameplate`). Every row is always shown, stating "none given",
/// "none (custom architecture)" or "not recorded" rather than hiding, so an
/// absent value is never confused with one the popover failed to show.
struct AboutNameplateRows: View {
    let nameplate: ModelNameplate

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Name: \(ModelNaming.nameRowText(nameplate.naming))")
            Text("Preset: \(ModelNaming.presetRowText(nameplate.naming))")
            Text("File format: \(nameplate.formatRowText)")
        }
        .font(.system(.callout, design: .monospaced))
        .textSelection(.enabled)
    }
}
