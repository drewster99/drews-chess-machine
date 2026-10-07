import SwiftUI

/// Under the New Network screen's Name field: what the model will record as
/// its name and starting preset (`BuildNewModelModel.makeModelNaming`), or
/// why the typed name can't be used (Build is disabled then).
struct BuildNameStatusText: View {
    let model: BuildNewModelModel

    var body: some View {
        Text(model.nameError ?? Self.recordedText(model: model))
            .font(.caption)
            .foregroundStyle(model.nameError == nil ? AnyShapeStyle(.secondary) : AnyShapeStyle(Color.orange))
    }

    /// "Records name "my-net", preset v4_5block_7x7 (edited)".
    private static func recordedText(model: BuildNewModelModel) -> String {
        do {
            let naming = try model.makeModelNaming()
            return "Records " + ModelNaming.recordedSummaryText(naming)
        } catch {
            return String(describing: error)
        }
    }
}
