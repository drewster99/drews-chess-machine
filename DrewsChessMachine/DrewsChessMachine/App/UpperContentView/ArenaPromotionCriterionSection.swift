import SwiftUI

/// The promotion-criterion picker and its SPRT hypothesis fields, as they
/// appear in the Arena settings popover.
///
/// Its own `View` struct rather than a `@ViewBuilder` property on
/// `ArenaSettingsPopover`, per the project's one-view-struct-per-file rule
/// (helper `some View` properties other than `body` are not blessed).
///
/// **Why the SPRT fields stay visible under the score threshold.** Hiding them
/// would shrink the popover when the criterion changes, and — worse — leave no
/// on-screen explanation for why a stored `elo1` seems to do nothing. They are
/// dimmed and disabled instead, which says "these exist, they are not in
/// effect right now". The reverse is also true: `# of games` and
/// `Promote threshold` dim under SPRT, because a sequential test decides its
/// own sample size and a fixed cap on it destroys the calibration the test
/// exists to provide.
struct ArenaPromotionCriterionSection: View {
    @Bindable var model: ArenaSettingsPopoverModel

    var body: some View {
        let sprtActive = model.promotionCriterion == .sprt

        VStack(alignment: .leading, spacing: 8) {
            Text("Promotion criterion")
                .font(.subheadline.weight(.semibold))

            Picker("", selection: $model.promotionCriterion) {
                ForEach(ArenaPromotionCriterion.allCases) { criterion in
                    Text(criterion.displayName).tag(criterion)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            Text(model.criterionHint)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            // The hypothesis block. Dimmed and disabled as a unit under the
            // score threshold rather than removed, so the popover keeps its
            // height and the settings stay visibly present-but-inactive.
            VStack(alignment: .leading, spacing: 8) {
                ArenaPopoverField(
                    label: "H₀ elo0:",
                    text: $model.sprtElo0Text,
                    error: model.sprtElo0Error,
                    placeholder: "0",
                    width: 80,
                    hint: "no improvement"
                )
                ArenaPopoverField(
                    label: "H₁ elo1:",
                    text: $model.sprtElo1Text,
                    error: model.sprtElo1Error,
                    placeholder: "10",
                    width: 80,
                    hint: "smallest gain to detect"
                )
                ArenaPopoverField(
                    label: "α (false promote):",
                    text: $model.sprtAlphaText,
                    error: model.sprtAlphaError,
                    placeholder: "0.050",
                    width: 80
                )
                ArenaPopoverField(
                    label: "β (false reject):",
                    text: $model.sprtBetaText,
                    error: model.sprtBetaError,
                    placeholder: "0.050",
                    width: 80
                )
                ArenaPopoverField(
                    label: "Min games:",
                    text: $model.sprtMinGamesText,
                    error: model.sprtMinGamesError,
                    placeholder: "32",
                    width: 80,
                    hint: "before it may fire"
                )
                ArenaPopoverField(
                    label: "Max games:",
                    text: $model.sprtMaxGamesText,
                    error: model.sprtMaxGamesError,
                    placeholder: "20000",
                    width: 80,
                    hint: "0 = unbounded"
                )

                // Cross-field failures (elo1 > elo0, α + β < 1, minGames ≤
                // maxGames) belong to a relation, not a box, so they report
                // here rather than reddening one of the two fields
                // arbitrarily. Kept in the tree at zero height when absent so
                // the popover does not resize as the message comes and goes.
                Text(model.sprtRelationError ?? "")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
                    .opacity(model.sprtRelationError == nil ? 0 : 1)
                    .frame(height: model.sprtRelationError == nil ? 0 : nil)

                Text("A sequential test stops as soon as the evidence is decisive, "
                     + "so it has no fixed game count. Reaching Max games with the "
                     + "evidence still ambiguous is inconclusive — it does not promote, "
                     + "and it is not a rejection.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .disabled(!sprtActive)
            .opacity(sprtActive ? 1 : 0.4)
        }
    }
}
