import SwiftUI

/// The training-settings popover's Health tab (alarms plan Part K item 7):
/// one aligned row per rule — its name, what it watches, and an action
/// picker — then the enable switch, the check interval and the learning
/// grace. Commit-on-Save through `TrainingSettingsPopoverModel`, which logs
/// each change as a `[PARAM]` line.
///
/// Rules without a critical level (loss spike, policy offset drift, BN
/// running variance, gradient spike) cannot stop on critical: that picker
/// choice is disabled and the row says "no critical level".
struct TrainingHealthTab: View {
    @Binding var alarmsEnabled: Bool
    @Binding var checkIntervalText: String
    @Binding var learningGraceText: String
    @Binding var actions: TrainingHealthActions
    let checkIntervalError: Bool
    let learningGraceError: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Rules")
                .font(.subheadline.weight(.semibold))
            VStack(alignment: .leading, spacing: 8) {
                ForEach(TrainingHealthRule.allCases, id: \.self) { rule in
                    TrainingHealthRuleRow(rule: rule, action: $actions[rule])
                }
            }
            Text("Every rule logs by default ([ALARM] health lines, the list under the banner, results.json). A stop ends a command-line run through its final save (exit status 35); in the app it suspends training — self-play and the periodic autosave go on, arenas and Promote Trainee Now are refused — until Stop. For unattended runs consider Stop on critical for non-finite values, illegal-move mass, gradient collapse and dead channels.")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Divider()

            Toggle("Training health alarms enabled", isOn: $alarmsEnabled)
                .toggleStyle(.checkbox)
            TrainingHealthStepsField(
                label: "Check line every (steps):",
                text: $checkIntervalText,
                error: checkIntervalError,
                range: TrainingHealthCheckIntervalSteps.declaredClosedRange,
                hint: "[HEALTH] check line, active-alarm reminders and the worsen rate limit. The rules themselves run every 50 trainer steps.")
            TrainingHealthStepsField(
                label: "Learning grace (steps):",
                text: $learningGraceText,
                error: learningGraceError,
                range: TrainingHealthLearningGraceSteps.declaredClosedRange,
                hint: "Added to LR warmup before illegal-move mass counts as not learned.")
        }
    }
}

/// One rule: name and meaning on the left, the action picker on the right,
/// aligned across rows by fixed column widths.
struct TrainingHealthRuleRow: View {
    let rule: TrainingHealthRule
    @Binding var action: TrainingHealthAction

    var body: some View {
        let hasCritical = rule.hasCriticalLevel
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(rule.displayName)
                    .font(.callout.weight(.medium))
                Text(rule.meaning + (hasCritical ? "" : " · no critical level"))
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Picker("", selection: $action) {
                ForEach(TrainingHealthAction.allCases, id: \.self) { choice in
                    Text(choice.displayName)
                        .tag(choice)
                        .selectionDisabled(choice == .stopOnCritical && !hasCritical)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(width: 160)
            .accessibilityLabel("\(rule.displayName) action")
        }
    }
}

/// A whole-number trainer-step field with its declared range, monospaced
/// digits, and a red outline while Save has flagged it.
struct TrainingHealthStepsField: View {
    let label: String
    @Binding var text: String
    let error: Bool
    let range: ClosedRange<Int>
    let hint: String

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(label)
                .frame(width: 180, alignment: .trailing)
            TextField("", text: $text)
                .font(.system(.body, design: .monospaced))
                .multilineTextAlignment(.trailing)
                .frame(width: 90)
                .overlay(
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(Color.red, lineWidth: error ? 1.5 : 0)
                )
                .accessibilityLabel(label)
            Text("\(range.lowerBound)…\(range.upperBound)")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.secondary)
            Text(hint)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
