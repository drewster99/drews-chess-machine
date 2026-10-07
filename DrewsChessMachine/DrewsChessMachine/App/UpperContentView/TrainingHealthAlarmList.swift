import SwiftUI

/// The active training-health alarms, under the training-alarm banner (the
/// alarms plan T9). One aligned row per alarm, in rule order, mirroring the
/// monitor's active set through `TrainingAlarmController.healthAlarms`.
///
/// Always in the hierarchy: when there is nothing to show it is hidden with
/// opacity 0 and a zero frame in both dimensions, never removed with an
/// `if`, so its identity is stable and its appearance does not rebuild the
/// surrounding stack. The same holds for the suspension header and the
/// Silence button.
///
/// Colors are semantic (`.orange` warning / `.red` critical symbols on
/// primary text over the secondary background), so the list reads in light
/// and dark mode alike; the banner's fixed yellow is not reused.
struct TrainingHealthAlarmList: View {
    let alarmController: TrainingAlarmController

    var body: some View {
        let alarms = alarmController.healthAlarms
        let isEmpty = alarms.isEmpty
        let suspendedRule = alarmController.healthSuspendedRule
        let showsSilence = alarmController.shouldSound
            && alarms.contains { $0.severity == .critical }
        VStack(alignment: .leading, spacing: 4) {
            TrainingHealthAlarmListHeader(
                suspendedRule: suspendedRule,
                showsSilence: showsSilence,
                onSilence: { alarmController.silence() })
            ForEach(alarms, id: \.rule) { alarm in
                TrainingHealthAlarmRow(alarm: alarm)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, isEmpty ? 0 : 6)
        .frame(maxWidth: isEmpty ? 0 : .infinity, maxHeight: isEmpty ? 0 : nil, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 6))
        .opacity(isEmpty ? 0 : 1)
        .accessibilityHidden(isEmpty)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Training health alarms")
    }
}

/// The list's header: its title, the suspension notice (when a health stop
/// suspended training) and the Silence button (while a critical alarm is
/// sounding). Hidden parts keep their place in the tree.
struct TrainingHealthAlarmListHeader: View {
    let suspendedRule: TrainingHealthRule?
    let showsSilence: Bool
    let onSilence: () -> Void

    var body: some View {
        let suspensionText = suspendedRule.map {
            "Training suspended by \($0.displayName). Stop to clear; set its action to Log to keep training."
        } ?? ""
        HStack(spacing: 8) {
            Text("Training health")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(suspensionText)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.red)
                .opacity(suspendedRule == nil ? 0 : 1)
                .frame(maxWidth: suspendedRule == nil ? 0 : nil)
                .accessibilityHidden(suspendedRule == nil)
            Spacer(minLength: 0)
            Button("Silence", action: onSilence)
                .controlSize(.small)
                .opacity(showsSilence ? 1 : 0)
                .frame(width: showsSilence ? nil : 0, height: showsSilence ? nil : 0)
                .disabled(!showsSilence)
                .accessibilityHidden(!showsSilence)
        }
    }
}

/// One active alarm: severity symbol, rule name, the measured value
/// (monospaced), the trainer step it was raised at (monospaced, padded), and
/// what its rule's current action does at this severity.
struct TrainingHealthAlarmRow: View {
    let alarm: TrainingHealthActiveAlarm

    var body: some View {
        let isCritical = alarm.severity == .critical
        let action = TrainingParameters.shared[keyPath: TrainingParameters.trainingHealthActionKeyPath(for: alarm.rule)]
        let stops = TrainingHealthStopPolicy.qualifies(severity: alarm.severity, action: action)
        let since = String(alarm.since)
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: isCritical ? "xmark.octagon.fill" : "exclamationmark.triangle.fill")
                .foregroundStyle(isCritical ? Color.red : Color.orange)
                .frame(width: 16)
            Text(alarm.rule.displayName)
                .font(.callout.weight(.medium))
                .frame(width: 170, alignment: .leading)
            Text(alarm.value)
                .font(.system(.callout, design: .monospaced))
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(maxWidth: .infinity, alignment: .leading)
            Text("since step " + String(repeating: " ", count: max(0, 9 - since.count)) + since)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
            Text(stops ? "stops run" : "log")
                .font(.caption)
                .foregroundStyle(stops ? Color.red : Color.secondary)
                .frame(width: 60, alignment: .trailing)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(
            "\(isCritical ? "Critical" : "Warning"): \(alarm.rule.displayName), \(alarm.value), "
                + "since trainer step \(alarm.since), \(stops ? "stops the run" : "logged only")")
    }
}
