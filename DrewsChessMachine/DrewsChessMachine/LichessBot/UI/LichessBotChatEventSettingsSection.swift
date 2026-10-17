import SwiftUI

/// Settings for messages triggered by game events (chat-events plan §2): a
/// master switch, the spam guards shared by every rule, and one editable
/// rule per trigger. Every rule ships disabled, and the default texts are
/// neutral or self-deprecating — never gloating (owner decision 2026-10-15).
struct LichessBotChatEventSettingsSection: View {
    @Binding var settings: LichessBotChatEventSettings

    var body: some View {
        Section("Event messages — applies to the next game") {
            Toggle("Send messages on game events", isOn: $settings.enabled)
            LichessBotIntegerField(label: "At most per game", value: $settings.maxMessagesPerGame, unit: "msgs")
            LichessBotIntegerField(label: "Only above", value: $settings.minClockSecondsEitherSide, unit: "sec on clock")
            LichessBotIntegerField(label: "At least apart", value: $settings.minSecondsBetweenMessages, unit: "sec")
            LichessBotIntegerField(label: "And at least", value: $settings.minPliesBetweenMessages, unit: "plies")
            ForEach($settings.rules) { $rule in
                LichessBotChatEventRuleRow(rule: $rule)
            }
            Text("Placeholders: {opponent} {winPct} {move} {ply}")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}

/// One event rule: enable it, choose the room, the odds and (where the
/// trigger has one) the numeric threshold, and edit the message variants
/// (one per line; a random one is sent).
struct LichessBotChatEventRuleRow: View {
    @Binding var rule: LichessBotChatEventRule

    var body: some View {
        GroupBox(rule.trigger.title) {
            VStack(alignment: .leading, spacing: 6) {
                Toggle("Enabled", isOn: $rule.enabled)
                LichessBotChatEventRuleControls(rule: $rule)
                TextField("Messages (one per line)", text: variantsText, axis: .vertical)
                    .lineLimit(2...5)
                    .disabled(!rule.enabled)
            }
            .padding(.vertical, 2)
        }
    }

    /// The variants joined for editing; split back on every change. A
    /// computed binding keeps the stored `[String]` the single source of
    /// truth without a shadow `@State` copy.
    private var variantsText: Binding<String> {
        Binding(
            get: { rule.templates.joined(separator: "\n") },
            set: { newValue in
                rule.templates = newValue
                    .split(separator: "\n", omittingEmptySubsequences: true)
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                    .filter { !$0.isEmpty }
            }
        )
    }
}

/// The room, chance and optional threshold controls for one rule, laid out
/// on one line. Whether a threshold is shown is fixed by the trigger, so the
/// conditional does not shuffle the form as values change.
struct LichessBotChatEventRuleControls: View {
    @Binding var rule: LichessBotChatEventRule

    var body: some View {
        HStack(spacing: 12) {
            Picker("Room", selection: $rule.room) {
                Text("Player").tag(LichessBotChatRoom.player)
                Text("Spectator").tag(LichessBotChatRoom.spectator)
            }
            .frame(width: 180)
            LichessBotIntegerField(label: "Chance", value: $rule.chancePercent, unit: "%")
            if let parameter = rule.trigger.parameter {
                LichessBotIntegerField(label: parameter.label, value: $rule.threshold, unit: parameter.unit)
            }
        }
        .disabled(!rule.enabled)
    }
}
