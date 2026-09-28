import SwiftUI

/// Every bot setting (plan §12). Edits apply as they are made; settings that
/// fail validation are shown with their problems and not applied, and the
/// last valid settings stay in force.
struct LichessBotSettingsView: View {
    let controller: LichessBotController
    @State private var draft: LichessBotSettings
    @State private var problems: String?

    init(controller: LichessBotController) {
        self.controller = controller
        _draft = State(initialValue: controller.settings)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                Text(controller.settingsError ?? "")
                    .foregroundStyle(.red)
                    .shown(controller.settingsError != nil)
                Button("Reset to Defaults") {
                    resetAll()
                }
                .shown(controller.settingsError != nil)
                Text(problems ?? "")
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
                    .shown(problems != nil)
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, (controller.settingsError != nil || problems != nil) ? 8 : 0)
            Form {
                LichessBotAccountSettingsSection(controller: controller, expectedAccountID: $draft.connection.expectedAccountID)
                LichessBotChallengeSettingsSection(settings: $draft.challenge)
                LichessBotPlaySettingsSection(settings: $draft.play)
                LichessBotChatSettingsSection(settings: $draft.chat)
                LichessBotModelSettingsSection(settings: $draft.model)
                LichessBotConnectionSettingsSection(settings: $draft.connection, display: $draft.display)
            }
            .formStyle(.grouped)
        }
        .onChange(of: draft) {
            Task { @MainActor in
                apply()
            }
        }
    }

    private func apply() {
        guard draft != controller.settings else {
            problems = nil
            return
        }
        do {
            try controller.updateSettings(draft)
            problems = nil
        } catch {
            problems = error.localizedDescription
        }
    }

    private func resetAll() {
        do {
            try controller.resetSettings()
            draft = controller.settings
            problems = nil
        } catch {
            problems = error.localizedDescription
        }
    }
}

// MARK: - Field helpers

/// A labeled whole-number field with monospaced digits.
struct LichessBotIntegerField: View {
    let label: String
    @Binding var value: Int
    var unit: String = ""

    var body: some View {
        LabeledContent(label) {
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                TextField(label, value: $value, format: .number.grouping(.never))
                    .labelsHidden()
                    .font(.system(.body, design: .monospaced))
                    .multilineTextAlignment(.trailing)
                    .frame(width: 90)
                Text(unit)
                    .foregroundStyle(.secondary)
                    .frame(width: 44, alignment: .leading)
            }
        }
    }
}

/// A labeled decimal field with monospaced digits.
struct LichessBotDecimalField: View {
    let label: String
    @Binding var value: Float

    var body: some View {
        LabeledContent(label) {
            HStack(spacing: 4) {
                TextField(label, value: $value, format: .number.precision(.fractionLength(2...3)))
                    .labelsHidden()
                    .font(.system(.body, design: .monospaced))
                    .multilineTextAlignment(.trailing)
                    .frame(width: 90)
                Spacer()
                    .frame(width: 48)
            }
        }
    }
}

// MARK: - Sections

/// Challenge acceptance (plan §7).
struct LichessBotChallengeSettingsSection: View {
    @Binding var settings: LichessBotChallengeSettings
    @State private var blockedText = ""

    var body: some View {
        Section("Challenges — applies to the next challenge") {
            Toggle("Accept casual games", isOn: $settings.acceptCasual)
            Toggle("Accept rated games", isOn: $settings.acceptRated)
            LabeledContent("Speeds") {
                HStack {
                    ForEach([LichessBotSpeed.ultraBullet, .bullet, .blitz, .rapid, .classical], id: \.self) { speed in
                        Button {
                            if settings.allowedSpeeds.contains(speed) {
                                settings.allowedSpeeds.remove(speed)
                            } else {
                                settings.allowedSpeeds.insert(speed)
                            }
                        } label: {
                            Label(speed.rawValue, systemImage: settings.allowedSpeeds.contains(speed) ? "checkmark.square.fill" : "square")
                        }
                        .buttonStyle(.borderless)
                    }
                }
            }
            LichessBotIntegerField(label: "Clock at least", value: $settings.minimumInitialSeconds, unit: "s")
            LichessBotIntegerField(label: "Clock at most", value: $settings.maximumInitialSeconds, unit: "s")
            LichessBotIntegerField(label: "Increment at least", value: $settings.minimumIncrementSeconds, unit: "s")
            LichessBotIntegerField(label: "Increment at most", value: $settings.maximumIncrementSeconds, unit: "s")
            Toggle("Accept humans", isOn: $settings.acceptHumans)
            Toggle("Accept bots", isOn: $settings.acceptBots)
            Toggle("Accept provisional opponents", isOn: $settings.acceptProvisionalOpponents)
            Toggle("Accept rematches", isOn: $settings.acceptRematches)
            LichessBotIntegerField(label: "Opponent rating at least", value: $settings.minimumOpponentRating)
            LichessBotIntegerField(label: "Opponent rating at most", value: $settings.maximumOpponentRating)
            LichessBotIntegerField(label: "Games at once", value: $settings.maxConcurrentGames)
            LichessBotIntegerField(label: "Of those, reserved for humans", value: $settings.gamesReservedForHumans)
            LichessBotIntegerField(label: "Games at once per opponent", value: $settings.maxSimultaneousGamesPerOpponent)
            LichessBotIntegerField(label: "Games per day", value: $settings.maxGamesPerDay)
            LichessBotIntegerField(label: "Games per opponent per day", value: $settings.maxGamesPerOpponentPerDay)
            LichessBotIntegerField(label: "Stop vs one bot at", value: $settings.botPairDailyStop, unit: "/day")
            LichessBotIntegerField(label: "Challenge responses at most", value: $settings.challengeResponseBudgetPerMinute, unit: "/min")
            LabeledContent("Blocked players") {
                TextField("Blocked players", text: $blockedText, prompt: Text("ids, comma-separated"))
                    .labelsHidden()
                    .frame(width: 260)
            }
        }
        .onAppear {
            blockedText = settings.blockedUserIDs.joined(separator: ", ")
        }
        .onChange(of: blockedText) {
            Task { @MainActor in
                settings.blockedUserIDs = blockedText.split(separator: ",")
                    .map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
                    .filter { !$0.isEmpty }
            }
        }
    }
}

/// How the bot plays (plan §12.4).
struct LichessBotPlaySettingsSection: View {
    @Binding var settings: LichessBotPlaySettings

    var body: some View {
        Section("Play — applies from the next move") {
            LichessBotDecimalField(label: "Temperature at the start", value: $settings.temperatureStart)
            LichessBotDecimalField(label: "Temperature decay per ply", value: $settings.temperatureDecayPerPly)
            LichessBotDecimalField(label: "Temperature floor", value: $settings.temperatureFloor)
            LichessBotIntegerField(label: "Minimum think time", value: $settings.minimumThinkMilliseconds, unit: "ms")
            Toggle("Resign when the value head says lost", isOn: $settings.resignEnabled)
            LichessBotDecimalField(label: "  … at p(loss) of at least", value: $settings.resignLossProbability)
            LichessBotIntegerField(label: "  … for this many moves", value: $settings.resignConsecutiveMoves)
            LichessBotIntegerField(label: "  … from ply", value: $settings.resignMinimumPly)
            Toggle("Offer draws when the value head says drawn", isOn: $settings.offerDrawEnabled)
            LichessBotDecimalField(label: "  … at p(draw) of at least", value: $settings.offerDrawProbability)
            LichessBotIntegerField(label: "  … for this many moves", value: $settings.offerDrawConsecutiveMoves)
            LichessBotIntegerField(label: "  … from ply", value: $settings.offerDrawMinimumPly)
            Toggle("Accept draw offers", isOn: $settings.acceptDrawEnabled)
            LichessBotDecimalField(label: "  … when expected score is at most", value: $settings.acceptDrawExpectedScore)
            Toggle("Claim when the opponent leaves", isOn: $settings.claimWhenOpponentGone)
            LichessBotIntegerField(label: "Takebacks accepted per game", value: $settings.maxTakebacksAcceptedPerGame)
        }
    }
}

/// Greeting and goodbye messages (plan §12.5).
struct LichessBotChatSettingsSection: View {
    @Binding var settings: LichessBotChatSettings

    var body: some View {
        Section("Chat — applies to the next game") {
            Toggle("Greet at the start", isOn: $settings.greetingEnabled)
            LabeledContent("Greeting") {
                TextField("Greeting", text: $settings.greetingTemplate)
                    .labelsHidden()
                    .frame(minWidth: 320)
            }
            Toggle("Say goodbye at the end", isOn: $settings.goodbyeEnabled)
            LabeledContent("Goodbye") {
                TextField("Goodbye", text: $settings.goodbyeTemplate)
                    .labelsHidden()
                    .frame(minWidth: 320)
            }
            Picker("Room", selection: $settings.room) {
                Text("Player").tag(LichessBotChatRoom.player)
                Text("Spectator").tag(LichessBotChatRoom.spectator)
            }
            Text("Placeholders: {modelID} {source} {build} {opponent}")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}

/// Which model plays (plan §9).
struct LichessBotModelSettingsSection: View {
    @Binding var settings: LichessBotModelSettings

    var body: some View {
        Section("Model — applies to the next game") {
            Picker("Source", selection: $settings.source) {
                Text("Champion").tag(LichessBotModelSourceKind.champion)
                Text("Trainer snapshot").tag(LichessBotModelSourceKind.trainerSnapshot)
                Text("Live trainer").tag(LichessBotModelSourceKind.liveTrainer)
                Text("Model file").tag(LichessBotModelSourceKind.file)
            }
            // Rows for other sources stay in place, disabled, so the form
            // doesn't reshuffle as the source changes.
            LabeledContent("Model file") {
                HStack {
                    Text(settings.filePath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "None")
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    Button("Choose…") {
                        chooseFile()
                    }
                }
            }
            .disabled(settings.source != .file)
            LichessBotIntegerField(label: "Live-trainer refresh every", value: $settings.liveTrainerRefreshIntervalSeconds, unit: "s")
                .disabled(settings.source != .liveTrainer)
            Toggle("Live trainer: games in progress switch to each new snapshot", isOn: $settings.midGameRefresh)
                .disabled(settings.source != .liveTrainer)
        }
    }

    private func chooseFile() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.directoryURL = CheckpointPaths.modelsDir
        if panel.runModal() == .OK, let url = panel.url {
            settings.filePath = url.path
        }
    }
}

/// Connection, pacing and display (plan §5, §6, §13, §14.3a).
struct LichessBotConnectionSettingsSection: View {
    @Binding var settings: LichessBotConnectionSettings
    @Binding var display: LichessBotDisplaySettings

    var body: some View {
        Section("Connection — applies when the bot next goes online") {
            Toggle("Prevent system sleep while online", isOn: $settings.preventSleepWhileOnline)
            LichessBotIntegerField(label: "Reconnect after", value: $settings.reconnectInitialSeconds, unit: "s")
            LichessBotIntegerField(label: "Reconnect delay at most", value: $settings.reconnectCapSeconds, unit: "s")
            LichessBotIntegerField(label: "Event stream silent for", value: $settings.eventStreamStallTimeoutSeconds, unit: "s")
            LichessBotIntegerField(label: "Game stream resync after", value: $settings.gameStreamResyncSeconds, unit: "s")
            LichessBotIntegerField(label: "Rate-limit breaker window", value: $settings.rateLimitBreakerWindowMinutes, unit: "min")
            LichessBotIntegerField(label: "Export spacing", value: $settings.exportMinimumSpacingSeconds, unit: "s")
            LichessBotIntegerField(label: "Low-clock threshold", value: $settings.lowClockThresholdMilliseconds, unit: "ms")
            LichessBotIntegerField(label: "Keep finished games in the grid for", value: $display.finishedGameRetentionMinutes, unit: "min")
        }
    }
}
