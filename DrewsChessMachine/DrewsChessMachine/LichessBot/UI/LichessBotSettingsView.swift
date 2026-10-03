import SwiftUI

/// Every bot setting (plan §12), on tabs. Edits apply as they are made;
/// settings that fail validation are shown with their problems and not
/// applied, and the last valid settings stay in force.
///
/// The tab picker and the problem header sit above the tabs, so one draft
/// covers every tab and a rejected edit stays visible whichever tab is
/// shown. A tab whose own fields have a problem is marked in the picker
/// (see `LichessBotSettingsTab` for how problems are attributed), so a
/// problem on a tab that isn't shown can still be found. A rejected edit
/// never switches tabs by itself: edits apply as they are typed, so a
/// switch would pull the operator off the field they are typing in on every
/// rejected keystroke.
///
/// Every tab stays mounted, hidden rather than removed, so state a section
/// keeps for itself (a half-typed token, the blocked-players text) survives
/// switching tabs, as it did when all sections shared one scroll.
struct LichessBotSettingsView: View {
    let controller: LichessBotController
    @State private var draft: LichessBotSettings
    @State private var problems: String?
    /// Bumped on Reset to Defaults: sections that keep their own editing
    /// text (blocked players, the account id) are rebuilt and re-read it.
    @State private var resetGeneration = 0
    /// The tab shown. Only a pick made in the picker is also remembered
    /// (`controller.rememberedSettingsTab`); Account, which this view
    /// selects by itself when the token check finds no token, is shown but
    /// not remembered.
    @State private var selectedTab: LichessBotSettingsTab

    init(controller: LichessBotController) {
        self.controller = controller
        _draft = State(initialValue: controller.settings)
        _selectedTab = State(initialValue: LichessBotSettingsTab.opening(
            tokenState: controller.tokenState,
            remembered: controller.rememberedSettingsTab
        ))
    }

    var body: some View {
        // Derived from the draft on every render rather than stored, so the
        // marks can never disagree with what the draft holds.
        let tabsWithProblems = LichessBotSettingsTab.tabsWithProblems(in: draft, comparedWith: controller.settings)
        // The picker is the only place the operator picks a tab, so the pick
        // is remembered right here, where it is known to be theirs; the
        // getter is `selectedTab` itself.
        let operatorSelection = Binding<LichessBotSettingsTab>(
            get: { selectedTab },
            set: { tab in
                selectedTab = tab
                controller.rememberedSettingsTab = tab
            }
        )
        VStack(alignment: .leading, spacing: 0) {
            LichessBotSettingsTabPicker(selection: operatorSelection, tabsWithProblems: tabsWithProblems)
                .frame(maxWidth: .infinity)
                .padding(.horizontal, 16)
                .padding(.top, 12)
                .padding(.bottom, 4)
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
            ZStack {
                LichessBotSettingsTabPage(isSelected: selectedTab == .games) {
                    LichessBotChallengeSettingsSection(settings: $draft.challenge)
                    LichessBotMatchmakingSettingsSection(settings: $draft.matchmaking)
                }
                LichessBotSettingsTabPage(isSelected: selectedTab == .play) {
                    LichessBotPlaySettingsSection(settings: $draft.play)
                    LichessBotModelSettingsSection(settings: $draft.model)
                }
                LichessBotSettingsTabPage(isSelected: selectedTab == .chat) {
                    LichessBotChatSettingsSection(settings: $draft.chat)
                }
                LichessBotSettingsTabPage(isSelected: selectedTab == .alerts) {
                    LichessBotAlertSettingsSection(settings: $draft.alerts)
                }
                LichessBotSettingsTabPage(isSelected: selectedTab == .connection) {
                    LichessBotConnectionSettingsSection(settings: $draft.connection, display: $draft.display)
                }
                LichessBotSettingsTabPage(isSelected: selectedTab == .account) {
                    LichessBotAccountSettingsSection(controller: controller, expectedAccountID: $draft.connection.expectedAccountID)
                }
            }
            .id(resetGeneration)
        }
        .onChange(of: draft) {
            Task { @MainActor in
                apply()
            }
        }
        // Moves to Account only as the token state becomes "no token": the
        // handler runs once when the view appears (`initial` — the check may
        // already have finished) and afterwards only when the state changes,
        // so a "no token" that persists never pulls the operator back to
        // Account after they pick another tab.
        .onChange(of: controller.tokenState, initial: true) { _, tokenState in
            Task { @MainActor in
                if case .none = tokenState {
                    selectedTab = .account
                }
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
            resetGeneration += 1
            problems = nil
        } catch {
            problems = error.localizedDescription
        }
    }
}

// MARK: - Tabs

/// The Settings tabs as one segmented control — the same control the bot
/// window uses for its other in-section switches (Single / Grid, the game
/// panels, the challenge sheet's player lists). A tab with a problem in its
/// own fields carries a warning sign in its title. The sign is part of the
/// title text, not a separate image or color, because a segment shows its
/// title text reliably and may drop an embedded image or a text color.
struct LichessBotSettingsTabPicker: View {
    @Binding var selection: LichessBotSettingsTab
    let tabsWithProblems: [LichessBotSettingsTab]

    var body: some View {
        Picker("Settings", selection: $selection) {
            ForEach(LichessBotSettingsTab.allCases) { tab in
                // U+26A0 WARNING SIGN with U+FE0E: the text glyph, not the
                // emoji, so it takes the segment's own text color.
                Text(tabsWithProblems.contains(tab) ? "\(tab.title) \u{26A0}\u{FE0E}" : tab.title)
                    .accessibilityLabel(tabsWithProblems.contains(tab) ? "\(tab.title), has a problem" : tab.title)
                    .tag(tab)
            }
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        // Wide enough that a warning sign never truncates a title, narrow
        // enough to fit beside the sidebar at the window's minimum width.
        .frame(maxWidth: 600)
    }
}

/// One tab's form. Hidden tabs keep their full size and stay mounted, so a
/// form keeps its scroll position and its sections keep their own state;
/// they are disabled while hidden so keyboard focus can't land in a field
/// nobody can see.
struct LichessBotSettingsTabPage<Content: View>: View {
    let isSelected: Bool
    let content: Content

    init(isSelected: Bool, @ViewBuilder content: () -> Content) {
        self.isSelected = isSelected
        self.content = content()
    }

    var body: some View {
        Form {
            content
        }
        .formStyle(.grouped)
        .opacity(isSelected ? 1 : 0)
        .allowsHitTesting(isSelected)
        .accessibilityHidden(!isSelected)
        .disabled(!isSelected)
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
                        Button(
                            action: {
                                if settings.allowedSpeeds.contains(speed) {
                                    settings.allowedSpeeds.remove(speed)
                                } else {
                                    settings.allowedSpeeds.insert(speed)
                                }
                            },
                            label: {
                                Label(speed.rawValue, systemImage: settings.allowedSpeeds.contains(speed) ? "checkmark.square.fill" : "square")
                            }
                        )
                        .buttonStyle(.borderless)
                    }
                }
            }
            LichessBotIntegerField(label: "Clock at least", value: $settings.minimumInitialSeconds, unit: "s")
            LichessBotIntegerField(label: "Clock at most", value: $settings.maximumInitialSeconds, unit: "s")
            LichessBotIntegerField(label: "Increment at least", value: $settings.minimumIncrementSeconds, unit: "s")
            LichessBotIntegerField(label: "Increment at most", value: $settings.maximumIncrementSeconds, unit: "s")
            LichessBotRuledOutSpeedsWarning(speeds: settings.speedsRuledOutByClockBounds)
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
            LichessBotIntegerField(label: "Withdraw unanswered challenges after", value: $settings.outgoingChallengeTimeoutSeconds, unit: "s (0 = never)")
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

/// Automatic challenges to online bots (plan §7.3 B).
struct LichessBotMatchmakingSettingsSection: View {
    @Binding var settings: LichessBotMatchmakingSettings

    var body: some View {
        Section("Matchmaking — applies from the next pass") {
            Toggle("Challenge online bots automatically", isOn: $settings.enabled)
            Picker("Fill", selection: $settings.fillMode) {
                Text("Every free slot").tag(LichessBotMatchmakingSettings.FillMode.everyFreeSlot)
                Text("Only when idle").tag(LichessBotMatchmakingSettings.FillMode.onlyWhenIdle)
            }
            .help("Slots reserved for humans are never used. \"Only when idle\" sends one challenge at a time, only when nothing is in play.")
            LabeledContent("Time controls") {
                // Wraps: the full list is wider than the form.
                LazyVGrid(columns: Array(repeating: GridItem(.fixed(72), alignment: .leading), count: 5), alignment: .leading, spacing: 4) {
                    ForEach(LichessBotClockChoice.allCases) { choice in
                        Button(
                            action: {
                                if settings.timeControls.contains(choice) {
                                    settings.timeControls.remove(choice)
                                } else {
                                    settings.timeControls.insert(choice)
                                }
                            },
                            label: {
                                Label(choice.rawValue, systemImage: settings.timeControls.contains(choice) ? "checkmark.square.fill" : "square")
                                    .font(.system(.body, design: .monospaced))
                            }
                        )
                        .buttonStyle(.borderless)
                        .help("\(choice.rawValue) is \(choice.speed.rawValue)")
                    }
                }
                .frame(width: 380, alignment: .leading)
            }
            Toggle("Rated", isOn: $settings.rated)
            Toggle("Fall back to casual when asked", isOn: $settings.fallBackToCasual)
                .disabled(!settings.rated)
                .help("When a bot declines one of matchmaking's rated challenges asking for a casual game instead, send it the same challenge once more as casual. Matchmaking's limits still apply; if one stops the resend, or the bot declines it, the bot gets the usual decline cool-down. Challenges you send yourself still get the manual Resend as Casual offer.")
            LichessBotIntegerField(label: "Opponent rating from DCM's rating plus", value: $settings.minimumRatingOffset)
            LichessBotIntegerField(label: "  … up to DCM's rating plus", value: $settings.maximumRatingOffset)
            LichessBotIntegerField(label: "Without an established DCM rating, from", value: $settings.minimumRatingWithoutOwnRating)
            LichessBotIntegerField(label: "  … up to", value: $settings.maximumRatingWithoutOwnRating)
            Toggle("Prefer favorites", isOn: $settings.preferFavorites)
            LichessBotIntegerField(label: "Challenges at most", value: $settings.maxChallengesPerHour, unit: "/h")
            LichessBotIntegerField(label: "Leave a bot that declined alone for", value: $settings.declineCooldownHours, unit: "h")
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
    @State private var showingLinePicker = false

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
                    Button("Latest by lineage") {
                        showingLinePicker = true
                    }
                    .help("Every model lineage in the models folder, with its latest file")
                    Button("Choose File…") {
                        chooseFile()
                    }
                }
            }
            .disabled(settings.source != .file)
            .sheet(isPresented: $showingLinePicker) {
                LichessBotModelLinePicker(isPresented: $showingLinePicker) { url in
                    settings.filePath = url.path
                }
            }
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
        Section("Connection") {
            Toggle("Prevent system sleep while online (applies when the bot next goes online)", isOn: $settings.preventSleepWhileOnline)
            LichessBotIntegerField(label: "Reconnect after", value: $settings.reconnectInitialSeconds, unit: "s")
            LichessBotIntegerField(label: "Reconnect delay at most", value: $settings.reconnectCapSeconds, unit: "s")
            LichessBotIntegerField(label: "Event stream silent for", value: $settings.eventStreamStallTimeoutSeconds, unit: "s")
            LichessBotIntegerField(label: "Game stream resync after", value: $settings.gameStreamResyncSeconds, unit: "s")
            LichessBotIntegerField(label: "Rate-limit breaker window", value: $settings.rateLimitBreakerWindowMinutes, unit: "min")
            LichessBotIntegerField(label: "Export spacing", value: $settings.exportMinimumSpacingSeconds, unit: "s")
            LichessBotIntegerField(label: "Low-clock threshold", value: $settings.lowClockThresholdMilliseconds, unit: "ms")
        }
        Section("Live games") {
            LichessBotIntegerField(label: "Keep finished games in the grid for", value: $display.finishedGameRetentionMinutes, unit: "min")
        }
    }
}

/// Warns when a checked speed can never be accepted because the clock and
/// increment bounds exclude every clock at that speed.
struct LichessBotRuledOutSpeedsWarning: View {
    let speeds: [LichessBotSpeed]

    var body: some View {
        Label(text, systemImage: "exclamationmark.triangle.fill")
            .font(.callout)
            .foregroundStyle(.orange)
            .shown(!speeds.isEmpty)
    }

    private var text: String {
        let names = speeds.map(\.rawValue).joined(separator: ", ")
        return "No clock within these bounds is \(names), so \(speeds.count == 1 ? "that speed is" : "those speeds are") never accepted"
    }
}
