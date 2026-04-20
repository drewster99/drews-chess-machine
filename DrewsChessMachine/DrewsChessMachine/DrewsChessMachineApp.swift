import AppKit
import SwiftUI

@main
struct DrewsChessMachineApp: App {
    /// Single shared command hub that bridges the menu bar commands
    /// to `ContentView`'s state and action functions. Owned here at
    /// the `App` level so the `.commands` DSL below and the
    /// `ContentView` in `WindowGroup` see the same instance.
    @State private var commandHub = AppCommandHub()

    init() {
        // Start the session logger before any view work so every event
        // from this launch — button taps, arena results, periodic
        // stats — lands in a single `dcm_log_yyyymmdd-HHMMSS.txt`
        // file under the app's Library/Logs directory.
        SessionLogger.shared.start()
        let dirtyMarker = BuildInfo.gitDirty ? "*" : ""
        let archHashHex = String(format: "0x%08x", ModelCheckpointFile.currentArchHash)
        SessionLogger.shared.log(
            "[APP] launched build=\(BuildInfo.buildNumber) git=\(BuildInfo.gitHash)\(dirtyMarker) branch=\(BuildInfo.gitBranch) date=\(BuildInfo.buildDate) timestamp=\(BuildInfo.buildTimestamp) arch_hash=\(archHashHex) inputPlanes=\(ChessNetwork.inputPlanes) policySize=\(ChessNetwork.policySize)"
        )
        if let path = SessionLogger.shared.activeLogPath {
            SessionLogger.shared.log("[APP] session log: \(path)")
            print("[APP] session log: \(path)")
        } else {
            print("[APP] session log: (failed to open)")
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView(commandHub: commandHub)
        }
        .commands {
            // File menu additions — Save / Load / reveal-in-Finder.
            // Placed after the standard "New" slot so they appear at
            // the top of the File menu alongside the other
            // file-scope operations.
            CommandGroup(after: .newItem) {
                Divider()
                Button("Save Session") { commandHub.saveSession() }
                    .disabled(
                        !commandHub.realTraining
                        || commandHub.isArenaRunning
                        || commandHub.checkpointSaveInFlight
                    )
                Button("Save Champion") { commandHub.saveChampion() }
                    .disabled(
                        !commandHub.networkReady
                        || commandHub.checkpointSaveInFlight
                        || commandHub.isArenaRunning
                        || (commandHub.isBusy && !commandHub.realTraining)
                    )
                Divider()
                Button("Load Session…") { commandHub.loadSession() }
                    .disabled(
                        commandHub.realTraining
                        || commandHub.continuousPlay
                        || commandHub.continuousTraining
                        || commandHub.sweepRunning
                        || commandHub.gameIsPlaying
                        || commandHub.isBuilding
                        || commandHub.checkpointSaveInFlight
                    )
                Button("Load Model…") { commandHub.loadModel() }
                    .disabled(
                        commandHub.realTraining
                        || commandHub.continuousPlay
                        || commandHub.continuousTraining
                        || commandHub.sweepRunning
                        || commandHub.gameIsPlaying
                        || commandHub.isBuilding
                        || commandHub.checkpointSaveInFlight
                        || !commandHub.networkReady
                    )
                Divider()
                Button("Open Data Folder in Finder") { commandHub.revealSaves() }
            }

            // Train menu — the primary training-session lifecycle
            // plus the arena-stage controls. SwiftUI places
            // `CommandMenu` entries before Window; on a standard
            // macOS menu bar we get: File Edit View Train Debug
            // Window Help. "Debug between Window and Help" isn't
            // reachable via `CommandMenu`, so Debug lands adjacent
            // to Train (before Window) as the closest SwiftUI
            // approximation.
            CommandMenu("Train") {
                Button("Build Network") { commandHub.buildNetwork() }
                    .disabled(commandHub.isBusy || commandHub.networkReady)
                Button(commandHub.pendingLoadedSessionExists ? "Continue Training" : "Play and Train") {
                    commandHub.startRealTraining()
                }
                .disabled(
                    commandHub.isBusy
                    || !commandHub.networkReady
                    || commandHub.realTraining
                    || commandHub.continuousPlay
                    || commandHub.continuousTraining
                    || commandHub.sweepRunning
                )
                Divider()
                Button("Stop") { commandHub.stopAnyContinuous() }
                    .keyboardShortcut(.escape, modifiers: [])
                    .disabled(
                        !(commandHub.continuousPlay
                          || commandHub.continuousTraining
                          || commandHub.sweepRunning
                          || commandHub.realTraining)
                    )
                Divider()
                Button("Run Arena") { commandHub.runArena() }
                    .disabled(!commandHub.realTraining || commandHub.isArenaRunning)
                Button("Abort Arena") { commandHub.abortArena() }
                    .disabled(!commandHub.realTraining || !commandHub.isArenaRunning)
                Button("Promote Trainee") { commandHub.promoteCandidate() }
                    .disabled(!commandHub.realTraining || !commandHub.isArenaRunning)
            }

            CommandMenu("Debug") {
                Button("Run Forward Pass") { commandHub.runForwardPass() }
                    .keyboardShortcut(.return, modifiers: [])
                    .disabled(commandHub.isBusy || !commandHub.networkReady)
                Divider()
                Button("Play Game") { commandHub.playSingleGame() }
                    .disabled(commandHub.isBusy || !commandHub.networkReady)
                Button("Play Continuous") { commandHub.startContinuousPlay() }
                    .disabled(
                        commandHub.isBusy
                        || !commandHub.networkReady
                        || commandHub.continuousPlay
                        || commandHub.continuousTraining
                        || commandHub.realTraining
                    )
                Divider()
                Button("Train Once") { commandHub.trainOnce() }
                    .disabled(commandHub.isBusy || !commandHub.networkReady)
                Button("Train Continuous") { commandHub.startContinuousTraining() }
                    .disabled(
                        commandHub.isBusy
                        || !commandHub.networkReady
                        || commandHub.continuousTraining
                        || commandHub.continuousPlay
                        || commandHub.sweepRunning
                        || commandHub.realTraining
                    )
                Divider()
                Button("Sweep Batch Sizes") { commandHub.startSweep() }
                    .disabled(
                        commandHub.isBusy
                        || !commandHub.networkReady
                        || commandHub.sweepRunning
                        || commandHub.continuousPlay
                        || commandHub.continuousTraining
                        || commandHub.realTraining
                    )
                Divider()
                Button("Run Engine Diagnostics") { commandHub.runEngineDiagnostics() }
                    .disabled(commandHub.isBusy)
                Divider()
                Button("Open Session Log") {
                    if let path = SessionLogger.shared.activeLogPath {
                        NSWorkspace.shared.open(URL(fileURLWithPath: path))
                    }
                }
                .disabled(SessionLogger.shared.activeLogPath == nil)
                Button("Analyze Log") {
                    LogAnalysisLauncher.openWindow()
                }
                .disabled(SessionLogger.shared.activeLogPath == nil)
            }
        }
    }
}
