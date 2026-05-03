import Foundation

@MainActor
final class LogAnalysisViewModel: ObservableObject {
    let logPath: String
    let logContent: String
    let claudePath: String

    @Published var claudeResponse: String = ""
    @Published var errorMessage: String?
    @Published var isAnalyzing: Bool = false

    /// Reference to the live `claude` subprocess so `cancel()` can
    /// terminate it when the window is closed. Written only from
    /// the main actor; the background runner hops back to the main
    /// thread to assign/clear it.
    private var activeProcess: Process?
    private var analysisTask: Task<Void, Never>?

    /// Prompt sent to `claude -p`. Kept here as a constant so the
    /// exact wording the user chose lives in one place.
    static let analysisPrompt = """
    Analyze this log. It is from an app called 'drews-chess-machine', \
    which is a project to train a convolutional neural network to \
    play chess, based on solely self-play (no MCTS or expert game \
    data). Summarize what you find, point out deficiencies, issues, \
    positive notes, and make any suggestions you can that will be \
    helpful.
    """

    init(logPath: String, logContent: String, claudePath: String) {
        self.logPath = logPath
        self.logContent = logContent
        self.claudePath = claudePath
    }

    deinit {
        // Nonisolated deinit — can run on any thread. `Process.terminate`
        // is thread-safe, so a best-effort kill here is fine if the
        // view model is torn down without an explicit `cancel()`.
        activeProcess?.terminate()
    }

    /// Idempotent start. Called once by the window controller after
    /// init; subsequent calls are no-ops so a view refresh can't
    /// trigger a second subprocess.
    func start() {
        guard analysisTask == nil else { return }
        isAnalyzing = true
        errorMessage = nil
        claudeResponse = ""
        let capturedPath = logPath
        let capturedClaude = claudePath
        analysisTask = Task { [weak self] in
            await self?.runAnalysis(
                logPath: capturedPath,
                claudePath: capturedClaude
            )
        }
    }

    /// Best-effort cancel. Terminates the live process if one is
    /// running and cancels the waiting task. Safe to call more than
    /// once.
    func cancel() {
        analysisTask?.cancel()
        activeProcess?.terminate()
        activeProcess = nil
    }

    private func runAnalysis(logPath: String, claudePath: String) async {
        let prompt = Self.analysisPrompt

        // Run the cat | claude pipeline on a background GCD queue
        // so the main actor and Swift's cooperative executor stay
        // free during the potentially multi-second subprocess
        // lifetime. The run/waitUntilExit pair is blocking — doing
        // it inline inside a Task would pin a cooperative thread
        // for the duration, which is exactly what the project's
        // concurrency rules forbid.
        let result: Result<String, Error> = await withCheckedContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                let catProc = Process()
                catProc.executableURL = URL(fileURLWithPath: "/bin/cat")
                catProc.arguments = [logPath]

                let claudeProc = Process()
                claudeProc.executableURL = URL(fileURLWithPath: claudePath)
                claudeProc.arguments = ["-p", prompt]
                // Macos app processes launched via LaunchServices get
                // a minimal PATH (often just /usr/bin:/bin), which is
                // not enough for Node-based CLIs that spawn `node`,
                // `git`, etc. Start from the inherited env so any
                // HOME / shell-specific vars stay in place, then
                // override PATH with the locations where the claude
                // CLI and its dependencies typically live. Not using
                // a login shell because we already know the
                // executable path — we just need its dependencies
                // to be findable.
                var env = ProcessInfo.processInfo.environment
                let home = NSString(string: "~").expandingTildeInPath
                let path = [
                    "\(home)/.local/bin",
                    "/opt/homebrew/bin",
                    "/opt/homebrew/sbin",
                    "/usr/local/bin",
                    "/usr/bin",
                    "/bin",
                    "/usr/sbin",
                    "/sbin",
                    env["PATH"] ?? ""
                ]
                .filter { !$0.isEmpty }
                .joined(separator: ":")
                env["PATH"] = path
                claudeProc.environment = env

                let catOut = Pipe()
                catProc.standardOutput = catOut
                claudeProc.standardInput = catOut

                let claudeOut = Pipe()
                let claudeErr = Pipe()
                claudeProc.standardOutput = claudeOut
                claudeProc.standardError = claudeErr

                do {
                    try catProc.run()
                } catch {
                    // cat failed to launch — nothing else to
                    // clean up yet. Surface the error.
                    cont.resume(returning: .failure(error))
                    return
                }

                do {
                    try claudeProc.run()
                } catch {
                    // claude failed to launch. `catProc` is
                    // already running and will fill the pipe
                    // buffer to block on write with no reader.
                    // Terminate it before returning so we
                    // don't leave a zombie cat blocked on a
                    // write forever.
                    catProc.terminate()
                    catProc.waitUntilExit()
                    cont.resume(returning: .failure(error))
                    return
                }

                // Publish the live claude process to the view
                // model now that it's launched. Assigning
                // earlier risked `cancel() -> terminate()`
                // firing on an un-launched Process, which
                // raises `NSInvalidArgumentException`. The
                // tiny window before this hop lands on main
                // is the "not cancellable yet" period — the
                // cat stage and the fork/exec of claude —
                // which is milliseconds at most.
                DispatchQueue.main.async {
                    self?.activeProcess = claudeProc
                }

                catProc.waitUntilExit()
                claudeProc.waitUntilExit()

                DispatchQueue.main.async {
                    self?.activeProcess = nil
                }

                let outData = claudeOut.fileHandleForReading.readDataToEndOfFile()
                let errData = claudeErr.fileHandleForReading.readDataToEndOfFile()

                if claudeProc.terminationStatus != 0 {
                    let errStr = String(data: errData, encoding: .utf8)?
                        .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                    let detail = errStr.isEmpty ? "no stderr" : errStr
                    let err = NSError(
                        domain: "drewschess.loganalysis",
                        code: Int(claudeProc.terminationStatus),
                        userInfo: [NSLocalizedDescriptionKey:
                            "claude exited with status \(claudeProc.terminationStatus): \(detail)"]
                    )
                    cont.resume(returning: .failure(err))
                    return
                }

                let response = String(data: outData, encoding: .utf8) ?? ""
                cont.resume(returning: .success(response))
            }
        }

        // `Task` inherits the enclosing actor; `runAnalysis` is
        // declared on a `@MainActor` class, so this assignment lands
        // on the main actor without an explicit hop.
        isAnalyzing = false
        analysisTask = nil
        switch result {
        case .success(let response):
            claudeResponse = response
        case .failure(let error):
            errorMessage = error.localizedDescription
        }
    }
}
