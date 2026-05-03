import AppKit
import Foundation
import SwiftUI

// MARK: - Window Controller

/// Standalone window that opens the current session log and, in
/// parallel, runs `cat <log> | ~/.local/bin/claude -p "..."` to
/// generate an analysis. Top pane shows the raw log in a
/// monospaced font; bottom pane shows Claude's markdown-formatted
/// response as it arrives. Each analysis window is independent —
/// a new one is built every time the user selects Debug > Analyze
/// Log so a long-running analysis doesn't block a second pass.
@MainActor
final class LogAnalysisWindowController: NSWindowController, NSWindowDelegate {
    /// The view model is owned here (rather than only inside the
    /// SwiftUI view tree) so we can reach in on window-close and
    /// terminate the running `claude` subprocess. Otherwise a
    /// user closing the window mid-analysis would leave a zombie
    /// process until it finished on its own.
    private let viewModel: LogAnalysisViewModel

    init(logPath: String, logContent: String, claudePath: String) {
        let vm = LogAnalysisViewModel(
            logPath: logPath,
            logContent: logContent,
            claudePath: claudePath
        )
        self.viewModel = vm
        let view = LogAnalysisView(viewModel: vm)
        let hosting = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: hosting)
        window.setContentSize(NSSize(width: 900, height: 800))
        window.minSize = NSSize(width: 500, height: 400)
        window.title = "Log Analysis — \((logPath as NSString).lastPathComponent)"
        window.isReleasedWhenClosed = false
        window.center()
        super.init(window: window)
        window.delegate = self
        vm.start()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) not supported for LogAnalysisWindowController")
    }

    func windowWillClose(_ notification: Notification) {
        viewModel.cancel()
        LogAnalysisWindowRegistry.shared.unregister(self)
    }
}

// MARK: - Window Registry

/// Keeps `LogAnalysisWindowController` instances alive for as long
/// as their window is open. Without this a menu-driven
/// `showWindow(nil)` call would leave the controller unretained
/// and let ARC tear it down the moment the launching closure
/// returned, taking the window with it.
@MainActor
final class LogAnalysisWindowRegistry {
    static let shared = LogAnalysisWindowRegistry()
    private var controllers: [LogAnalysisWindowController] = []

    private init() {}

    func register(_ controller: LogAnalysisWindowController) {
        controllers.append(controller)
    }

    func unregister(_ controller: LogAnalysisWindowController) {
        controllers.removeAll { $0 === controller }
    }
}

// MARK: - Analysis View Model

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

// MARK: - Analysis View

struct LogAnalysisView: View {
    @ObservedObject var viewModel: LogAnalysisViewModel

    var body: some View {
        VSplitView {
            logPane
            analysisPane
        }
    }

    private var logPane: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Session log · \((viewModel.logPath as NSString).lastPathComponent)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.horizontal, 8)
            .padding(.top, 6)
            .padding(.bottom, 2)
            ScrollView {
                Text(viewModel.logContent)
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
            }
        }
        .frame(minHeight: 150)
    }

    private var analysisPane: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 6) {
                Text("Claude analysis")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if viewModel.isAnalyzing {
                    ProgressView()
                        .controlSize(.small)
                    Text("Analyzing…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 8)
            .padding(.top, 6)
            .padding(.bottom, 2)
            ScrollView {
                Group {
                    if let err = viewModel.errorMessage {
                        Text("Error: \(err)")
                            .font(.body)
                            .foregroundStyle(.red)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(8)
                    } else if !viewModel.claudeResponse.isEmpty {
                        MarkdownText(content: viewModel.claudeResponse)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(8)
                    } else if viewModel.isAnalyzing {
                        Text("Waiting for Claude to finish…")
                            .foregroundStyle(.secondary)
                            .padding(8)
                    } else {
                        Text("No response.")
                            .foregroundStyle(.secondary)
                            .padding(8)
                    }
                }
            }
        }
        .frame(minHeight: 150)
    }
}

// MARK: - Markdown Renderer

/// Lightweight block-level Markdown view. Supports headings (`#`..
/// `######`), unordered lists (`- ` / `* `), ordered lists
/// (`1. `), fenced code blocks (```` ``` ````), and paragraphs —
/// which is roughly the surface area of a typical `claude -p`
/// response. Inline formatting (bold, italic, `code`, links) is
/// delegated to `AttributedString(markdown:)`, which SwiftUI's
/// `Text` renders natively. Block types outside this list
/// (tables, blockquotes, horizontal rules) fall through as
/// paragraphs with their raw markers visible — acceptable for
/// a debug panel.
struct MarkdownText: View {
    let content: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                render(block)
            }
        }
    }

    private var blocks: [MarkdownBlock] {
        MarkdownBlock.parse(content)
    }

    @ViewBuilder
    private func render(_ block: MarkdownBlock) -> some View {
        switch block {
        case .heading(let level, let text):
            Text(inline(text))
                .font(.system(size: Self.headingSize(level), weight: .bold))
                .padding(.top, level <= 2 ? 8 : 4)
                .padding(.bottom, 2)
        case .paragraph(let text):
            Text(inline(text))
                .fixedSize(horizontal: false, vertical: true)
        case .list(let items, let ordered):
            VStack(alignment: .leading, spacing: 3) {
                ForEach(Array(items.enumerated()), id: \.offset) { idx, item in
                    HStack(alignment: .top, spacing: 6) {
                        Text(ordered ? "\(idx + 1)." : "•")
                            .monospacedDigit()
                            .frame(width: 22, alignment: .trailing)
                        Text(inline(item))
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        case .codeBlock(let code):
            Text(code)
                .font(.system(.body, design: .monospaced))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(8)
                .background(Color.secondary.opacity(0.15))
                .cornerRadius(4)
        }
    }

    /// Inline-only Markdown parse for a single block of text.
    /// `.inlineOnlyPreservingWhitespace` keeps literal spacing in
    /// the caller's string (important for multi-line paragraphs)
    /// while still rendering `**bold**`, `*italic*`,
    /// `` `code` ``, and `[links](url)`. On parser rejection
    /// (unbalanced delimiters, unusual escape sequences) we fall
    /// through to a plain-text `AttributedString` rather than
    /// dropping the text — the reader should always see what
    /// Claude produced, even if it's not styled.
    private func inline(_ text: String) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(
            interpretedSyntax: .inlineOnlyPreservingWhitespace
        )
        do {
            return try AttributedString(markdown: text, options: options)
        } catch {
            return AttributedString(text)
        }
    }

    private static func headingSize(_ level: Int) -> CGFloat {
        switch level {
        case 1: return 22
        case 2: return 19
        case 3: return 16
        case 4: return 14
        default: return 13
        }
    }
}

// MARK: - Markdown Block Parser

enum MarkdownBlock {
    case heading(level: Int, text: String)
    case paragraph(String)
    case list(items: [String], ordered: Bool)
    case codeBlock(String)

    static func parse(_ content: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        let lines = content.components(separatedBy: "\n")
        var i = 0
        while i < lines.count {
            let line = lines[i]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // Fenced code block — consume until the matching closing
            // fence. Unterminated fences swallow the rest of the
            // document, which matches GitHub-flavored markdown.
            if trimmed.hasPrefix("```") {
                i += 1
                var code = ""
                while i < lines.count,
                      !lines[i].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                    code += lines[i] + "\n"
                    i += 1
                }
                if i < lines.count { i += 1 } // consume closing fence
                blocks.append(.codeBlock(
                    code.trimmingCharacters(in: CharacterSet.newlines)
                ))
                continue
            }

            // Blank line — paragraph separator.
            if trimmed.isEmpty {
                i += 1
                continue
            }

            // ATX heading (`#` through `######` followed by a space).
            if let level = headingLevel(trimmed: trimmed) {
                let headerStart = trimmed.index(trimmed.startIndex, offsetBy: level + 1)
                let headerText = String(trimmed[headerStart...])
                blocks.append(.heading(level: level, text: headerText))
                i += 1
                continue
            }

            // Unordered list.
            if trimmed.hasPrefix("- ") || trimmed.hasPrefix("* ") {
                var items: [String] = []
                while i < lines.count {
                    let t = lines[i].trimmingCharacters(in: .whitespaces)
                    if t.hasPrefix("- ") || t.hasPrefix("* ") {
                        items.append(String(t.dropFirst(2)))
                        i += 1
                    } else {
                        break
                    }
                }
                blocks.append(.list(items: items, ordered: false))
                continue
            }

            // Ordered list (`N. text`).
            if orderedListItem(trimmed: trimmed) != nil {
                var items: [String] = []
                while i < lines.count {
                    let t = lines[i].trimmingCharacters(in: .whitespaces)
                    if let rest = orderedListItem(trimmed: t) {
                        items.append(rest)
                        i += 1
                    } else {
                        break
                    }
                }
                blocks.append(.list(items: items, ordered: true))
                continue
            }

            // Default: paragraph. Collect consecutive non-blank,
            // non-special lines into a single paragraph string so
            // inline markdown can span multiple source lines.
            var para = line
            i += 1
            while i < lines.count {
                let next = lines[i]
                let nextTrim = next.trimmingCharacters(in: .whitespaces)
                if nextTrim.isEmpty { break }
                if nextTrim.hasPrefix("#") || nextTrim.hasPrefix("```")
                    || nextTrim.hasPrefix("- ") || nextTrim.hasPrefix("* ") {
                    break
                }
                if orderedListItem(trimmed: nextTrim) != nil { break }
                para += "\n" + next
                i += 1
            }
            blocks.append(.paragraph(para))
        }
        return blocks
    }

    /// ATX-style heading detector. Returns `1…6` if `trimmed`
    /// begins with that many `#` followed by a space; `nil`
    /// otherwise. Bare `#######` (7+) is not a heading per CommonMark.
    private static func headingLevel(trimmed: String) -> Int? {
        var count = 0
        for ch in trimmed {
            if ch == "#" {
                count += 1
                if count > 6 { return nil }
            } else if ch == " " {
                return count > 0 ? count : nil
            } else {
                return nil
            }
        }
        return nil // hashes-only line isn't a heading
    }

    /// Parse a single ordered-list item line (`N. text`). Returns
    /// the text after `N. ` or `nil` if the line isn't one.
    private static func orderedListItem(trimmed: String) -> String? {
        var idx = trimmed.startIndex
        var hasDigit = false
        while idx < trimmed.endIndex, trimmed[idx].isNumber {
            hasDigit = true
            idx = trimmed.index(after: idx)
        }
        guard hasDigit, idx < trimmed.endIndex, trimmed[idx] == "." else { return nil }
        let afterDot = trimmed.index(after: idx)
        guard afterDot < trimmed.endIndex, trimmed[afterDot] == " " else { return nil }
        let textStart = trimmed.index(after: afterDot)
        return String(trimmed[textStart...])
    }
}

// MARK: - Menu Launcher

/// Bridges the SwiftUI Debug menu button to the window launcher.
/// Handles the three preconditions (claude binary exists and is
/// executable, there's an active log, the log can be read) and
/// surfaces a clear NSAlert for any failure instead of silently
/// doing nothing.
@MainActor
enum LogAnalysisLauncher {
    static func openWindow() {
        let home = NSString(string: "~").expandingTildeInPath
        let claudePath = "\(home)/.local/bin/claude"
        let fm = FileManager.default

        guard fm.fileExists(atPath: claudePath),
              fm.isExecutableFile(atPath: claudePath) else {
            presentAlert(
                title: "Claude CLI not found",
                info: "~/.local/bin/claude does not exist or is not executable."
            )
            return
        }

        guard let logPath = SessionLogger.shared.activeLogPath else {
            presentAlert(
                title: "No active session log",
                info: "The session logger has not opened a file. Start the app session logging and try again."
            )
            return
        }

        let logContent: String
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: logPath))
            logContent = String(data: data, encoding: .utf8) ?? ""
        } catch {
            presentAlert(
                title: "Could not read session log",
                info: "\(logPath): \(error.localizedDescription)"
            )
            return
        }

        let controller = LogAnalysisWindowController(
            logPath: logPath,
            logContent: logContent,
            claudePath: claudePath
        )
        LogAnalysisWindowRegistry.shared.register(controller)
        controller.showWindow(nil)
        controller.window?.makeKeyAndOrderFront(nil)
    }

    private static func presentAlert(title: String, info: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = info
        alert.alertStyle = .warning
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }
}
