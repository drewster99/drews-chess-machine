import Foundation

/// Resolves the weight checkpoint to use for a UCI session and loads
/// it into a fresh inference network.
///
/// Two entry shapes:
///
/// - `loadExplicit(modelFileURL:)` — caller passed `--model <path>`.
///   The path must name a `.dcmmodel` file (champion or trainer
///   variant; we slice off optimizer velocity if present). Session
///   directories are deliberately NOT accepted here — a `.dcmsession`
///   has two `.dcmmodel` files inside and the user can name whichever
///   they want directly.
///
/// - `loadDefault()` — no `--model` flag. Resolves to the most
///   recently saved session via `LastSessionPointer`, then loads
///   that session's `trainer.dcmmodel` (the trainee, which is
///   generally the most up-to-date weights). Hard-errors if no
///   pointer exists or the pointed-to directory is missing — the
///   caller is the UCI pre-flight, which prints the error and exits.
enum UCIModelLoader {

    enum LoadError: Swift.Error, CustomStringConvertible {
        case noSavedSession
        case sessionDirectoryMissing(URL)
        case modelFileMissing(URL)
        case explicitMustBeDcmmodel(URL)
        case weightCountTooSmall(have: Int, need: Int)
        case ambiguousModelName(String, [String])
        case modelNameNotFound(String, URL)

        var description: String {
            switch self {
            case .noSavedSession:
                return "No saved session found. Use --model <path.dcmmodel> or save a session first."
            case .sessionDirectoryMissing(let url):
                return "The most recent session directory is missing on disk: \(url.path)"
            case .modelFileMissing(let url):
                return "Model file does not exist: \(url.path)"
            case .explicitMustBeDcmmodel(let url):
                return "--model expects a .safetensors or .dcmmodel file; got: \(url.path) (session directories are not accepted)"
            case .weightCountTooSmall(let have, let need):
                return "Model file has \(have) tensors but the network needs at least \(need)"
            case .ambiguousModelName(let name, let candidates):
                return "Model name '\(name)' is ambiguous — matches: \(candidates.joined(separator: ", ")). Use a full filename or a more specific run name."
            case .modelNameNotFound(let name, let dir):
                return "No model named '\(name)' in \(dir.path). Pass a full path, a filename (with or without extension), or a run name whose '<name>-replay-latest.safetensors' exists."
            }
        }
    }

    /// Description of a successfully loaded model — used by the UCI
    /// engine to populate the `id name` handshake line and the
    /// session-log banner.
    struct Loaded {
        let network: ChessMPSNetwork
        let modelID: String
        let sourceLabel: String
        /// Total trainable parameter count of the loaded architecture.
        let parameterCount: Int
        /// One-line human-readable architecture description (blocks, channels,
        /// kernels, SE, heads) — surfaced to the UCI GUI as an `info string`.
        let archSummary: String
        /// Absolute path the model resolved to — used to make `setoption Model`
        /// idempotent (skip a redundant reload when a GUI re-sends the same value).
        let resolvedPath: String
    }

    /// Resolve and load using the precedence: explicit `--model` if
    /// given, otherwise the latest session's trainer file.
    static func resolveAndLoad(explicitPath: String?) async throws -> Loaded {
        if let path = explicitPath {
            return try await loadExplicit(path: path)
        }
        return try await loadDefault()
    }

    /// Resolve the weight-file URL to use, WITHOUT loading it, using the
    /// same precedence as `resolveAndLoad`: explicit `--model <path>` if
    /// given, otherwise the most recently saved session's trainer file.
    /// Shared by the UCI loader (which then loads it) and the
    /// `--playchess` launch path (which hands the URL straight to the
    /// human-play `.loadedFile` opponent). Returns the URL plus a short
    /// human-readable source label for logging. Throws the same
    /// `LoadError`s as the load paths so the caller can surface one
    /// message regardless of resolution vs. load failure.
    static func resolveModelURL(explicitPath: String?) throws -> (url: URL, sourceLabel: String) {
        if let ref = explicitPath {
            let expanded = (ref as NSString).expandingTildeInPath
            // A reference is treated as a filesystem PATH if it contains a path
            // separator, is tilde-rooted, or already exists on disk; otherwise
            // it's a bare NAME resolved against the Models directory.
            let looksLikePath = ref.contains("/") || ref.hasPrefix("~")
                || FileManager.default.fileExists(atPath: expanded)
            if looksLikePath {
                let url = URL(fileURLWithPath: expanded)
                let ext = url.pathExtension.lowercased()
                guard ext == "safetensors" || ext == "dcmmodel" else {
                    throw LoadError.explicitMustBeDcmmodel(url)
                }
                guard FileManager.default.fileExists(atPath: url.path) else {
                    throw LoadError.modelFileMissing(url)
                }
                return (url, "--model \(url.path)")
            }
            // Bare name → resolve within Models/ (exact filename, +extension, or
            // run-name→latest). See resolveModelName for the precedence.
            let modelsDir = CheckpointPaths.modelsDir
            let files = (try? FileManager.default.contentsOfDirectory(atPath: modelsDir.path)) ?? []
            switch resolveModelName(ref, among: files) {
            case .found(let filename):
                if filename.hasSuffix("-replay-latest.safetensors") {
                    SessionLogger.shared.log("[UCI] Model '\(ref)' → \(filename): this is a LIVE 'replay-latest' checkpoint the trainer overwrites — it can change mid-run. Use a frozen '-stepNNNN' file for a reproducible result.")
                }
                return (modelsDir.appendingPathComponent(filename), "Model \(filename)")
            case .ambiguous(let candidates):
                throw LoadError.ambiguousModelName(ref, candidates)
            case .notFound:
                throw LoadError.modelNameNotFound(ref, modelsDir)
            }
        }
        guard let pointer = LastSessionPointer.read() else {
            throw LoadError.noSavedSession
        }
        let dir = pointer.directoryURL
        guard pointer.directoryExists else {
            throw LoadError.sessionDirectoryMissing(dir)
        }
        let trainerURL = SessionCheckpointLayout.existingTrainerURL(in: dir)
        guard FileManager.default.fileExists(atPath: trainerURL.path) else {
            throw LoadError.modelFileMissing(trainerURL)
        }
        return (trainerURL, "session \(pointer.sessionID) trainer")
    }

    /// Result of resolving a bare model name against a directory listing.
    enum NameResolution: Equatable {
        case found(String)        // matching filename (not a path)
        case ambiguous([String])  // several run-latest files matched
        case notFound
    }

    /// Resolve a bare model NAME against `files` (a Models-directory listing),
    /// in precedence order:
    ///   1. exact filename            e.g. `20260704-Qeu8e5-replay-latest.safetensors`
    ///   2. filename + extension      e.g. `…-step14000` → `…-step14000.safetensors` / `.dcmmodel`
    ///   3. run name → latest         e.g. `Qeu8e` → `<date>-Qeu8e-replay-latest.safetensors`
    /// The run-name match is anchored so the run token is bounded by a leading
    /// `-` (or is the whole prefix), which keeps `Qeu8e` distinct from `Qeu8e5`.
    /// Pure (no filesystem) so it is unit-testable. Returns the filename only.
    static func resolveModelName(_ name: String, among files: [String]) -> NameResolution {
        let fileSet = Set(files)
        // 1. exact filename (full name incl. extension, or a folder name)
        if fileSet.contains(name) { return .found(name) }
        // 2. filename without extension
        for ext in ["safetensors", "dcmmodel"] where fileSet.contains("\(name).\(ext)") {
            return .found("\(name).\(ext)")
        }
        // 3. run name → latest checkpoint
        let latestSuffix = "-replay-latest.safetensors"
        let matches = files.filter { file in
            guard file.hasSuffix(latestSuffix) else { return false }
            let runToken = String(file.dropLast(latestSuffix.count))  // e.g. "20260704-Qeu8e5"
            return runToken == name || runToken.hasSuffix("-\(name)")
        }
        let unique = Array(Set(matches)).sorted()
        if unique.count == 1 { return .found(unique[0]) }
        if unique.count > 1 { return .ambiguous(unique) }
        return .notFound
    }

    private static func loadExplicit(path: String) async throws -> Loaded {
        let (url, label) = try resolveModelURL(explicitPath: path)
        let file = try CheckpointManager.loadModelFile(at: url)
        let network = try await buildAndLoad(weights: file.weights, arch: file.architecture)
        return Loaded(network: network, modelID: file.modelID, sourceLabel: label,
                      parameterCount: file.architecture.parameterCount,
                      archSummary: file.architecture.architectureSummary,
                      resolvedPath: url.path)
    }

    private static func loadDefault() async throws -> Loaded {
        let (url, label) = try resolveModelURL(explicitPath: nil)
        let file = try CheckpointManager.loadModelFile(at: url)
        let network = try await buildAndLoad(weights: file.weights, arch: file.architecture)
        return Loaded(network: network, modelID: file.modelID, sourceLabel: label,
                      parameterCount: file.architecture.parameterCount,
                      archSummary: file.architecture.architectureSummary,
                      resolvedPath: url.path)
    }

    /// Build a fresh `ChessMPSNetwork` (inference BN mode) and load the
    /// supplied weight tensors into it. Trainer-file weights include
    /// optimizer velocity buffers after the base trainables + BN
    /// running stats; UCI inference only needs the base block, so we
    /// take the leading prefix of length
    /// `trainableVariables.count + bnRunningStatsVariables.count`.
    /// Champion-file weights are exactly that length already, so the
    /// prefix is a no-op for them.
    private static func buildAndLoad(weights: [[Float]], arch: NetworkArchitecture) async throws -> ChessMPSNetwork {
        let network = try ChessMPSNetwork(.randomWeights, arch: arch)
        let baseCount = network.network.trainableVariables.count
            + network.network.bnRunningStatsVariables.count
        guard weights.count >= baseCount else {
            throw LoadError.weightCountTooSmall(have: weights.count, need: baseCount)
        }
        let baseWeights = Array(weights.prefix(baseCount))
        try await network.network.loadWeights(baseWeights)
        return network
    }
}
