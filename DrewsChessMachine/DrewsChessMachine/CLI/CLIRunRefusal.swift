import Foundation

/// A headless training run refused before it trained: a usage or
/// configuration problem the operator fixes and reruns (conflicting start
/// flags, an unknown preset, a `--resume-exact` that cannot continue its
/// checkpoint exactly).
///
/// The runners throw it instead of ending the process where the problem is
/// found. Ending the process there lost the session log's last lines —
/// `SessionLogger.log` is asynchronous and only `shutdown()` drains it, so the
/// `[RESUME]` verdict and the refusal itself could be missing from the log —
/// and it made the decision impossible to test in-process, since the test
/// runner went down with it. Each runner's `runAndExit` catches it, writes
/// `error: <message>` to stderr, logs the refusal, drains the log and exits
/// with status 2: the same status and stderr text the inline exits produced.
struct CLIRunRefusal: LocalizedError, Equatable {
    /// The refusal, without the `error: ` prefix stderr adds.
    let message: String

    var errorDescription: String? { message }
}
