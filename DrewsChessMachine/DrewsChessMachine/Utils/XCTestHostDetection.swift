import Foundation

/// Whether this process is an XCTest host: the app launched by xctest to run
/// the `DrewsChessMachineTests` bundle inside it.
///
/// The test target is hosted by the app (`TEST_HOST`), so every test run goes
/// through the app's whole launch path, and several launch-time behaviors must
/// differ there or a test run reaches into the user's real state: the strict
/// command-line parser would reject xctest's own arguments and tear the run
/// down, the orphan-staging sweep would clean the user's real `Sessions/` and
/// `Models/` folders, the auto-resume sheet would offer to resume the user's
/// last real session, and the session logger would write into the user's real
/// `~/Library/Logs/DrewsChessMachine/` folder. Those checks used to read the
/// environment inline each in its own way, and the logger never read it at
/// all — which is how test logs ended up among the real sessions. Every one of
/// them now reads this, so the signal has one definition.
///
/// `XCTestConfigurationFilePath` is set by xctest in the environment of the
/// process it launches to host the bundle, and is absent from every normal app
/// launch and every command-line run. It is read once: the environment of a
/// running process does not gain or lose it.
enum XCTestHostDetection {
    /// True only in a process xctest launched to host the test bundle.
    static let isRunningUnderXCTest: Bool =
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
}
