//
//  CommandLineVersion.swift
//
//  `--version`: which build this binary is, printed before anything else
//  runs. The launcher scripts (`run_*.sh`) call it before exec'ing the app,
//  so every launch says which build — counter, commit, configuration, build
//  time — it is about to run; the `[APP] launched` line says the same thing,
//  but only inside the session log, after the launch.
//

import Foundation

enum CommandLineVersion {

    static let flag = "--version"

    /// The build's identity on one line, from `BuildInfo` (the one source of
    /// it): build counter, commit with the `[APP]` banner's `*` dirty marker,
    /// branch, configuration, build timestamp and toolchain.
    static var versionLine: String {
        let dirtyMarker = BuildInfo.gitDirty ? "*" : ""
        return "DrewsChessMachine build \(BuildInfo.buildNumber) git=\(BuildInfo.gitHash)\(dirtyMarker)"
            + " branch=\(BuildInfo.gitBranch) configuration=\(BuildInfo.configuration)"
            + " built=\(BuildInfo.buildTimestamp) xcode=\(BuildInfo.xcodeBuild) sdk=\(BuildInfo.sdkBuild)"
    }

    /// When `rawArgs` hold `--version`, prints the version line to stdout and
    /// exits 0 — never returns. `--version` must be the only argument: beside a
    /// mode it would be ambiguous whether the mode should run, so that is a
    /// usage error (exit 2) rather than a silent choice. Called right after
    /// the help check, before any GUI, GPU or log setup.
    static func handleIfRequested(rawArgs: [String]) {
        guard rawArgs.contains(flag) else { return }
        guard rawArgs == [flag] else {
            FileHandle.standardError.write(Data("DrewsChessMachine: error: \(flag) takes no other arguments\n".utf8))
            Darwin._exit(2)
        }
        FileHandle.standardOutput.write(Data("\(versionLine)\n".utf8))
        Darwin.exit(0)
    }
}
