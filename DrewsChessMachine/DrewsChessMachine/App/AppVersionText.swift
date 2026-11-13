//
//  AppVersionText.swift
//
//  The app's version and build as people read them: the About panel and the
//  Lichess Bot window. The marketing version and bundle version come from
//  Info.plist (Xcode's `MARKETING_VERSION` / `CURRENT_PROJECT_VERSION`); the
//  build counter, commit and date come from `BuildInfo`, the one source of
//  the build's identity. The bundle version alone doesn't identify a build —
//  it moves only when someone edits the project — so every place that shows
//  a version also shows the build counter.
//

import Foundation

enum AppVersionText {

    /// The app's name: the process name, which is the bundle's executable
    /// name and always present.
    static var appName: String {
        ProcessInfo.processInfo.processName
    }

    /// "1.0.1 (4)": marketing version and bundle version, as the standard
    /// About panel prints them. A key missing from Info.plist is named in the
    /// text rather than replaced by a made-up number.
    static var marketingVersionWithBundleVersion: String {
        let marketing = bundleString("CFBundleShortVersionString") ?? "(CFBundleShortVersionString missing)"
        let bundle = bundleString("CFBundleVersion") ?? "(CFBundleVersion missing)"
        return "\(marketing) (\(bundle))"
    }

    /// "Build 2663 · b1ed8b52 · 2026-11-09 · Release". A `*` after the commit
    /// means the compiled project differed from that commit (`BuildInfo.gitDirty`),
    /// the same marker as the `[APP]` launch line.
    static var buildLine: String {
        let dirtyMarker = BuildInfo.gitDirty ? "*" : ""
        return "Build \(BuildInfo.buildNumber) · \(BuildInfo.gitHash)\(dirtyMarker)"
            + " · \(BuildInfo.buildDate) · \(BuildInfo.configuration)"
    }

    private static func bundleString(_ key: String) -> String? {
        guard let value = Bundle.main.object(forInfoDictionaryKey: key) as? String,
              !value.isEmpty
        else { return nil }
        return value
    }
}
