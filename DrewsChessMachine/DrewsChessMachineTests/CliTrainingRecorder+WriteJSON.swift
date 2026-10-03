//
//  CliTrainingRecorder+WriteJSON.swift
//  DrewsChessMachineTests
//
//  The tests' unchecked results writer. Runs write their results through
//  `write(to:totalTrainingSeconds:)` with a `CliResultsOutput` checked before
//  training, which never replaces a file the run was not told it may
//  replace; this one replaces whatever regular file is at `url`, which is
//  what a test writing into its own temporary folder wants and what no run
//  may do, so it lives in the test target.
//

import Foundation
@testable import DrewsChessMachine

extension CliTrainingRecorder {
    /// Encode the results and write them to `url`, replacing a regular file
    /// already there (never a folder or link), with no ownership check.
    func writeJSON(to url: URL, totalTrainingSeconds: Double) throws {
        let data = try encodedJSONData(totalTrainingSeconds: totalTrainingSeconds)
        try FileSafety.replaceRegularFile(data, at: url, expectedIdentity: nil)
    }
}
