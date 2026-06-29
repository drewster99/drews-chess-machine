//
//  SlowTestGate.swift
//  DrewsChessMachineTests
//
//  Opt-in gate for the heavy Metal/MPSGraph *forensic* suites.
//
//  A handful of suites exist to localize precision/accumulation bugs and to
//  refute specific hypotheses (the macOS-27 NaN isolation matrix, the
//  conv-kernel execution-path numerics sweep). They build and run full
//  multi-step trainer loops across precision × batch × step-count cells, so
//  they dominate test execution time — the NaN-isolation matrix alone is
//  ~11.6 minutes, about 60% of the whole suite's ~19 min run time. They are
//  diagnostic
//  forensics, NOT core-correctness gates: the bugs they chase are already
//  understood, and the cells are kept as a living bug-map.
//
//  Everyday runs should stay fast, so these suites are SKIPPED by default and
//  run only when `DCM_RUN_SLOW_TESTS` is set (`1` or `true`) in the scheme's
//  Test-action environment (Product ▸ Scheme ▸ Edit Scheme… ▸ Test ▸
//  Arguments ▸ Environment Variables), or in the shell that drives
//  `xcodebuild test`:
//
//      DCM_RUN_SLOW_TESTS=1 xcodebuild test -scheme DrewsChessMachine ...
//
//  Gate a suite by calling `try SlowTestGate.requireEnabled("<label>")` from
//  its `setUpWithError()` — that skips the whole class cleanly (the skip
//  reason names the env var). See `DrewsChessMachine/CLAUDE.md` ("Running the
//  tests") for the suite-by-suite cost table and which suites are gated.
//

import XCTest

enum SlowTestGate {

    /// True when the opt-in env var requests the slow forensic suites.
    static var enabled: Bool {
        guard let v = ProcessInfo.processInfo.environment["DCM_RUN_SLOW_TESTS"] else { return false }
        return v == "1" || v.lowercased() == "true"
    }

    /// Skips the calling test/suite unless `DCM_RUN_SLOW_TESTS` is set. Call
    /// from `setUpWithError()` to gate an entire `XCTestCase` subclass.
    static func requireEnabled(_ label: String) throws {
        try XCTSkipUnless(
            enabled,
            "Slow forensic suite '\(label)' skipped (its cells are a diagnostic bug-map, not a correctness gate). Set DCM_RUN_SLOW_TESTS=1 to run it."
        )
    }
}
